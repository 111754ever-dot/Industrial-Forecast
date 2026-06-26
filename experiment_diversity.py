# -*- coding: utf-8 -*-
"""诊断：现有组合还有多少"多样性空间"？加新模型能否改善？

(1) 用现有回测明细算【各模型误差相关性】——相关性越高，再加同族模型越无用；
(2) 实测一个【真正不同角度】的候选：纯当月高频 nowcast(不锚定AR, PLS)，看它是否
    与现有模型去相关、且并入后能降低组合 RMSE。用结果说话。
"""
import numpy as np, pandas as pd
import industrial_va_forecast as M

# ---------- (1) 现有模型误差相关性 ----------
bt = pd.read_csv("output/backtest_predictions.csv", parse_dates=["target_month"])
bt = bt[~bt["is_jan_feb"]].set_index("target_month")   # 只看正常月
models = ["AR", "ARX", "DI", "LightGBM"]
err = pd.DataFrame({m: bt[f"{m}__point"] - bt["y_true"] for m in models}).dropna()
print("===== (1) 现有4模型 预测误差相关矩阵 (剔除1-2月, n=%d) =====" % len(err))
print(err.corr().round(2).to_string())
print("\n各模型 RMSE:", {m: round(np.sqrt((err[m]**2).mean()), 3) for m in models})
print("误差两两相关均值:", round(
    err.corr().values[np.triu_indices(len(models), 1)].mean(), 3))

# ---------- (2) 候选：纯当月高频 nowcast(不锚AR, PLS) ----------
cfg = M.Config()
def load_raw(cfg):
    xl = pd.ExcelFile(cfg.excel_path); raw = {}
    for sh in [cfg.sheet_target, cfg.sheet_month, cfg.sheet_tenday,
               cfg.sheet_week, cfg.sheet_day]:
        df = xl.parse(sh); df.columns = [str(c).strip() for c in df.columns]
        dc = df.columns[0]; df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).rename(columns={dc: "date"})
        for c in df.columns:
            if c != "date": df[c] = pd.to_numeric(df[c], errors="coerce")
        raw[sh] = df
    return raw
raw = load_raw(cfg)
inds = M.build_indicator_dict(cfg)
aligner = M.FrequencyAligner(cfg, raw, inds)
ctx = M.Context(cfg, aligner, M.FeatureBuilder(cfg)); M._wire_context_cache(ctx)
full = aligner.build_monthly_panel(pd.Timestamp("2100-01-01"))
cfg.asof_gap_days = M.resolve_asof_gap(
    cfg, raw, M.month_end(full["IVA_yoy"].dropna().index.max() + pd.offsets.MonthBegin(1)))
feat = ctx.realtime_feature_matrix()
ytar = feat["__target__"]

def f_HFnowcast(target_month):
    """纯当月高频 nowcast：仅用【当月可观测高频 _t0】列, PLS(2成分), 不含任何AR滞后。

    与 AR/ARX/DI 的根本区别：不锚定目标自身惯性，只让当月高频说话 -> 误差结构不同。
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    if target_month.month in (1, 2) or target_month not in feat.index:
        return None
    nf = feat.index[~feat.index.month.isin([1, 2])]
    # 仅当月高频(_t0)、且为高频sheet(日/周/旬)来源——当月真正可观测的 nowcast 信号
    hf_t0 = [c for c in feat.columns if c.endswith("_t0")
             and not c.startswith(("y_",)) and "__" not in c]
    test_row = feat.loc[target_month]
    train_idx = nf[(nf < target_month) & ytar.loc[nf].notna()]
    if len(train_idx) > cfg.max_train_months: train_idx = train_idx[-cfg.max_train_months:]
    if len(train_idx) < 36: return None
    ytr = ytar.loc[train_idx]
    min_cov = max(24, len(train_idx)//3)
    cand = [c for c in hf_t0 if pd.notna(test_row[c])
            and feat.loc[train_idx, c].notna().sum() >= min_cov]
    corr = {}
    for c in cand:
        v = feat.loc[train_idx, c]; mk = v.notna() & ytr.notna()
        if mk.sum() >= 24 and v[mk].std() > 1e-9:
            corr[c] = abs(np.corrcoef(v[mk], ytr[mk])[0, 1])
    top = [c for c, _ in sorted(corr.items(), key=lambda kv: kv[1], reverse=True)[:30]]
    if len(top) < 5: return None
    Xtr = feat.loc[train_idx, top]; Xte = feat.loc[[target_month], top]
    med = Xtr.median(); Xtr = Xtr.fillna(med).fillna(0.0); Xte = Xte.fillna(med).fillna(0.0)
    try:
        sc = StandardScaler(); Xs = sc.fit_transform(Xtr.values); Xe = sc.transform(Xte.values)
        pls = PLSRegression(n_components=2); pls.fit(Xs, ytr.values)
        return float(pls.predict(Xe).ravel()[0])
    except Exception:
        return None

# 在与主回测相同的月份上算候选预测
hf_pred = pd.Series({T: f_HFnowcast(T) for T in err.index}, name="HFnowcast")
err2 = err.copy(); err2["HFnowcast"] = hf_pred - bt["y_true"].reindex(err.index)
err2 = err2.dropna()
print("\n===== (2) 候选 HFnowcast(纯当月高频, 不锚AR, PLS) =====")
print("HFnowcast RMSE:", round(np.sqrt((err2["HFnowcast"]**2).mean()), 3),
      " n=", len(err2))
print("与现有模型误差相关:", err2.corr()["HFnowcast"].drop("HFnowcast").round(2).to_dict())

# ---------- (3) 加入候选后，组合RMSE是否改善？(逆MSE^2加权, 同主程序口径) ----------
def ens_rmse(cols):
    sub = bt.loc[err2.index]
    preds = {c: (sub[f"{c}__point"] if c in models else hf_pred.reindex(err2.index))
             for c in cols}
    P = pd.DataFrame(preds)
    r = {c: np.sqrt(((P[c]-sub["y_true"])**2).mean()) for c in cols}
    w = {c: 1.0/max(r[c], 1e-6)**cfg.ensemble_weight_power for c in cols}
    s = sum(w.values()); w = {c: v/s for c, v in w.items()}
    comb = sum(w[c]*P[c] for c in cols)
    return np.sqrt(((comb - sub["y_true"])**2).mean()), w

r4, w4 = ens_rmse(models)
r5, w5 = ens_rmse(models + ["HFnowcast"])
print("\n===== (3) 组合效果对比 (同一批正常月, n=%d) =====" % len(err2))
print(f"4模型组合(AR+ARX+DI+LightGBM)        RMSE={r4:.3f}")
print(f"5模型组合(+HFnowcast)               RMSE={r5:.3f}  权重={ {k:round(v,3) for k,v in w5.items()} }")
print(f"\n改善: {r4-r5:+.4f} pp  ->", "候选有助" if r5 < r4-1e-3 else "几乎无改善/无助")
