# -*- coding: utf-8 -*-
"""判决性实验：重写的【混频动态因子模型 MF-DFM】是否值得加入组合？

新逻辑（针对旧DFM"系统高估+过拟合"的修正）：
  (a) 用 statsmodels DynamicFactorMQ 的【状态空间 Kalman 滤波】从精选核心序列提取共同
      因子——这是 DFM 真正发挥混频/ragged-edge 优势处(优于 DI 的"中位数填充+PCA")；
  (b) 【不让因子自行外推】，而是把当月平滑因子 f_T 作为回归量、【锚定 AR 惯性】做岭回归
      → 抑制高估；
  (c) 控过拟合：单因子、仅~10条长核心序列、Ridge收缩、滚动窗口、每K月重估EM参数。

用真实时点无泄漏回测，量化其 RMSE、与现有5模型的误差去相关度、并入后组合是否改善。
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import industrial_va_forecast as M

cfg = M.Config()

# 精选核心序列（长、与工业生产直接相关：发电/景气/价格/钢铁/产量/汽车/需求）
DFM_CORE = ["power_yoy", "pmi", "pmi_neworder", "pmi_newexport", "ppi_yoy",
            "retail_yoy", "prod_ic", "prod_power_equip",
            "steel_crude_key", "steel_rolled_key", "car_wholesale"]
DFM_REFIT_EVERY = 6      # 每6个月重估一次EM参数，其余月仅用固定参数滤波（提速、稳健）

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
y_full = full["IVA_yoy"].dropna()

# 缓存：每个 as_of 的 DFM 因子（避免重复滤波）；EM 参数按 refit 周期复用
_param_cache = {"params": None, "anchor_month": None}

def dfm_factor_at(as_of, target_month):
    """用 DynamicFactorMQ 提取【当月平滑因子 f_T】(含目标自身，ragged-edge 由Kalman处理)。"""
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    panel = ctx.aligner.build_monthly_panel(as_of)   # vintage：当月IAV=NaN, 高频=残月
    cols = ["IVA_yoy"] + [c for c in DFM_CORE if c in panel.columns]
    X = panel.loc[panel.index <= M.month_end(target_month), cols].copy()
    # 只保留有足够历史的起点
    X = X.loc[X.dropna(how="all").index.min():]
    if len(X) < 72:
        return None, None
    try:
        mod = DynamicFactorMQ(X, factors=1, factor_orders=1,
                              idiosyncratic_ar1=True, standardize=True)
        # 每 refit 周期重估 EM；其余月用上次参数仅滤波（提速 + 参数稳健）
        ai = _param_cache["anchor_month"]
        reuse = (_param_cache["params"] is not None and ai is not None
                 and 0 <= (target_month.to_period("M") - ai.to_period("M")).n < DFM_REFIT_EVERY)
        if reuse:
            try:
                res = mod.smooth(_param_cache["params"])
            except Exception:
                res = mod.fit(maxiter=60, disp=False)
                _param_cache.update(params=res.params, anchor_month=target_month)
        else:
            res = mod.fit(maxiter=80, disp=False)
            _param_cache.update(params=res.params, anchor_month=target_month)
        fac = np.asarray(res.factors_filtered.iloc[:, 0]
                         if hasattr(res, "factors_filtered") else
                         res.states.smoothed.iloc[:, 0])
        fseries = pd.Series(fac, index=X.index)
        return fseries, fseries.loc[M.month_end(target_month)]
    except Exception as e:
        return None, None

def f_MFDFM(target_month):
    """锚定 MF-DFM：非1-2月 IAV ~ arlag1,2 + 当月平滑因子 f_T，RidgeCV(TimeSeriesSplit)。"""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    if target_month.month in (1, 2):
        ynf = y_full[~y_full.index.month.isin([1, 2])]
        ynf = ynf[ynf.index < target_month]
        return float(ynf.tail(cfg.janfeb_fallback_k).mean()) if len(ynf) >= 6 else None
    as_of = M.asof_for_target(cfg, target_month)
    fseries, fT = dfm_factor_at(as_of, target_month)
    if fseries is None or not np.isfinite(fT):
        return None
    nf = y_full.index[~y_full.index.month.isin([1, 2])]
    ynf = y_full.loc[nf]
    df = pd.DataFrame(index=nf)
    df["arlag1"] = ynf.shift(1); df["arlag2"] = ynf.shift(2)
    df["factor"] = fseries.reindex(nf)
    df["y"] = ynf
    tr = df[df.index < target_month].dropna()
    if len(tr) > cfg.max_train_months: tr = tr.iloc[-cfg.max_train_months:]
    if len(tr) < 48: return None
    te = df.loc[[target_month], ["arlag1", "arlag2", "factor"]]
    if te.isna().any(axis=1).iloc[0]: return None
    try:
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[["arlag1", "arlag2", "factor"]].values)
        Xte = sc.transform(te.values)
        r = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=TimeSeriesSplit(5))
        r.fit(Xtr, tr["y"].values)
        return float(r.predict(Xte)[0])
    except Exception:
        return None

# ---------- 回测：与主程序相同的正常月集合 ----------
bt0 = pd.read_csv("output/backtest_predictions.csv", parse_dates=["target_month"])
bt0 = bt0[~bt0["is_jan_feb"]].set_index("target_month")
months = list(bt0.index)
print("回测正常月数:", len(months), " 区间:", months[0].date(), "->", months[-1].date())

pred = {}
for i, T in enumerate(months, 1):
    pred[T] = f_MFDFM(T)
    if i % 10 == 0: print(f"  进度 {i}/{len(months)}  最新 {T.date()} -> {pred[T]}")
mfdfm = pd.Series(pred, name="MFDFM")

# ---------- 评估 ----------
existing = ["AR", "ARX", "DI", "LightGBM", "PLS"]
err = pd.DataFrame({m: bt0[f"{m}__point"] - bt0["y_true"] for m in existing})
err["MFDFM"] = mfdfm - bt0["y_true"]
err = err.dropna()
print("\n===== 新 MF-DFM 实测 (剔除1-2月, n=%d) =====" % len(err))
rmse = {m: np.sqrt((err[m]**2).mean()) for m in err.columns}
bias = {m: err[m].mean() for m in err.columns}
print("RMSE:", {k: round(v, 3) for k, v in rmse.items()})
print("bias:", {k: round(v, 3) for k, v in bias.items()})
print("\nMFDFM 误差与现有模型相关:",
      err.corr()["MFDFM"].drop("MFDFM").round(2).to_dict())

# 组合对比：现有5模型 vs +MFDFM（逆MSE^2加权，同主程序口径）
sub = bt0.loc[err.index]
def ens(cols):
    P = pd.DataFrame({c: (sub[f"{c}__point"] if c in existing
                          else mfdfm.reindex(err.index)) for c in cols})
    r = {c: np.sqrt(((P[c]-sub["y_true"])**2).mean()) for c in cols}
    w = {c: 1/max(r[c], 1e-6)**cfg.ensemble_weight_power for c in cols}
    s = sum(w.values()); w = {c: v/s for c, v in w.items()}
    comb = sum(w[c]*P[c] for c in cols)
    return np.sqrt(((comb-sub["y_true"])**2).mean()), w
r5, w5 = ens(existing)
r6, w6 = ens(existing + ["MFDFM"])
print("\n===== 组合效果 (n=%d) =====" % len(err))
print(f"现有5模型组合              RMSE={r5:.3f}")
print(f"6模型组合(+MFDFM)         RMSE={r6:.3f}  MFDFM权重={round(w6['MFDFM'],3)}")
print(f"改善: {r5-r6:+.4f} pp ->", "MFDFM 有助" if r6 < r5-1e-3 else "几乎无改善")
# 最新预测
print(f"\n最新月 {months[-1].date()} 之后一期预测 MFDFM:",
      f_MFDFM(M.month_end(y_full.index.max()+pd.offsets.MonthBegin(1))))
