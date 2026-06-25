# -*- coding: utf-8 -*-
"""判决性实验：在【累计同比(YTD)】口径上，高频指标是否优于纯自身AR？

复用主模块的真实时点(vintage)对齐与无泄漏机制；目标替换为 YTD。
所有调参用 TimeSeriesSplit / BIC，绝不使用会泄漏的随机CV。
"""
import numpy as np, pandas as pd
import industrial_va_forecast as M

YTD_FILE = "工业增加值累计同比.xlsx"
YTD_COL = "中国:工业增加值:规模以上工业企业:累计同比"

cfg = M.Config()

# ---------- 1) 构造 raw：主表4个指标sheet + 用YTD替换目标sheet ----------
def load_raw_with_ytd(cfg):
    xl = pd.ExcelFile(cfg.excel_path)
    raw = {}
    for sh in [cfg.sheet_month, cfg.sheet_tenday, cfg.sheet_week, cfg.sheet_day]:
        df = xl.parse(sh)
        df.columns = [str(c).strip() for c in df.columns]
        dc = df.columns[0]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).rename(columns={dc: "date"})
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        raw[sh] = df
    yt = pd.ExcelFile(YTD_FILE).parse("Sheet1")
    yt.columns = ["date", YTD_COL]
    yt["date"] = pd.to_datetime(yt["date"], errors="coerce")
    yt[YTD_COL] = pd.to_numeric(yt[YTD_COL], errors="coerce")
    yt = yt.dropna(subset=["date"]).sort_values("date")
    raw[cfg.sheet_target] = yt
    return raw

raw = load_raw_with_ytd(cfg)

# ---------- 2) 指标字典：目标替换为 YTD（key 仍为 IVA_yoy 以复用机器） ----------
inds = M.build_indicator_dict(cfg)
inds[0] = M.Indicator("IVA_yoy", YTD_COL, cfg.sheet_target, "target", "last",
                      True, "target", "目标改为累计同比(YTD)")

aligner = M.FrequencyAligner(cfg, raw, inds)
ctx = M.Context(cfg, aligner, M.FeatureBuilder(cfg)); M._wire_context_cache(ctx)

# 真实YTD序列(月末索引, 去NaN——YTD天然无1月)
ys = raw[cfg.sheet_target].set_index("date")[YTD_COL]
ys.index = M.to_month_index(ys.index); ys = ys[~ys.index.duplicated()].sort_index().dropna()
print("YTD 样本:", ys.index.min().date(), "->", ys.index.max().date(), " n=", len(ys))

# ---------- 3) 模型(YTD专用, 无1-2月拆分噪声问题) ----------
def visible_ys(target_month):
    """as_of=当月23号时可见的YTD(发布滞后16天 -> 到T-1)。"""
    as_of = M.asof_for_target(cfg, target_month)
    vis = ys[(ys.index + pd.Timedelta(days=cfg.pub_lag_target_days)) <= as_of]
    return vis

def f_AR(target_month):
    """基准: AR(p) BIC on 可见YTD。"""
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    v = visible_ys(target_month)
    if len(v) < 36: return None
    vals = v.values.astype(float)
    try:
        maxlag = int(min(12, max(1, len(vals)//6)))
        try:
            sel = ar_select_order(vals, maxlag=maxlag, ic="bic", old_names=False)
            lags = sel.ar_lags if sel.ar_lags is not None and len(sel.ar_lags) else 1
        except Exception:
            lags = 1
        res = AutoReg(vals, lags=lags, old_names=False).fit()
        return float(np.asarray(res.predict(start=len(vals), end=len(vals)))[0])
    except Exception:
        return None

def f_RW(target_month):
    """对照: 随机游走(上一可见YTD)。"""
    v = visible_ys(target_month)
    return float(v.iloc[-1]) if len(v) else None

def _ytd_design(target_month, n_arlags=3):
    """YTD监督设计: YTD的AR滞后 + 高频(vintage,无泄漏) top-K。"""
    feat = ctx.realtime_feature_matrix()
    if target_month not in feat.index: return None
    # YTD AR 滞后(用真实YTD的连续滞后)
    lag = pd.DataFrame(index=ys.index)
    for L in range(1, n_arlags+1): lag[f"arlag{L}"] = ys.shift(L)
    hf_cols = [c for c in feat.columns if c != "__target__" and not c.startswith("y_")]
    test_row = feat.loc[target_month]
    train_idx = ys.index[ys.index < target_month]
    train_idx = train_idx[train_idx.isin(feat.index)]
    if len(train_idx) > cfg.max_train_months: train_idx = train_idx[-cfg.max_train_months:]
    if len(train_idx) < 36: return None
    min_cov = max(18, len(train_idx)//3)
    ytr = ys.loc[train_idx]
    cand = [c for c in hf_cols if pd.notna(test_row[c])
            and feat.loc[train_idx, c].notna().sum() >= min_cov]
    corr={}
    for c in cand:
        vv=feat.loc[train_idx,c]; m=vv.notna()&ytr.notna()
        if m.sum()>=18 and vv[m].std()>1e-9: corr[c]=abs(np.corrcoef(vv[m],ytr[m])[0,1])
    top=[c for c,_ in sorted(corr.items(),key=lambda kv:kv[1],reverse=True)[:cfg.midas_topk_features]]
    design = pd.concat([feat.loc[:, top], lag], axis=1)
    Xtr_raw=design.loc[train_idx]; Xte_raw=design.loc[[target_month]]
    if Xte_raw[list(lag.columns)].isna().any(axis=1).iloc[0]: return None
    med=Xtr_raw.median(); Xtr=Xtr_raw.fillna(med).fillna(0.0); Xte=Xte_raw.fillna(med).fillna(0.0)
    return Xtr, ytr, Xte

def f_ARX(target_month):
    """AR滞后+高频, ElasticNet, TimeSeriesSplit(无泄漏)。"""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    d=_ytd_design(target_month)
    if d is None: return None
    Xtr,ytr,Xte=d
    try:
        sc=StandardScaler(); Xs=sc.fit_transform(Xtr.values); Xe=sc.transform(Xte.values)
        m=ElasticNetCV(l1_ratio=list(cfg.enet_l1_ratios), cv=TimeSeriesSplit(5),
                       max_iter=50000, n_jobs=-1, random_state=cfg.random_state)
        m.fit(Xs, ytr.values)
        return float(m.predict(Xe)[0])
    except Exception:
        return None

# ---------- 4) 真实时点回测 ----------
months = [m for m in ys.index if m >= pd.Timestamp("2021-01-01")]  # 留足训练
print("回测月数:", len(months))
ctx.realtime_feature_matrix()  # 预构建
rec=[]
for T in months:
    rec.append({"month":T, "actual":float(ys.loc[T]),
                "RW":f_RW(T), "AR":f_AR(T), "ARX":f_ARX(T)})
bt=pd.DataFrame(rec).set_index("month")
bt.to_csv("output_ytd/backtest_ytd.csv", encoding="utf-8-sig")

def metrics(col):
    e=(bt[col]-bt["actual"]).dropna()
    return np.sqrt((e**2).mean()), e.abs().mean(), e.mean(), len(e)

print("\n===== YTD 口径 回测 (2021+) =====")
print(f"{'model':6s} {'RMSE':>7s} {'MAE':>7s} {'bias':>7s} {'n':>4s}")
for c in ["RW","AR","ARX"]:
    r,a,b,n=metrics(c); print(f"{c:6s} {r:7.3f} {a:7.3f} {b:+7.3f} {n:4d}")

# DM: ARX vs AR
pair=bt[["AR","ARX","actual"]].dropna()
dm,p=M.diebold_mariano((pair["ARX"]-pair["actual"]).values,(pair["AR"]-pair["actual"]).values)
print(f"\nDM[ARX vs AR]: DM={dm:+.3f} p={p:.3f} -> "
      + ("ARX显著优于AR(指标有用!)" if (np.isfinite(dm) and dm<0 and p<0.10)
         else "ARX显著差于AR" if (np.isfinite(dm) and dm>0 and p<0.10)
         else "与AR无显著差异(指标无显著增量)"))

# 最新预测
Tnext = M.month_end(ys.index.max() + pd.offsets.MonthBegin(1))
print(f"\n最新预测月(YTD) {Tnext.date()}: AR={f_AR(Tnext)} ARX={f_ARX(Tnext)} RW={f_RW(Tnext)}")

# 画图
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(bt.index, bt["actual"],"k-o",ms=4,lw=2,label="Actual YTD")
ax.plot(bt.index, bt["AR"],"b--",marker=".",label="AR (self only)")
ax.plot(bt.index, bt["ARX"],"r-.",marker=".",label="ARX (+indicators)")
ax.set_title("YTD (cumulative YoY) backtest: AR vs ARX(+indicators)")
ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("output_ytd/backtest_ytd.png", dpi=130, bbox_inches="tight")
print("saved output_ytd/backtest_ytd.png")
