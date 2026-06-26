# -*- coding: utf-8 -*-
"""判决性实验：VAR 模型适不适合预测【工业增加值当月同比】？

用真实时点(vintage)无泄漏回测，让 VAR 与基准 AR、以及主系统的 DI 在同一条件下
比拼实际预测效果，用结果（RMSE / DM 检验 / 最新预测）给出客观结论。

VAR 的现实约束（这正是要检验的核心）：
  VAR 把【目标 + 若干月度宏观】放进同频系统联合预测。在 as_of(T) 时点，月度宏观
  变量与工业增加值一样【当月都尚未发布】，因此诚实的 VAR 只能用到 T-1 的信息，把
  整个向量外推一步(h=1)，【无法利用当月已观测的高频指标】——而 DI/ARX 恰恰用了
  当月高频 nowcast。本实验量化这一差距。

绝不使用任何会泄漏的设定；滞后阶由 BIC 选。
"""
import numpy as np, pandas as pd
import industrial_va_forecast as M

cfg = M.Config()
raw = None

# ---------- 复用主模块装载 raw / 对齐 / 上下文 ----------
def load_raw(cfg):
    xl = pd.ExcelFile(cfg.excel_path)
    raw = {}
    for sh in [cfg.sheet_target, cfg.sheet_month, cfg.sheet_tenday,
               cfg.sheet_week, cfg.sheet_day]:
        df = xl.parse(sh)
        df.columns = [str(c).strip() for c in df.columns]
        dc = df.columns[0]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).rename(columns={dc: "date"})
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        raw[sh] = df
    return raw

if raw is None:
    raw = load_raw(cfg)

inds = M.build_indicator_dict(cfg)
aligner = M.FrequencyAligner(cfg, raw, inds)
ctx = M.Context(cfg, aligner, M.FeatureBuilder(cfg)); M._wire_context_cache(ctx)

# 解析 as_of gap（与主程序一致），保证回测时点同口径
target_live = M.month_end(pd.Timestamp("2100-01-01"))  # 占位
full = aligner.build_monthly_panel(pd.Timestamp("2100-01-01"))
last_realized = full["IVA_yoy"].dropna().index.max()
target_live = M.month_end(last_realized + pd.offsets.MonthBegin(1))
cfg.asof_gap_days = M.resolve_asof_gap(cfg, raw, target_live)
print("as_of gap(天):", cfg.asof_gap_days, " 实盘目标月:", target_live.date())

# ---------- 月度面板(已实现值)：VAR 用的同频变量 ----------
panel = full.copy()
y_full = panel["IVA_yoy"].dropna()
print("目标(当月同比)样本:", y_full.index.min().date(), "->",
      y_full.index.max().date(), " n=", len(y_full))

# 候选月度变量(近似平稳：同比/比率口径)，按覆盖度筛选
CANDIDATE_VARS = ["IVA_yoy", "power_yoy", "ppi_yoy", "retail_yoy",
                  "pmi", "pmi_neworder", "ppirm_yoy"]
CANDIDATE_VARS = [v for v in CANDIDATE_VARS if v in panel.columns]
cov = {v: panel[v].notna().sum() for v in CANDIDATE_VARS}
print("候选变量覆盖度:", cov)
# 取覆盖度 >=120 月且与目标重叠充分的变量
VARS = [v for v in CANDIDATE_VARS if cov[v] >= 120]
if "IVA_yoy" not in VARS:
    VARS = ["IVA_yoy"] + VARS
print("VAR 系统变量:", VARS)

# ---------- 真实时点可见性：as_of(T) 仅见到 T-1 及更早的月度值 ----------
def visible_panel(target_month):
    """as_of(T) 月度宏观与目标均当月未发布 -> 截断到 < T。"""
    T = M.month_end(target_month)
    sub = panel.loc[panel.index < T, VARS].copy()
    return sub

def f_AR(target_month):
    """基准: AR(p) BIC on 可见目标(当月同比)。"""
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    sub = visible_panel(target_month)
    v = sub["IVA_yoy"].dropna()
    if len(v) > cfg.max_train_months:
        v = v.iloc[-cfg.max_train_months:]
    if len(v) < 48:
        return None
    vals = v.values.astype(float)
    try:
        maxlag = int(min(13, max(1, len(vals)//6)))
        try:
            sel = ar_select_order(vals, maxlag=maxlag, ic="bic", old_names=False)
            lags = sel.ar_lags if sel.ar_lags is not None and len(sel.ar_lags) else 1
        except Exception:
            lags = 1
        res = AutoReg(vals, lags=lags, old_names=False).fit()
        return float(np.asarray(res.predict(start=len(vals), end=len(vals)))[0])
    except Exception:
        return None

def f_VAR(target_month, ic="bic"):
    """VAR(p)：同频系统联合估计，h=1 外推，取 IVA 分量。

    缺失处理：对系统做共同样本(各变量取交集时段)后线性插值小缺口；滞后阶由 IC 选。
    """
    from statsmodels.tsa.api import VAR
    sub = visible_panel(target_month)
    # 取所有变量都开始有值之后的连续段
    sub = sub.dropna(how="all")
    start = max(sub[v].first_valid_index() for v in VARS if sub[v].notna().any())
    sub = sub.loc[sub.index >= start, VARS]
    # 小缺口插值(系统内部缺失，时间方向)，仍无前视(只用 <T 的数据)
    sub = sub.interpolate(limit_direction="both").dropna()
    if len(sub) > cfg.max_train_months:
        sub = sub.iloc[-cfg.max_train_months:]
    if len(sub) < 60:
        return None
    try:
        model = VAR(sub.values)
        maxlag = int(min(12, max(1, len(sub)//(5*len(VARS)))))
        if maxlag < 1:
            maxlag = 1
        sel = model.select_order(maxlags=maxlag)
        p = getattr(sel, ic) if isinstance(getattr(sel, ic), int) else None
        if not p or p < 1:
            # select_order 返回的是各IC最优阶；用属性 .bic 等
            try:
                p = int(sel.selected_orders[ic])
            except Exception:
                p = 1
        p = max(1, min(p, maxlag))
        res = model.fit(p)
        fc = res.forecast(sub.values[-p:], steps=1)
        iva_idx = VARS.index("IVA_yoy")
        return float(fc[0, iva_idx])
    except Exception as e:
        return None

def f_DI(target_month):
    """主系统 DI(扩散指数, 用当月高频 nowcast) 作为对照上界。"""
    as_of = M.asof_for_target(cfg, target_month)
    try:
        pr = M.ModelDI(cfg).predict_asof(M.month_end(target_month), as_of, ctx)
        return None if pr is None else float(pr.point)
    except Exception:
        return None

# ---------- 真实时点回测 ----------
ctx.realtime_feature_matrix()  # 预构建(DI 用)
# 回测集：剔除 1-2 月(拆分口径噪声大，主系统也单独处理)，留足训练
months = [m for m in y_full.index
          if m >= pd.Timestamp("2021-03-01") and m.month not in (1, 2)]
print("回测月数(剔除1-2月):", len(months))

rec = []
for T in months:
    rec.append({"month": T, "actual": float(y_full.loc[T]),
                "AR": f_AR(T), "VAR": f_VAR(T), "DI": f_DI(T)})
bt = pd.DataFrame(rec).set_index("month")
import os; os.makedirs("output_var", exist_ok=True)
bt.to_csv("output_var/backtest_var.csv", encoding="utf-8-sig")

def metrics(col):
    e = (bt[col] - bt["actual"]).dropna()
    if len(e) == 0:
        return np.nan, np.nan, np.nan, 0
    return np.sqrt((e**2).mean()), e.abs().mean(), e.mean(), len(e)

print("\n===== 当月同比 口径 真实时点回测 (2021-03+, 剔除1-2月) =====")
print(f"{'model':6s} {'RMSE':>7s} {'MAE':>7s} {'bias':>7s} {'n':>4s}  说明")
notes = {"AR": "基准(仅自身历史)", "VAR": "系统外推(无当月高频)",
         "DI": "扩散指数(用当月高频nowcast)"}
for c in ["AR", "VAR", "DI"]:
    r, a, b, n = metrics(c)
    print(f"{c:6s} {r:7.3f} {a:7.3f} {b:+7.3f} {n:4d}  {notes[c]}")

# DM 检验：VAR vs AR， DI vs AR（同样本对比）
def dm_vs_ar(col):
    pair = bt[["AR", col, "actual"]].dropna()
    if len(pair) < 8:
        return None, None, len(pair)
    dm, p = M.diebold_mariano((pair[col]-pair["actual"]).values,
                              (pair["AR"]-pair["actual"]).values)
    return dm, p, len(pair)

print("\n----- DM 检验(对基准 AR) -----")
for c in ["VAR", "DI"]:
    dm, p, n = dm_vs_ar(c)
    if dm is None:
        print(f"{c} vs AR: 样本不足"); continue
    if np.isfinite(dm) and dm < 0 and p < 0.10:
        verdict = f"{c} 显著【优于】AR(指标/结构有用)"
    elif np.isfinite(dm) and dm > 0 and p < 0.10:
        verdict = f"{c} 显著【差于】AR"
    else:
        verdict = f"{c} 与 AR 无显著差异"
    print(f"{c:4s} vs AR: DM={dm:+.3f} p={p:.3f} n={n} -> {verdict}")

# ---------- 最新预测(实盘目标月) ----------
print(f"\n最新预测月 {target_live.date()}:")
print(f"  AR ={f_AR(target_live)}")
print(f"  VAR={f_VAR(target_live)}")
print(f"  DI ={f_DI(target_live)}")

# ---------- 画图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(bt.index, bt["actual"], "k-o", ms=4, lw=2, label="Actual")
ax.plot(bt.index, bt["AR"], "b--", marker=".", label="AR (self only)")
ax.plot(bt.index, bt["VAR"], "g-.", marker="s", ms=4,
        label="VAR (system, no current-month HF)")
ax.plot(bt.index, bt["DI"], "r:", marker="^", ms=4,
        label="DI (diffusion index, uses current-month HF)")
ax.set_title("Industrial VA (MoM YoY) real-time backtest: VAR vs AR vs DI")
ax.set_ylabel("YoY %"); ax.legend(); ax.grid(alpha=.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig("output_var/backtest_var.png", dpi=130, bbox_inches="tight")
print("\nsaved output_var/backtest_var.png")
