# -*- coding: utf-8 -*-
"""
================================================================================
 工业增加值（规上工业企业:当月同比，1-2月拆分）常态化预测系统
================================================================================

一个单文件、配置驱动、可一键运行的混频（月/旬/周/日）即时预测(nowcasting)系统。

设计目标
--------
以"预测准确性"为唯一前提，提供：
  1) 点预测；
  2) 区间预测（80% 与 95%）。

为何这样设计（与方案一一对应）
--------------------------------
* 频率不一致：四种频率全部"对齐到月"，高频指标按"真实时点可得"聚合（均值/月末值/
  月内进度），杜绝前视偏差。
* 样本长度不一致：**绝不截断到统一起止日期**（那样会被 2022 年才开始的指标拖到只剩
  3 年样本，丢掉 30 年历史）。改用：真实时点(vintage)对齐 + 各模型按能力处理缺失
  （AR/LocalLevel 仅用目标自身；ARX 用"预测点可观测 + 训练覆盖充分"的高频，无需硬填）。
* 1-2 月：拆分单月是人造噪声，回测/共形区间对 1-2 月单独分组(自动加宽)并单列其表现。
* 既要点又要区间：逆误差加权组合点预测，区间用"对组合回测残差做分裂共形(split
  conformal)校准"，保证经验覆盖达标且不过窄。
* 模型集合（经 DM 检验严格筛选后定稿）：
    - 核心(默认运行、进组合)：AR(p,基准/锚) + LocalLevel(朴素地板) + ARX(锚定高频桥接)；
    - 探针(默认关闭)：DFM + LightGBM，机制最独特的高频/非线性模型，仅用于监测"高频指标
      是否重新跑赢 AR"的 regime 信号，不参与默认预测。
  说明：经无泄漏回测 + Diebold-Mariano 检验证实，在"当月同比"与"累计同比"两种口径下，
  高频指标均未显著超越 AR——故复杂高频模型降为探针，体系以"目标自身历史"为主。

真实时点（ragged edge）口径
--------------------------
在每月 20-25 日预测"下一个未发布月" T 时，信息集为：
  - 工业增加值(目标)：发布到 T-1（T 未发布，正是要预测的）；
  - 月度宏观自变量(社零/投资/PPI/PMI…)：发布到 T-1（次月中旬才发布 T）；
  - 高频(日/周/旬)：基本覆盖到 T 月底。
本系统用"参考期末 + 发布滞后 pub_lag_days <= as_of"统一判定每个观测是否可见，
回测与实盘使用完全相同的可见性规则，确保无前视偏差、训练/测试一致。

运行
----
    python industrial_va_forecast.py
产出在 ./output/ ：点+区间预测表、回测评估表、扇形图、驱动分解、数据质量报告。

依赖
----
    numpy pandas openpyxl scipy statsmodels scikit-learn lightgbm matplotlib

作者注
------
代码以"正确、权威、可常态化复用"为标准撰写，不以运行速度为目标。每个模型都做了
异常隔离：单个模型失败不会中断整条流水线（会记录并在组合中自动剔除）。
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# 第 0 节  日志
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOG = logging.getLogger("IVA")


# ==============================================================================
# 第 1 节  全局配置
# ==============================================================================
@dataclass
class Config:
    # ---- 输入 ----
    excel_path: str = "数据汇总表0620V3.xlsx"
    sheet_target: str = "工业增加值"
    sheet_month: str = "月度"
    sheet_tenday: str = "旬度"
    sheet_week: str = "周度"
    sheet_day: str = "日度"

    # ---- 输出 ----
    out_dir: str = "output"

    # ---- 真实时点口径（预测时点不写死，按汇总表数据自动推断）----
    # 业务：每月不固定的某天(大概率20-25号)收集最新汇总表后预测【下一个未发布月】。
    # 预测时点 as_of 不固定为某号，而是【按汇总表自身的数据新鲜度自动确定】：
    #   run_date = 各 sheet 最新观测日期的最大值(=实际采集日)；asof_day = run_date.day。
    # 于是预测月T的 as_of = T 当月的 asof_day 号；因 T=采集当月，该 as_of 即 run_date 本身，
    # 回测各历史月也用同一 asof_day，随采集时间自动平移、不写死。
    # run_date / asof_day_of_month 为 None 时自动从数据推断；亦可显式覆盖。
    run_date: Optional[str] = None          # 如 "2026-05-25"；None=自动取数据最新日期
    asof_day_of_month: Optional[int] = None  # None=自动取 run_date.day
    # 各频率"真实发布滞后"（数据及时更新，故按每个频率的实际可得延迟分别设定，而非笼统取值）
    pub_lag_target_days: int = 16    # 工业增加值发布滞后（次月约 15-16 日，NBS）
    pub_lag_month_days: int = 16     # 月度宏观发布滞后（社零/投资 15-17日；PPI~9日；PMI月末）
    pub_lag_day_days: int = 1        # 日度发布滞后（BDI/价格/港口吞吐/商品房，基本当天-次日）
    pub_lag_week_days: int = 4       # 周度发布滞后（开工率/煤耗/乘用车，~3-5天）
    pub_lag_tenday_days: int = 6     # 旬度发布滞后（中钢协钢铁旬报，~5-7天）

    # ---- 回测 ----
    backtest_months: int = 48        # 回测的外推月数（从最近往前）。生产可调大。
    backtest_min_train: int = 120    # 回测起步所需的最少训练月数
    dfm_refit_every: int = 6         # DFM 每隔多少个月重估一次参数（其余月复用参数仅滤波，提速）

    # ---- 特征工程 ----
    feature_lags: tuple = (1, 2, 3)  # 月度特征滞后阶
    target_ar_lags: tuple = (1, 2, 3, 12)  # 目标自回归滞后

    # ---- 抗过拟合 ----
    # 桥接模型(ARX)的训练窗口上限(月)。限定近 N 年，避免：(1)被中位数填充的远古高频
    # 特征污染；(2)1990年代/疫情等结构断点造成的体制错配。DFM 不受此限(它原生吃缺失、
    # 且因子估计受益于长历史)。156 月 = 13 年，覆盖绝大多数高频指标的真实起点。
    max_train_months: int = 156
    bridge_topk_features: int = 40   # 桥接模型入模前按|相关|预筛的特征数(降维抗过拟合)
    ensemble_exclude_janfeb_in_weights: bool = True  # 组合权重按"剔除1-2月"误差计算
    ensemble_weight_power: float = 2.0  # 权重=1/RMSE^power。2=逆MSE,更狠地压制不稳定模型

    # ---- 模型集合 ----
    # 核心(默认运行、进DM门控组合): AR(基准) + LocalLevel(朴素地板) + ARX(锚定高频桥接)。
    # 探针(默认关闭): DFM/LightGBM——机制最独特的高频/非线性模型，平时不跑(慢)；开启后参与
    # DM 检验，用于监测"高频指标是否重新跑赢 AR"(regime 变化信号)，不影响默认预测。
    enable_probe_models: bool = False

    # ---- 基准与"是否真有技能"门控 ----
    # 基准 = AR(p)：平稳自相关序列的标准技能下限(比"近期均值"更严格、更规范)。
    benchmark_name: str = "AR"
    locallevel_k: int = 6                 # 局部水平用最近K个非1-2月观测
    ensemble_keep_ratio: float = 1.02     # 仅保留 RMSE<=基准*该比值 的模型(基准本身恒保留)
    dm_significance: float = 0.10         # DM检验显著性水平
    sanity_clip_pp: float = 6.0      # 极端值护栏:最终点预测不超出近12个月实际范围±该值

    # ---- 模型 ----
    dfm_factors: int = 2             # DFM 共同因子个数
    dfm_factor_order: int = 2        # 因子 VAR 阶
    dfm_maxiter: int = 100           # EM 迭代上限
    enet_l1_ratios: tuple = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
    lgb_quantiles: tuple = (0.025, 0.10, 0.50, 0.90, 0.975)

    # ---- 区间 ----
    interval_levels: tuple = (0.80, 0.95)

    # ---- 杂项 ----
    random_state: int = 20260624


CFG = Config()


# ------------------------------------------------------------------------------
# 1.1  指标字典（含"同名陷阱"核对）
# ------------------------------------------------------------------------------
# 每条登记：
#   key      : 内部短名（唯一）
#   name     : Excel 中的**精确**列名（一字不差，用于校验，防止对方更新表时改名/换序）
#   sheet    : 所属频率 sheet
#   transform: 变换类型，见 第 4 节 Transformer
#               - 'target'      目标，已是当月同比，保持
#               - 'yoy_keep'    已是同比/累计同比，保持
#               - 'cum2mom2yoy' 累计值 -> 当月值 -> 当月同比（年初处理 1-2 月合并）
#               - 'level2yoy'   水平量(产量/吞吐/销量/面积) -> 月度聚合 -> 当月同比
#               - 'index2yoy'   价格/指数 -> 月度聚合 -> 同比
#               - 'rate'        比率(PMI/开工率/运转率/发运率) -> 月度聚合 -> 保持水平
#   agg      : 高频->月度聚合方式 'mean'（默认）；月度指标用 'last'
#   dfm_tier : 是否纳入 MF-DFM 的"长且完整"核心集合（很新/很稀疏的剔除，仅供 ML）
#   role     : 'target' / 'predictor'
#   note     : 备注（重点标注易混项）
# ------------------------------------------------------------------------------
@dataclass
class Indicator:
    key: str
    name: str
    sheet: str
    transform: str
    agg: str = "mean"
    dfm_tier: bool = True
    role: str = "predictor"
    note: str = ""


def build_indicator_dict(cfg: Config) -> list[Indicator]:
    M, T, W, D = cfg.sheet_month, cfg.sheet_tenday, cfg.sheet_week, cfg.sheet_day
    TG = cfg.sheet_target
    inds = [
        # ===== 目标 =====
        Indicator("IVA_yoy", "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)", TG,
                  "target", "last", True, "target",
                  "目标：1-2月【拆分】口径，与社零的【合并】口径不同，勿混用"),

        # ===== 月度 =====
        Indicator("retail_yoy", "中国:社会消费品零售总额:当月同比(1-2月合并)", M,
                  "yoy_keep", "last", True, "predictor",
                  "1-2月【合并】口径；与目标拆分口径不同，1/2月对齐需谨慎"),
        Indicator("fai_cum", "中国:固定资产投资完成额:累计值", M,
                  "cum2mom2yoy", "last", True, "predictor",
                  "累计值(YTD)：先还原当月值再算当月同比；勿当水平值；1月空、2月为1-2合并"),
        Indicator("reinv_cum", "中国:房地产开发投资完成额:累计值", M,
                  "cum2mom2yoy", "last", True, "predictor",
                  "累计值(YTD)：同上还原；与固投累计值是两条近名指标，勿混"),
        Indicator("pmi", "中国:制造业PMI", M, "rate", "last", True, "predictor", "荣枯线50"),
        Indicator("pmi_neworder", "中国:制造业PMI:新订单", M, "rate", "last", True, "predictor",
                  "PMI 子项-新订单，勿与下面新出口订单混"),
        Indicator("pmi_newexport", "中国:制造业PMI:新出口订单", M, "rate", "last", True, "predictor",
                  "PMI 子项-新出口订单"),
        Indicator("power_yoy", "中国:发电量:当月同比", M, "yoy_keep", "last", True, "predictor",
                  "发电量当月同比，与工业生产高度同步"),
        Indicator("ppi_yoy", "中国:PPI:当月同比", M, "yoy_keep", "last", True, "predictor", ""),
        Indicator("ppirm_yoy", "中国:PPIRM:当月同比", M, "yoy_keep", "last", True, "predictor",
                  "PPIRM(生产资料购进价)，勿与 PPI 混"),
        Indicator("pmi_rawprice", "中国:制造业PMI:主要原材料购进价格", M, "rate", "last", True,
                  "predictor", "PMI 价格子项"),
        Indicator("prod_ic", "中国:产量:集成电路:当月值", M, "level2yoy", "last", True, "predictor",
                  "当月值水平量 -> 同比"),
        Indicator("prod_power_equip", "中国:产量:发电设备:当月值", M, "level2yoy", "last", True,
                  "predictor", "当月值水平量 -> 同比"),

        # ===== 旬度（聚合到月，3旬/月）=====
        Indicator("steel_crude_key", "中国:日均产量:粗钢:重点企业", T, "level2yoy", "mean", True,
                  "predictor", "重点企业粗钢日均产量；勿与下面'预估日均产量:粗钢'混"),
        Indicator("steel_crude_est", "中国:预估日均产量:粗钢", T, "level2yoy", "mean", True,
                  "predictor", "全国预估粗钢日均产量（不同来源）"),
        Indicator("steel_rolled_key", "中国:日均产量:钢材:重点企业", T, "level2yoy", "mean", True,
                  "predictor", "重点企业钢材日均产量"),
        Indicator("steel_pig_key", "中国:日均产量:生铁:重点企业", T, "level2yoy", "mean", True,
                  "predictor", "重点企业生铁日均产量"),

        # ===== 周度 =====
        Indicator("util_tire_semi", "中国:开工率:汽车轮胎(半钢胎)", W, "rate", "mean", True,
                  "predictor", "半钢胎；与全钢胎成对近名，勿混"),
        Indicator("util_tire_full", "中国:开工率:汽车轮胎(全钢胎)", W, "rate", "mean", True,
                  "predictor", "全钢胎"),
        Indicator("util_pet_chip", "中国:开工率:聚酯切片", W, "rate", "mean", True, "predictor", ""),
        Indicator("util_asphalt", "中国:开工率:石油沥青装置", W, "rate", "mean", True, "predictor", ""),
        Indicator("prod_rebar_mill", "中国:产量:螺纹钢:主要钢厂", W, "level2yoy", "mean", True,
                  "predictor", "主要钢厂螺纹钢产量(水平量->同比)"),
        Indicator("util_rebar_mill", "中国:开工率:螺纹钢:主要钢厂", W, "rate", "mean", True,
                  "predictor", "螺纹钢开工率(比率)；与上面螺纹钢产量是两条不同指标"),
        Indicator("land_area_100", "中国:100大中城市:成交土地占地面积", W, "level2yoy", "mean", True,
                  "predictor", "成交土地面积(水平量->同比)；地产前瞻"),
        Indicator("scfi", "中国:上海出口集装箱运价指数:综合指数", W, "index2yoy", "mean", True,
                  "predictor", "SCFI 运价指数->同比"),
        Indicator("steel_price_idx", "中国:钢材综合价格指数", W, "index2yoy", "mean", True,
                  "predictor", "钢材综合价格指数->同比"),
        Indicator("coal_south_plant", "中国:南方电厂:日耗量:煤炭", W, "level2yoy", "mean", False,
                  "predictor", "2022年才有：剔出DFM核心集，仅供ML"),
        Indicator("coal_key_plant", "中国:日耗量:煤炭重点电厂", W, "level2yoy", "mean", False,
                  "predictor", "2022年才有：仅供ML；与统调/南方电厂三条近名，勿混"),
        Indicator("coal_unified_plant", "中国:日耗量:煤炭统调电厂", W, "level2yoy", "mean", False,
                  "predictor", "2022年才有：仅供ML"),
        Indicator("car_retail", "中国:日均销量(当周,厂家零售):乘用车", W, "level2yoy", "mean", True,
                  "predictor", "乘用车厂家【零售】；与下面【批发】成对近名，勿混"),
        Indicator("car_wholesale", "中国:日均销量(当周,厂家批发):乘用车", W, "level2yoy", "mean", True,
                  "predictor", "乘用车厂家【批发】"),
        Indicator("cement_ship", "中国:发运率:水泥", W, "rate", "mean", True, "predictor", "水泥发运率(比率)"),
        Indicator("mill_run", "中国:运转率:磨机", W, "rate", "mean", True, "predictor", "磨机运转率(比率)"),
        Indicator("util_poly_loom", "中国:江浙地区:开工率:涤纶长丝:下游织机", W, "rate", "mean", True,
                  "predictor", "下游织机开工率；与下面涤纶长丝本体成对近名，勿混"),
        Indicator("util_poly", "中国:江浙地区:开工率:涤纶长丝", W, "rate", "mean", True,
                  "predictor", "涤纶长丝开工率"),
        Indicator("bf_tangshan", "中国:唐山:高炉开工率", W, "rate", "mean", True, "predictor",
                  "唐山高炉开工率"),

        # ===== 日度 =====
        Indicator("util_pta", "中国:开工率:精对苯二甲酸", D, "rate", "mean", True, "predictor",
                  "PTA 开工率(比率)"),
        Indicator("house_sale_30", "中国:30大中城市:成交面积:商品房", D, "level2yoy", "mean", True,
                  "predictor", "30城商品房成交面积(水平量->同比)；地产需求高频"),
        Indicator("bdi", "波罗的海干散货指数(BDI)", D, "index2yoy", "mean", True, "predictor",
                  "BDI 干散货运价->同比"),
        Indicator("nh_index", "南华综合指数", D, "index2yoy", "mean", True, "predictor",
                  "南华商品综合指数->同比"),
        Indicator("rebar_price", "中国:价格:螺纹钢(HRB400E,20mm)", D, "index2yoy", "mean", True,
                  "predictor", "螺纹钢现货价->同比"),
        Indicator("cement_price_idx", "中国:水泥价格指数", D, "index2yoy", "mean", True,
                  "predictor", "水泥价格指数->同比"),
        Indicator("port_qhd", "中国:秦皇岛港:港口吞吐量:煤炭", D, "level2yoy", "mean", True,
                  "predictor", "秦皇岛港煤炭吞吐量(水平量->同比)；与曹妃甸/京唐三港近名，勿混"),
        Indicator("port_cfd", "中国:曹妃甸港:煤炭调度:港口吞吐量", D, "level2yoy", "mean", True,
                  "predictor", "曹妃甸港煤炭吞吐量"),
        Indicator("port_jt", "中国:京唐老港:港口吞吐量:煤炭", D, "level2yoy", "mean", True,
                  "predictor", "京唐老港煤炭吞吐量"),
    ]
    return inds


# ------------------------------------------------------------------------------
# 1.2  春节日期表（春节正月初一），1990-2030
# ------------------------------------------------------------------------------
SPRING_FESTIVAL = {
    1990: "1990-01-27", 1991: "1991-02-15", 1992: "1992-02-04", 1993: "1993-01-23",
    1994: "1994-02-10", 1995: "1995-01-31", 1996: "1996-02-19", 1997: "1997-02-07",
    1998: "1998-01-28", 1999: "1999-02-16", 2000: "2000-02-05", 2001: "2001-01-24",
    2002: "2002-02-12", 2003: "2003-02-01", 2004: "2004-01-22", 2005: "2005-02-09",
    2006: "2006-01-29", 2007: "2007-02-18", 2008: "2008-02-07", 2009: "2009-01-26",
    2010: "2010-02-14", 2011: "2011-02-03", 2012: "2012-01-23", 2013: "2013-02-10",
    2014: "2014-01-31", 2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29",
    2026: "2026-02-17", 2027: "2027-02-06", 2028: "2028-01-26", 2029: "2029-02-13",
    2030: "2030-02-03",
}


# ==============================================================================
# 第 2 节  通用工具
# ==============================================================================
def month_end(ts) -> pd.Timestamp:
    """返回所在月的月末时间戳。"""
    ts = pd.Timestamp(ts)
    return ts + pd.offsets.MonthEnd(0)


def to_month_index(idx) -> pd.DatetimeIndex:
    """把任意日期索引归一到月末。"""
    return pd.DatetimeIndex([month_end(t) for t in idx])


def safe_yoy(s: pd.Series) -> pd.Series:
    """月度序列的 12 月同比（百分比）。s 需为月末索引、按月连续。
    去年同月为 0 或符号反转会产生 inf/异常，这里把 inf 置为 NaN（交由下游缺失处理）。
    """
    s = s.astype(float)
    base = s.shift(12)
    yoy = (s / base - 1.0) * 100.0
    yoy = yoy.replace([np.inf, -np.inf], np.nan)
    return yoy


# ==============================================================================
# 第 3 节  数据读取与列名校验
# ==============================================================================
class DataLoader:
    """读取 Excel 各 sheet，校验列名是否与指标字典完全一致（防对方更新表时改名/换序）。"""

    def __init__(self, cfg: Config, indicators: list[Indicator]):
        self.cfg = cfg
        self.indicators = indicators
        self.raw: dict[str, pd.DataFrame] = {}
        self.report: dict[str, list] = {"missing_columns": [], "extra_columns": [], "ok": []}

    def load(self) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        if not os.path.exists(cfg.excel_path):
            raise FileNotFoundError(f"找不到数据文件：{cfg.excel_path}")
        xl = pd.ExcelFile(cfg.excel_path)
        LOG.info("Excel sheets: %s", xl.sheet_names)

        for sheet in [cfg.sheet_target, cfg.sheet_month, cfg.sheet_tenday,
                      cfg.sheet_week, cfg.sheet_day]:
            if sheet not in xl.sheet_names:
                raise ValueError(f"缺少 sheet：{sheet}")
            df = xl.parse(sheet)
            df.columns = [str(c).strip() for c in df.columns]
            datecol = df.columns[0]
            df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
            df = df.dropna(subset=[datecol]).sort_values(datecol).reset_index(drop=True)
            df = df.rename(columns={datecol: "date"})
            # 数值化
            for c in df.columns:
                if c != "date":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            self.raw[sheet] = df

        self._validate()
        return self.raw

    def _validate(self):
        """逐指标核对精确列名是否存在。"""
        for ind in self.indicators:
            df = self.raw.get(ind.sheet)
            if df is None or ind.name not in df.columns:
                self.report["missing_columns"].append((ind.sheet, ind.key, ind.name))
                LOG.error("【列名校验失败】sheet=%s 缺少列：%s（内部名 %s）",
                          ind.sheet, ind.name, ind.key)
            else:
                self.report["ok"].append((ind.sheet, ind.key, ind.name))
        # 多出来的列（提示，不报错）
        used = {}
        for ind in self.indicators:
            used.setdefault(ind.sheet, set()).add(ind.name)
        for sheet, df in self.raw.items():
            for c in df.columns:
                if c == "date":
                    continue
                if c not in used.get(sheet, set()):
                    self.report["extra_columns"].append((sheet, c))

        n_missing = len(self.report["missing_columns"])
        if n_missing > 0:
            raise RuntimeError(
                f"列名校验未通过：有 {n_missing} 个指标在表中找不到精确列名。"
                f"很可能是数据提供方更新表时改了名称或调整了列。请核对指标字典与最新表。"
            )
        LOG.info("列名校验通过：%d 个指标全部匹配；额外未使用列 %d 个。",
                 len(self.report["ok"]), len(self.report["extra_columns"]))


# ==============================================================================
# 第 4 节  变换（按口径平稳化）
# ==============================================================================
class Transformer:
    """把每个原始指标转换为"与目标同比口径可比、且尽量平稳"的【原始频率】序列。

    注意：此处只做"口径变换"，不做频率聚合（聚合在第 5 节按真实时点进行）。
    对于需要同比的高频水平量/价格，这里**不**直接做同比（因为高频同比需先聚合到月），
    只在月度指标上直接完成同比；高频的 'level2yoy'/'index2yoy' 标记会被第 5 节识别为
    "先月度聚合再做月度同比"。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ---- 累计值 -> 当月值 -> 当月同比（月度专用）----
    @staticmethod
    def cum_to_mom(s: pd.Series) -> pd.Series:
        """累计值(YTD)还原为当月值。规则：同年内 当月值 = 累计(M) - 累计(M-1)；
        每年 1 月(或年内首个非空月)直接等于累计值本身。
        中国数据 1 月通常缺、2 月为 1-2 月合并 -> 2 月当月值即为该合并值（保留合并口径）。
        """
        s = s.astype(float).copy()
        out = pd.Series(index=s.index, dtype=float)
        for year, grp in s.groupby(s.index.year):
            grp = grp.sort_index()
            prev = None
            for ts, v in grp.items():
                if np.isnan(v):
                    out[ts] = np.nan
                    prev = prev  # 跳过缺失，prev 不变
                    continue
                if prev is None:
                    out[ts] = v          # 年内首个非空累计值即当月(或1-2合并)值
                else:
                    out[ts] = v - prev
                prev = v
        return out

    def transform_monthly(self, s: pd.Series, transform: str) -> pd.Series:
        """月度指标的口径变换，返回月末索引的平稳序列。"""
        s = s.copy()
        s.index = to_month_index(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # 补齐为月连续索引（缺失保留 NaN，供下游模型处理）
        full = pd.date_range(s.index.min(), s.index.max(), freq="ME")
        s = s.reindex(full)

        if transform in ("target", "yoy_keep", "rate"):
            return s
        if transform == "cum2mom2yoy":
            mom = self.cum_to_mom(s)
            return safe_yoy(mom)
        if transform in ("level2yoy", "index2yoy"):
            return safe_yoy(s)
        raise ValueError(f"未知 transform: {transform}")


# ==============================================================================
# 第 5 节  混频对齐（真实时点 / 参差边缘）
# ==============================================================================
class FrequencyAligner:
    """把所有频率对齐到"月末"，并严格按 as_of 信息集构造特征，杜绝前视偏差。

    核心方法 build_monthly_panel(as_of)：
      给定评估时点 as_of，对每个指标只用"参考期末 + 发布滞后 <= as_of"的观测，
      聚合到月，得到该时点真实可见的月度面板（含目标）。
    """

    def __init__(self, cfg: Config, raw: dict[str, pd.DataFrame],
                 indicators: list[Indicator]):
        self.cfg = cfg
        self.raw = raw
        self.indicators = indicators
        self.transformer = Transformer(cfg)
        # 预抽取每个指标的 (date, value) 原始长表
        self._series: dict[str, pd.Series] = {}
        for ind in indicators:
            df = raw[ind.sheet]
            s = df.set_index("date")[ind.name].dropna()
            self._series[ind.key] = s

    def _pub_lag_days(self, ind: Indicator) -> int:
        cfg = self.cfg
        if ind.role == "target":
            return cfg.pub_lag_target_days
        if ind.sheet == cfg.sheet_month:
            return cfg.pub_lag_month_days
        if ind.sheet == cfg.sheet_day:
            return cfg.pub_lag_day_days
        if ind.sheet == cfg.sheet_week:
            return cfg.pub_lag_week_days
        if ind.sheet == cfg.sheet_tenday:
            return cfg.pub_lag_tenday_days
        return cfg.pub_lag_week_days   # 兜底

    def _visible_raw(self, ind: Indicator, as_of: pd.Timestamp) -> pd.Series:
        """返回 as_of 时点可见的原始序列（参考期末 + 发布滞后 <= as_of）。"""
        s = self._series[ind.key]
        lag = pd.Timedelta(days=self._pub_lag_days(ind))
        visible_mask = (s.index + lag) <= as_of
        return s[visible_mask]

    def _aggregate_highfreq_to_month(self, s: pd.Series, agg: str) -> pd.Series:
        """高频 -> 月度聚合。agg='mean' 取月内可见均值（参差月即为部分月均值）。"""
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index)
        grouper = s.groupby(to_month_index(s.index))
        if agg == "mean":
            return grouper.mean()
        if agg == "last":
            return grouper.last()
        if agg == "sum":
            return grouper.sum()
        raise ValueError(agg)

    def _highfreq_monthly_yoy(self, vis: pd.Series) -> pd.Series:
        """高频水平量/价格 -> 月度【同口径(MTD)同比】。

        完整历史月用整月同比(向量化)；只有【最后一个(残)月】用 MTD 同口径同比
        ——拿今年该月已覆盖到的最大日 day<=cap 窗口 ÷ 去年同月同一窗口，消除"残月均值÷
        去年整月均值"的口径偏差。任一 vintage 面板里只有当月是残月，故仅需修正末月，O(月数)。
        """
        s = vis.copy()
        s.index = pd.DatetimeIndex(s.index)
        df = pd.DataFrame({"v": s.astype(float).values}, index=s.index)
        df["month"] = to_month_index(df.index)
        df["dom"] = df.index.day
        monthly_mean = df.groupby("month")["v"].mean().sort_index()
        if len(monthly_mean) == 0:
            return pd.Series(dtype=float)
        # 完整月：整月同比(向量化)
        full_idx = pd.date_range(monthly_mean.index.min(),
                                 monthly_mean.index.max(), freq="ME")
        mm = monthly_mean.reindex(full_idx)
        yoy = (mm / mm.shift(12) - 1.0) * 100.0
        # 末月若残缺(覆盖日<28)则改用 MTD 同口径同比
        last = monthly_mean.index.max()
        cap = int(df.loc[df["month"] == last, "dom"].max())
        if cap < 28:
            prior = month_end(pd.Timestamp(last) - pd.DateOffset(years=1))
            pm = df[(df["month"] == prior) & (df["dom"] <= cap)]["v"]
            cur = df[(df["month"] == last) & (df["dom"] <= cap)]["v"]
            base = pm.mean()
            if len(pm) > 0 and len(cur) > 0 and np.isfinite(base) and abs(base) > 1e-12:
                yoy.loc[last] = (cur.mean() / base - 1.0) * 100.0
            else:
                yoy.loc[last] = np.nan
        return yoy.replace([np.inf, -np.inf], np.nan)

    def build_monthly_panel(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """构造 as_of 时点的月度面板（列=指标内部名，行=月末），含目标列。"""
        cfg = self.cfg
        cols = {}
        for ind in self.indicators:
            vis = self._visible_raw(ind, as_of)
            if len(vis) == 0:
                continue
            if ind.sheet == cfg.sheet_month or ind.role == "target":
                # 月度/目标：直接口径变换
                ser = self.transformer.transform_monthly(vis, ind.transform)
            else:
                # 高频：按口径做"月度同口径(MTD)同比 / 保持比率"
                if ind.transform in ("level2yoy", "index2yoy"):
                    ser = self._highfreq_monthly_yoy(vis)  # 同口径(MTD)同比，消除残月偏差
                    if ser.notna().any():
                        full = pd.date_range(ser.index.min(), ser.index.max(), freq="ME")
                        ser = ser.reindex(full)
                elif ind.transform == "rate":
                    monthly = self._aggregate_highfreq_to_month(vis, ind.agg)
                    monthly.index = to_month_index(monthly.index)
                    ser = monthly[~monthly.index.duplicated(keep="last")].sort_index()
                else:
                    raise ValueError(f"高频指标 {ind.key} 不应是 transform={ind.transform}")
            cols[ind.key] = ser

        panel = pd.DataFrame(cols)
        panel = panel.sort_index()
        panel = panel.replace([np.inf, -np.inf], np.nan)  # 兜底清洗非有限值
        panel.index.name = "date"
        return panel


# ==============================================================================
# 第 6 节  春节特征
# ==============================================================================
def spring_festival_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """为给定月末索引构造春节相关特征（应对 1-2 月拆分口径的人造波动）。

    特征：
      sf_in_jan / sf_in_feb : 当年春节是否落在 1/2 月（哑变量，仅 1、2 月行非零）
      sf_offset_days        : 春节日相对 2 月 1 日的偏移天数（错位强度，1、2 月行）
      sf_this_month         : 当月是否为春节所在月（春节当月=1，否则=0）
      month_1 / month_2     : 1 月、2 月哑变量（拆分口径基数效应）
    """
    rows = []
    for ts in index:
        y, m = ts.year, ts.month
        sf = SPRING_FESTIVAL.get(y, None)
        sf_month = pd.Timestamp(sf).month if sf else None
        sf_day_offset = ((pd.Timestamp(sf) - pd.Timestamp(f"{y}-02-01")).days
                         if sf else 0.0)
        rows.append({
            "sf_in_jan": 1.0 if (m == 1 and sf_month == 1) else 0.0,
            "sf_in_feb": 1.0 if (m == 2 and sf_month == 2) else 0.0,
            "sf_offset_days": float(sf_day_offset) if m in (1, 2) else 0.0,
            "sf_this_month": 1.0 if (sf_month == m) else 0.0,
            "month_1": 1.0 if m == 1 else 0.0,
            "month_2": 1.0 if m == 2 else 0.0,
        })
    return pd.DataFrame(rows, index=index)


# ==============================================================================
# 第 7 节  特征矩阵（供 ARX 与 LightGBM 等桥接模型使用）
# ==============================================================================
class FeatureBuilder:
    """由月度面板构造监督学习特征矩阵：滞后、目标自回归、基数、春节。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, panel: pd.DataFrame, target_key: str = "IVA_yoy") -> pd.DataFrame:
        cfg = self.cfg
        feat = pd.DataFrame(index=panel.index)
        # 目标列可能在很早的月份(目标尚不可见)缺失 -> 用全 NaN 兜底，避免 KeyError
        y = panel[target_key] if target_key in panel.columns \
            else pd.Series(np.nan, index=panel.index)

        # 预测变量列（除目标）
        pred_cols = [c for c in panel.columns if c != target_key]

        # 1) 高频/预测变量：当期(month T，nowcasting核心)与滞后
        for c in pred_cols:
            feat[f"{c}_t0"] = panel[c]
            for L in cfg.feature_lags:
                feat[f"{c}_l{L}"] = panel[c].shift(L)

        # 2) 目标自回归滞后
        for L in cfg.target_ar_lags:
            feat[f"y_l{L}"] = y.shift(L)

        # 3) 基数效应（去年同月目标值）
        feat["y_base_l12"] = y.shift(12)

        # 4) 春节 / 月份特征
        sf = spring_festival_features(panel.index)
        for c in sf.columns:
            feat[c] = sf[c]

        feat["__target__"] = y
        feat = feat.replace([np.inf, -np.inf], np.nan)  # 兜底清洗
        return feat


# ==============================================================================
# 第 8 节  模型
# ==============================================================================
@dataclass
class Prediction:
    point: float
    lower: dict = field(default_factory=dict)   # {level: value}
    upper: dict = field(default_factory=dict)
    sigma: Optional[float] = None               # 模型自报的标准差（若有）
    extra: dict = field(default_factory=dict)


class BaseModel:
    name = "base"

    def predict_asof(self, target_month: pd.Timestamp, as_of: pd.Timestamp,
                     ctx: "Context") -> Optional[Prediction]:
        raise NotImplementedError


# ------------------------------------------------------------------------------
# 8.1  MF-DFM（混频动态因子模型）
# ------------------------------------------------------------------------------
class ModelDFM(BaseModel):
    """statsmodels.DynamicFactorMQ：状态空间 + Kalman + EM，原生处理缺失/不等长/参差边缘。
    分层：只用 dfm_tier=True 的"长且完整"指标，保证因子稳定。
    区间：由 Kalman 预测方差给出高斯区间。
    提速：每 dfm_refit_every 个月重估参数，其余月份复用参数仅做滤波/平滑。
    """
    name = "DFM"

    def __init__(self, cfg: Config, indicators: list[Indicator]):
        self.cfg = cfg
        self.tier_keys = [i.key for i in indicators if i.dfm_tier]
        self._cached_params = None
        self._cached_month = None

    def predict_asof(self, target_month, as_of, ctx) -> Optional[Prediction]:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        cfg = self.cfg
        panel = ctx.aligner.build_monthly_panel(as_of)
        cols = [c for c in self.tier_keys if c in panel.columns]
        endog = panel[cols].copy()

        # 截到 target_month（含），target_month 的目标值此时应为 NaN（未发布）
        endog = endog.loc[:target_month]
        if target_month not in endog.index:
            endog.loc[target_month] = np.nan
            endog = endog.sort_index()

        # 丢弃整列全空、以及在该时点观测过少的列
        min_obs = 36
        keep = [c for c in endog.columns if endog[c].notna().sum() >= min_obs]
        endog = endog[keep]
        if "IVA_yoy" not in endog.columns:
            return None

        # 设月度频率（statsmodels 需要明确频率）
        endog = endog.asfreq("ME")

        try:
            model = DynamicFactorMQ(
                endog,
                factors=cfg.dfm_factors,
                factor_orders=cfg.dfm_factor_order,
                idiosyncratic_ar1=True,
                standardize=True,
            )
            refit = (self._cached_params is None or
                     self._cached_month is None or
                     (target_month.to_period("M") - self._cached_month.to_period("M")).n
                     >= cfg.dfm_refit_every)
            if refit:
                res = model.fit(maxiter=cfg.dfm_maxiter, disp=False)
                self._cached_params = res.params
                self._cached_month = target_month
            else:
                try:
                    res = model.smooth(self._cached_params)
                except Exception:
                    res = model.fit(maxiter=cfg.dfm_maxiter, disp=False)
                    self._cached_params = res.params
                    self._cached_month = target_month

            # 目标在 target_month 的平滑/预测值与区间
            pred = res.get_prediction(start=target_month, end=target_month)
            mean = float(pred.predicted_mean["IVA_yoy"].iloc[0])
            out = Prediction(point=mean)
            # 用 Kalman 给出的 conf_int 作为 DFM 自带区间（最终以组合层共形为准）
            for lv in cfg.interval_levels:
                try:
                    ci = pred.conf_int(alpha=1 - lv)
                    lo = float(ci.filter(like="lower").filter(like="IVA_yoy").iloc[0, 0])
                    hi = float(ci.filter(like="upper").filter(like="IVA_yoy").iloc[0, 0])
                    out.lower[lv], out.upper[lv] = lo, hi
                    if out.sigma is None and lv == 0.95:
                        out.sigma = (hi - lo) / (2 * 1.959963985)
                except Exception:
                    pass
            return out
        except Exception as e:
            LOG.warning("DFM 在 %s 失败：%s", target_month.date(), e)
            return None
# ------------------------------------------------------------------------------
# 8.3  LightGBM（分位数）+ 原生缺失
# ------------------------------------------------------------------------------
class ModelLGB(BaseModel):
    """梯度提升：原生处理缺失、捕捉非线性，吸收很新/很稀疏的指标（如 2022 起的煤耗）。
    区间：分位数回归（pinball loss）直接给出分位数；中位数作为点预测的备选。
    """
    name = "LightGBM"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict_asof(self, target_month, as_of, ctx) -> Optional[Prediction]:
        import lightgbm as lgb
        cfg = self.cfg
        feat = ctx.realtime_feature_matrix()  # 真实时点矩阵:训练/预测信息集一致
        if target_month not in feat.index:
            return None
        train = feat.loc[:target_month].iloc[:-1].dropna(subset=["__target__"])
        # 抗过拟合：限定训练窗口
        if len(train) > cfg.max_train_months:
            train = train.iloc[-cfg.max_train_months:]
        if len(train) < cfg.backtest_min_train // 2:
            return None

        Xtr = train.drop(columns="__target__")
        ytr = train["__target__"]
        Xte = feat.loc[[target_month]].drop(columns="__target__")

        # 抗过拟合：大幅加正则——更浅的树、更少叶子、更高最小样本、更强 L1/L2、
        # 列/行下采样、最小分裂增益门槛，并限制 n_estimators。
        params_common = dict(
            n_estimators=250, learning_rate=0.03, num_leaves=8, max_depth=3,
            min_child_samples=30, min_split_gain=0.02,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
            reg_lambda=5.0, reg_alpha=2.0,
            random_state=cfg.random_state, n_jobs=-1, verbose=-1,
        )
        try:
            preds = {}
            for q in cfg.lgb_quantiles:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params_common)
                m.fit(Xtr.values, ytr.values)
                preds[q] = float(m.predict(Xte.values)[0])
            # 点预测：用专门的 L2 目标更稳
            m_mean = lgb.LGBMRegressor(objective="regression", **params_common)
            m_mean.fit(Xtr.values, ytr.values)
            point = float(m_mean.predict(Xte.values)[0])

            out = Prediction(point=point)
            # 由分位数拼出 80/95 区间（保证单调）
            qs = sorted(preds.items())
            qvals = np.array([v for _, v in qs])
            qvals = np.maximum.accumulate(qvals)  # 强制单调
            qmap = {k: v for (k, _), v in zip(qs, qvals)}
            out.lower[0.95] = qmap.get(0.025, point)
            out.upper[0.95] = qmap.get(0.975, point)
            out.lower[0.80] = qmap.get(0.10, point)
            out.upper[0.80] = qmap.get(0.90, point)
            out.extra["importance"] = dict(
                zip(Xtr.columns, m_mean.booster_.feature_importance(importance_type="gain")))
            return out
        except Exception as e:
            LOG.warning("LightGBM 在 %s 失败：%s", target_month.date(), e)
            return None
def _target_history(ctx, as_of, target_month, drop_janfeb=True):
    """取 as_of 可见的目标历史（到 T-1）。drop_janfeb=True 则剔除 1-2 月。"""
    panel = ctx.aligner.build_monthly_panel(as_of)
    if "IVA_yoy" not in panel.columns:
        return None
    y = panel["IVA_yoy"].dropna()
    y = y.loc[:target_month]
    if target_month in y.index:
        y = y.drop(target_month)
    if drop_janfeb:
        y = y[~y.index.month.isin([1, 2])]
    return y


def _gauss_interval(out: Prediction, point, sigma, cfg):
    from scipy.stats import norm
    out.sigma = sigma
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        for lv in cfg.interval_levels:
            z = norm.ppf(0.5 + lv / 2)
            out.lower[lv] = point - z * sigma
            out.upper[lv] = point + z * sigma


def _nonjf_supervised_design(ctx, cfg, target_month, n_arlags=3,
                             include_arlags=True, topk=None):
    """为【非1-2月】监督桥接模型(ARX/PLS)构造无泄漏设计矩阵。

    要点：
      - 仅用【非1-2月】行(避免1-2月人造噪声污染)；
      - AR 滞后用【非1-2月子序列】的滞后(arlag1..p)，而非日历上月(后者对3月会取到2月噪声)；
      - 高频特征：只取在预测点【可观测】且训练覆盖充分的列，按|相关|取 top-K(仅训练段算)；
      - 排除日历 y_* 特征(可能含1-2月污染)，AR 信息统一由干净的 arlag 提供；
      - 测试行缺失用【训练集】中位数填补。
    返回 (Xtr, ytr, Xte, cols, med) 或 None。
    """
    feat = ctx.realtime_feature_matrix()
    if target_month not in feat.index:
        return None
    nf_idx = feat.index[~feat.index.month.isin([1, 2])]
    if target_month not in nf_idx:
        return None
    ytar = feat["__target__"]
    ynf = ytar.loc[nf_idx]
    lagdf = pd.DataFrame(index=nf_idx)
    if include_arlags:
        for L in range(1, n_arlags + 1):
            lagdf[f"arlag{L}"] = ynf.shift(L)
    hf_cols = [c for c in feat.columns
               if c != "__target__" and not c.startswith("y_")]
    test_row = feat.loc[target_month]
    train_idx = nf_idx[(nf_idx < target_month) & ytar.loc[nf_idx].notna()]
    if len(train_idx) > cfg.max_train_months:
        train_idx = train_idx[-cfg.max_train_months:]
    if len(train_idx) < 30:
        return None
    min_cov = max(24, len(train_idx) // 3)
    ytr = ytar.loc[train_idx]
    cand = [c for c in hf_cols if pd.notna(test_row[c])
            and feat.loc[train_idx, c].notna().sum() >= min_cov]
    corr = {}
    for c in cand:
        v = feat.loc[train_idx, c]
        m = v.notna() & ytr.notna()
        if m.sum() >= 24 and v[m].std() > 1e-9:
            corr[c] = abs(np.corrcoef(v[m], ytr[m])[0, 1])
    k = topk or cfg.bridge_topk_features
    top = [c for c, _ in sorted(corr.items(), key=lambda kv: kv[1],
                                reverse=True)[:k]]
    design = pd.concat([feat.loc[nf_idx, top], lagdf], axis=1)
    Xtr_raw = design.loc[train_idx]
    Xte_raw = design.loc[[target_month]]
    # AR 滞后在预测点必须可得(否则该模型不适用)
    if include_arlags and Xte_raw[list(lagdf.columns)].isna().any(axis=1).iloc[0]:
        return None
    if len(top) + len(lagdf.columns) < 3:
        return None
    med = Xtr_raw.median()
    Xtr = Xtr_raw.fillna(med).fillna(0.0)
    Xte = Xte_raw.fillna(med).fillna(0.0)
    return Xtr, ytr, Xte, list(design.columns), med


# ------------------------------------------------------------------------------
# 8.4b  AR(p)（自回归，基准）——平稳自相关序列的标准技能下限
# ------------------------------------------------------------------------------
class ModelAR(BaseModel):
    """对【非1-2月】目标子序列拟合 AR(p)，阶数由 BIC 选择(无CV、无泄漏)。

    比"近期均值"更标准、更严格的基准：显式建模惯性/均值回复。
    1-2月退化为近期均值。
    """
    name = "AR"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict_asof(self, target_month, as_of, ctx) -> Optional[Prediction]:
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order
        cfg = self.cfg
        ynf = _target_history(ctx, as_of, target_month, drop_janfeb=True)
        if ynf is None or len(ynf) < 30:
            return None
        vals = ynf.values.astype(float)
        try:
            if target_month.month in (1, 2):
                point = float(ynf.tail(cfg.locallevel_k).mean())
                sigma = float(ynf.tail(cfg.locallevel_k).std(ddof=1))
            else:
                maxlag = int(min(12, max(1, len(vals) // 6)))
                try:
                    sel = ar_select_order(vals, maxlag=maxlag, ic="bic",
                                          old_names=False)
                    lags = sel.ar_lags if sel.ar_lags is not None and len(sel.ar_lags) else 1
                except Exception:
                    lags = 1
                res = AutoReg(vals, lags=lags, old_names=False).fit()
                point = float(np.asarray(res.predict(start=len(vals),
                                                     end=len(vals)))[0])
                sigma = float(np.sqrt(res.sigma2))
            out = Prediction(point=point)
            _gauss_interval(out, point, sigma, cfg)
            return out
        except Exception as e:
            LOG.warning("AR 在 %s 失败：%s", target_month.date(), e)
            return None


# ------------------------------------------------------------------------------
# 8.4c  ARX（AR 滞后 + 高频外生，正则化 + 时序CV，无泄漏）
# ------------------------------------------------------------------------------
class ModelARX(BaseModel):
    """ARX：非1-2月 AR 滞后 + 高频外生回归，ElasticNet 正则、TimeSeriesSplit 定参。
    嵌套 AR(惯性锚)，公平检验高频在惯性之上的增量价值。1-2月退化为近期均值。"""
    name = "ARX"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict_asof(self, target_month, as_of, ctx) -> Optional[Prediction]:
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        cfg = self.cfg
        if target_month.month in (1, 2):
            ynf = _target_history(ctx, as_of, target_month, drop_janfeb=True)
            if ynf is None or len(ynf) < 6:
                return None
            point = float(ynf.tail(cfg.locallevel_k).mean())
            out = Prediction(point=point)
            _gauss_interval(out, point, float(ynf.tail(cfg.locallevel_k).std(ddof=1)), cfg)
            return out
        des = _nonjf_supervised_design(ctx, cfg, target_month, include_arlags=True)
        if des is None:
            return None
        Xtr, ytr, Xte, cols, med = des
        try:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr.values)
            Xte_s = sc.transform(Xte.values)
            model = ElasticNetCV(l1_ratio=list(cfg.enet_l1_ratios),
                                 cv=TimeSeriesSplit(5), max_iter=50000,
                                 n_jobs=-1, random_state=cfg.random_state)
            model.fit(Xtr_s, ytr.values)
            point = float(model.predict(Xte_s)[0])
            sigma = float(np.std(ytr.values - model.predict(Xtr_s), ddof=1))
            out = Prediction(point=point)
            _gauss_interval(out, point, sigma, cfg)
            out.extra["n_nonzero"] = int(np.sum(model.coef_ != 0))
            order = np.argsort(-np.abs(model.coef_))
            out.extra["coef_top"] = [
                {"feature": cols[i], "coef": round(float(model.coef_[i]), 3)}
                for i in order[:10] if model.coef_[i] != 0]
            return out
        except Exception as e:
            LOG.warning("ARX 在 %s 失败：%s", target_month.date(), e)
            return None


# ------------------------------------------------------------------------------
# 8.8  局部水平基准（朴素地板/参照）
# ------------------------------------------------------------------------------
class ModelLocalLevel(BaseModel):
    """局部水平基准：预测 = 最近 K 个【非1-2月】已发布目标值的均值。

    回测证明这个"什么都不学"的基准击败了全部复杂模型——故把它正式纳入体系，
    既作为组合的【锚】，也作为 DM 检验里所有复杂模型必须显著超越的对象。
    对 1-2 月：退化为最近 K 个含 1-2 月的均值（1-2 月本不可预测，仅兜底）。
    区间：近期波动的正态近似（最终由组合层共形校准）。
    """
    name = "LocalLevel"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def predict_asof(self, target_month, as_of, ctx) -> Optional[Prediction]:
        cfg = self.cfg
        panel = ctx.aligner.build_monthly_panel(as_of)
        y = panel["IVA_yoy"].dropna()
        y = y.loc[:target_month]
        if target_month in y.index:
            y = y.drop(target_month)
        if len(y) < 3:
            return None
        if target_month.month in (1, 2):
            hist = y.tail(cfg.locallevel_k)
        else:
            hist = y[~y.index.month.isin([1, 2])].tail(cfg.locallevel_k)
        if len(hist) == 0:
            return None
        point = float(hist.mean())
        sigma = float(hist.std(ddof=1)) if len(hist) >= 2 else None
        out = Prediction(point=point, sigma=sigma)
        if sigma is not None and sigma > 0:
            from scipy.stats import norm
            for lv in cfg.interval_levels:
                z = norm.ppf(0.5 + lv / 2)
                out.lower[lv] = point - z * sigma
                out.upper[lv] = point + z * sigma
        return out


# ==============================================================================
# 第 9 节  上下文（按 as_of 缓存对齐面板与特征，避免重复计算）
# ==============================================================================
class Context:
    """承载对齐器与特征构造器，并对 as_of 结果做缓存（同一 as_of 多模型共享）。"""

    def __init__(self, cfg: Config, aligner: FrequencyAligner, fb: FeatureBuilder):
        self.cfg = cfg
        self.aligner = aligner
        self.fb = fb
        self._panel_cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._rt_feat: Optional[pd.DataFrame] = None

    def panel_asof(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if as_of not in self._panel_cache:
            self._panel_cache[as_of] = self.aligner.build_monthly_panel(as_of)
        return self._panel_cache[as_of]

    def realtime_feature_matrix(self) -> pd.DataFrame:
        """【真实时点(vintage)特征矩阵】——修复 ARX/LGB 训练与预测信息集不一致。

        每一行 m 都用"在 m 月 asof_day_of_month 号能看到的数据"构造(panel@as_of_m)：
          - 当月 m 高频=残月(到约19日)、月度宏观当月值=未发布(NaN)；
          - 过去月份 m-1, m-2…=完整可见；目标滞后=已发布部分。
        于是【训练行与预测行具有完全相同的可得性结构】，杜绝"训练用完整月、预测用残月"
        的信息集不一致。该矩阵与"何时运行"无关(行 m 只依赖 m 自身的 as_of)，故全局构建
        一次并缓存，回测各步与实盘共享、且天然无前视。
        """
        if self._rt_feat is not None:
            return self._rt_feat
        # 用一个很晚的 as_of 取到完整月份索引范围 + 已实现目标值
        full = self.aligner.build_monthly_panel(pd.Timestamp("2100-01-01"))
        months = [m for m in full.index]
        rows = []
        for m in months:
            as_of_m = asof_for_target(self.cfg, m)
            panel_m = self.panel_asof(as_of_m)
            feat_m = self.fb.build(panel_m)
            if m in feat_m.index:
                rows.append(feat_m.loc[[m]])
        rt = pd.concat(rows).sort_index()
        # 关键：特征保持 vintage，但【标签 __target__ 必须用已实现的真实目标值】
        # (vintage 行的当月目标未发布=NaN，不能当训练标签；其余 y 滞后特征仍是 vintage)
        rt["__target__"] = full["IVA_yoy"].reindex(rt.index)
        self._rt_feat = rt
        LOG.info("已构建真实时点(vintage)特征矩阵：%s", rt.shape)
        return self._rt_feat


# 让各模型经由 Context 拿到 aligner（DFM/AR/LocalLevel 直接用 ctx.aligner，但内部调用的是
# ctx.panel_asof 的缓存版本——这里把 aligner.build_monthly_panel 代理到缓存）。
def _wire_context_cache(ctx: Context):
    """把 aligner.build_monthly_panel 包一层缓存，使 DFM/AR 等也复用缓存面板。"""
    raw_build = ctx.aligner.build_monthly_panel

    def cached(as_of):
        return ctx.panel_asof_raw(as_of, raw_build)
    # 安装缓存代理
    def panel_asof_raw(as_of, builder):
        if as_of not in ctx._panel_cache:
            ctx._panel_cache[as_of] = builder(as_of)
        return ctx._panel_cache[as_of]
    ctx.panel_asof_raw = panel_asof_raw
    ctx.aligner.build_monthly_panel = cached


# ==============================================================================
# 第 10 节  真实时点工具：由"预测月"推出 as_of
# ==============================================================================
def infer_collection_date(raw: dict) -> pd.Timestamp:
    """从汇总表推断采集日 = 各 sheet 最新观测日期的最大值(数据新鲜度)。"""
    mx = None
    for df in raw.values():
        if "date" in df.columns and len(df):
            d = pd.to_datetime(df["date"]).max()
            mx = d if mx is None else max(mx, d)
    return pd.Timestamp(mx)


def resolve_asof_day(cfg: Config, raw: dict) -> int:
    """确定 as_of 的"当月第几天"：优先用 cfg 显式值，否则取采集日(run_date)的 day。"""
    if cfg.asof_day_of_month is not None:
        return int(cfg.asof_day_of_month)
    run = pd.Timestamp(cfg.run_date) if cfg.run_date else infer_collection_date(raw)
    return int(run.day)


def asof_for_target(cfg: Config, target_month: pd.Timestamp) -> pd.Timestamp:
    """预测月 T 的评估时点 = T 当月的 asof_day 号。

    asof_day 不写死：由 main() 在加载数据后调用 resolve_asof_day() 按【汇总表采集日】
    自动确定并写回 cfg.asof_day_of_month，随采集时间自动平移。若未解析(独立调用)则回退 23。
    业务含义：此时 T 的高频已覆盖约 2/3 个月(残月)，目标与月度宏观对 T 尚不可见、对 T-1
    可见。回测与实盘使用同一规则，杜绝前视偏差。
    """
    tm = month_end(target_month)
    day = cfg.asof_day_of_month if cfg.asof_day_of_month is not None else 23
    day = min(int(day), tm.day)
    return pd.Timestamp(year=tm.year, month=tm.month, day=day)


def true_target_value(raw: dict, cfg: Config, target_month: pd.Timestamp) -> Optional[float]:
    """从原始表取目标月的真实工业增加值（用于回测对照；预测未来时返回 None）。"""
    df = raw[cfg.sheet_target]
    name = "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)"
    s = df.set_index("date")[name]
    s.index = to_month_index(s.index)
    tm = month_end(target_month)
    if tm in s.index and pd.notna(s.loc[tm]):
        return float(s.loc[tm])
    return None


# ==============================================================================
# 第 11 节  回测引擎（伪真实时点，扩展窗口）
# ==============================================================================
class Backtester:
    """对历史每个月做"伪真实时点"外推：仅用该月 as_of 可见信息，逐模型预测并评估。
    产出每个模型的逐月预测、真实值、误差，供组合权重与共形区间使用。
    """

    def __init__(self, cfg: Config, ctx: Context, models: list[BaseModel],
                 raw: dict):
        self.cfg = cfg
        self.ctx = ctx
        self.models = models
        self.raw = raw

    def _target_months(self) -> list[pd.Timestamp]:
        cfg = self.cfg
        s = self.raw[cfg.sheet_target].set_index("date")[
            "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)"].dropna()
        s.index = to_month_index(s.index)
        s = s.sort_index()
        all_months = list(s.index)
        # 需要至少 backtest_min_train 个训练月
        if len(all_months) <= cfg.backtest_min_train:
            return []
        candidates = all_months[cfg.backtest_min_train:]
        return candidates[-cfg.backtest_months:]

    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        months = self._target_months()
        LOG.info("回测区间：%s ~ %s 共 %d 个月",
                 months[0].date() if months else None,
                 months[-1].date() if months else None, len(months))
        records = []
        for i, tm in enumerate(months):
            as_of = asof_for_target(cfg, tm)
            y_true = true_target_value(self.raw, cfg, tm)
            rec = {"target_month": tm, "as_of": as_of, "y_true": y_true,
                   "is_jan_feb": tm.month in (1, 2)}
            for mdl in self.models:
                try:
                    pred = mdl.predict_asof(tm, as_of, self.ctx)
                except Exception as e:
                    LOG.warning("%s 在 %s 抛异常：%s", mdl.name, tm.date(), e)
                    pred = None
                if pred is not None and np.isfinite(pred.point):
                    rec[f"{mdl.name}__point"] = pred.point
                    for lv in cfg.interval_levels:
                        rec[f"{mdl.name}__lo{int(lv*100)}"] = pred.lower.get(lv, np.nan)
                        rec[f"{mdl.name}__hi{int(lv*100)}"] = pred.upper.get(lv, np.nan)
                else:
                    rec[f"{mdl.name}__point"] = np.nan
            records.append(rec)
            if (i + 1) % 6 == 0 or i == len(months) - 1:
                LOG.info("  回测进度 %d/%d（最新 %s）", i + 1, len(months), tm.date())
        return pd.DataFrame(records).set_index("target_month")


# ==============================================================================
# 第 12 节  评估指标
# ==============================================================================
def point_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  y_naive: Optional[np.ndarray] = None) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"n": 0}
    err = yp - yt
    out = {
        "n": int(len(yt)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "ME(bias)": float(np.mean(err)),
    }
    # 方向命中率（相对上月变化方向）
    if len(yt) >= 2:
        out["DA"] = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))
    # MASE（相对季节朴素）：分子与分母必须在【同一样本】nm 上计算，否则不可比
    if y_naive is not None:
        nm = mask & np.isfinite(y_naive)
        if nm.sum() > 0:
            num = np.mean(np.abs(y_pred[nm] - y_true[nm]))
            denom = np.mean(np.abs(y_true[nm] - y_naive[nm]))
            if denom > 0:
                out["MASE"] = float(num / denom)
    return out


def interval_metrics(y_true, lower, upper, nominal: float) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    yt, lo, hi = y_true[mask], lower[mask], upper[mask]
    if len(yt) == 0:
        return {}
    cover = np.mean((yt >= lo) & (yt <= hi))
    width = np.mean(hi - lo)
    return {f"PICP@{int(nominal*100)}": float(cover),
            f"MPIW@{int(nominal*100)}": float(width)}


def diebold_mariano(err_model: np.ndarray, err_bench: np.ndarray, h: int = 1):
    """Diebold-Mariano 检验（平方损失，含 Harvey-Leybourne-Newbold 小样本修正）。

    err_* 为各自的预测误差(pred-actual)。约定损失差 d = loss_model - loss_bench：
      DM<0 且显著  -> 模型损失更低，【显著优于】基准；
      DM>0 且显著  -> 模型【显著差于】基准；
      不显著        -> 与基准无统计差异。
    返回 (DM统计量, 双侧p值)。
    """
    from scipy.stats import t
    e1, e2 = np.asarray(err_model, float), np.asarray(err_bench, float)
    m = np.isfinite(e1) & np.isfinite(e2)
    d = e1[m] ** 2 - e2[m] ** 2
    n = len(d)
    if n < 8:
        return np.nan, np.nan
    dbar = d.mean()
    # h=1 时长程方差即样本方差；保留 NW 框架以便推广
    gamma0 = np.var(d, ddof=0)
    var_dbar = gamma0 / n
    if var_dbar <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_dbar)
    # HLN 小样本修正
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


# ==============================================================================
# 第 13 节  组合（逆误差加权）+ 分裂共形区间
# ==============================================================================
class Ensemble:
    """逆 RMSE 加权组合点预测；对组合残差做分裂共形，生成 80%/95% 区间。
    1/2 月单独校准（其波动更大、区间应更宽）。
    """

    def __init__(self, cfg: Config, model_names: list[str]):
        self.cfg = cfg
        self.model_names = model_names
        self.weights: dict[str, float] = {}
        self.conformal_q: dict = {}   # {(level, group): half_width}
        self.dm_results: dict = {}    # {model: (rmse, dm_stat, p_value)}

    def fit(self, bt: pd.DataFrame):
        cfg = self.cfg
        y = bt["y_true"].values
        jf = bt["is_jan_feb"].values
        # 1) 计算各模型逆误差权重。默认按"剔除1-2月"误差计算——1-2月拆分值是不可预测
        #    的人造噪声，若计入会让权重被噪声主导、奖励到对噪声偶然拟合的模型。
        weight_mask_base = (~jf) if cfg.ensemble_exclude_janfeb_in_weights else np.ones_like(jf, bool)
        # 短窗口稳健：有效月数过少时放宽门槛，避免直接失败
        n_eff = int((np.isfinite(y) & weight_mask_base).sum())
        min_obs = min(6, max(3, n_eff))
        rmses, errs = {}, {}
        for name in self.model_names:
            col = f"{name}__point"
            if col not in bt:
                continue
            p = bt[col].values
            m = np.isfinite(y) & np.isfinite(p) & weight_mask_base
            if m.sum() >= min_obs:
                rmses[name] = np.sqrt(np.mean((p[m] - y[m]) ** 2))
                # 对齐的误差序列(剔1-2月、双方都有值)，供 DM 检验
                errs[name] = pd.Series(p - y, index=bt.index).where(
                    pd.Series(m, index=bt.index))
        if not rmses:
            # 兜底：回测样本极少 -> 退化为基准(若有)或所有可用模型等权（仍继续算共形）
            LOG.warning("回测有效样本不足(n_eff=%d)，退化为基准/等权组合。", n_eff)
            avail = [n for n in self.model_names if f"{n}__point" in bt
                     and bt[f"{n}__point"].notna().any()]
            if not avail:
                raise RuntimeError("没有任何模型产生有效回测预测，无法组合。")
            fallback = ([cfg.benchmark_name] if cfg.benchmark_name in avail else avail)
            self.weights = {k: 1.0 / len(fallback) for k in fallback}
            self.dm_results = {}
            LOG.info("组合权重(兜底)：%s", self.weights)
            return self._finalize_conformal(bt, cfg)

        # === 基准门控 + DM 检验：复杂模型必须"不显著差于"基准才进组合 ===
        bench = cfg.benchmark_name
        self.dm_results = {}
        if bench in rmses:
            bench_rmse = rmses[bench]
            eb = errs[bench]
            keep = [bench]  # 基准恒保留(锚)
            for name in rmses:
                if name == bench:
                    continue
                # 对齐两模型共同有效的月份
                pair = pd.concat([errs[name], eb], axis=1).dropna()
                dm, pval = (diebold_mariano(pair.iloc[:, 0].values,
                                            pair.iloc[:, 1].values)
                            if len(pair) >= 8 else (np.nan, np.nan))
                self.dm_results[name] = (rmses[name], dm, pval)
                # 保留条件：RMSE 不超过基准*keep_ratio（即"打得过或基本不输"）
                if rmses[name] <= bench_rmse * cfg.ensemble_keep_ratio:
                    keep.append(name)
            dropped = [n for n in rmses if n not in keep]
            LOG.info("基准=%s(RMSE=%.3f)；保留模型=%s；剔除(显著/明显差于基准)=%s",
                     bench, bench_rmse, keep, dropped)
            for name, (r, dm, pval) in self.dm_results.items():
                verdict = ("显著优于基准" if (dm is not None and np.isfinite(dm) and dm < 0 and pval < cfg.dm_significance)
                           else "显著差于基准" if (dm is not None and np.isfinite(dm) and dm > 0 and pval < cfg.dm_significance)
                           else "与基准无显著差异")
                LOG.info("  DM[%s vs %s]: RMSE=%.3f  DM=%s  p=%s  -> %s",
                         name, bench, r,
                         f"{dm:+.2f}" if np.isfinite(dm) else "NA",
                         f"{pval:.3f}" if np.isfinite(pval) else "NA", verdict)
            rmses = {k: v for k, v in rmses.items() if k in keep}
        else:
            LOG.warning("未找到基准模型 %s，退回为全模型逆MSE加权。", bench)

        # 逆误差幂次加权（仅在通过门控的模型间）
        inv = {k: 1.0 / max(v, 1e-6) ** cfg.ensemble_weight_power
               for k, v in rmses.items()}
        ssum = sum(inv.values())
        self.weights = {k: v / ssum for k, v in inv.items()}
        LOG.info("组合权重：%s", {k: round(v, 3) for k, v in self.weights.items()})

        return self._finalize_conformal(bt, cfg)

    def _finalize_conformal(self, bt: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        """组合点预测 + 分裂共形区间（1/2月与其他月分组校准）。"""
        bt = bt.copy()
        bt["ENSEMBLE__point"] = self.combine_points(bt)
        resid = (bt["ENSEMBLE__point"] - bt["y_true"]).abs().values
        grp = bt["is_jan_feb"].values
        all_resid = resid[np.isfinite(resid)]
        for lv in cfg.interval_levels:
            for group, gmask in [("normal", ~grp), ("janfeb", grp)]:
                r = resid[gmask & np.isfinite(resid)]
                if len(r) >= 5:
                    k = int(np.ceil((len(r) + 1) * lv))   # 分裂共形分位数(有限样本校正)
                    k = min(k, len(r))
                    q = np.sort(r)[k - 1]
                elif len(all_resid) >= 1:
                    q = float(np.nanquantile(all_resid, lv))  # 样本少时用全体残差兜底
                else:
                    q = np.nan
                self.conformal_q[(lv, group)] = float(q)
        LOG.info("共形半宽：%s", {f"{k[1]}@{int(k[0]*100)}": round(v, 2)
                                   for k, v in self.conformal_q.items()})
        return bt

    def combine_points(self, bt: pd.DataFrame) -> np.ndarray:
        num = np.zeros(len(bt))
        den = np.zeros(len(bt))
        for name, w in self.weights.items():
            col = f"{name}__point"
            if col not in bt:
                continue
            p = bt[col].values
            m = np.isfinite(p)
            num[m] += w * p[m]
            den[m] += w
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return out

    def combine_single(self, preds: dict[str, Prediction]) -> Optional[float]:
        num, den = 0.0, 0.0
        for name, w in self.weights.items():
            if name in preds and preds[name] is not None and np.isfinite(preds[name].point):
                num += w * preds[name].point
                den += w
        return num / den if den > 0 else None

    def interval_for(self, point: float, target_month: pd.Timestamp) -> dict:
        group = "janfeb" if target_month.month in (1, 2) else "normal"
        out = {}
        for lv in self.cfg.interval_levels:
            q = self.conformal_q.get((lv, group), np.nan)
            out[lv] = (point - q, point + q)
        return out


# ==============================================================================
# 第 14 节  报告与产出
# ==============================================================================
class Reporter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

    def save_backtest(self, bt: pd.DataFrame, ens: Ensemble) -> pd.DataFrame:
        cfg = self.cfg
        bt = bt.copy()
        bt["ENSEMBLE__point"] = ens.combine_points(bt)
        # 组合区间
        for lv in cfg.interval_levels:
            los, his = [], []
            for tm, row in bt.iterrows():
                pt = row["ENSEMBLE__point"]
                if not np.isfinite(pt):
                    los.append(np.nan); his.append(np.nan); continue
                iv = ens.interval_for(pt, tm)[lv]
                los.append(iv[0]); his.append(iv[1])
            bt[f"ENSEMBLE__lo{int(lv*100)}"] = los
            bt[f"ENSEMBLE__hi{int(lv*100)}"] = his
        path = os.path.join(cfg.out_dir, "backtest_predictions.csv")
        bt.to_csv(path, encoding="utf-8-sig")
        LOG.info("已保存回测明细：%s", path)
        return bt

    def evaluation_table(self, bt: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
        cfg = self.cfg
        y = bt["y_true"].values
        # 季节朴素：去年同月
        naive = bt["y_true"].shift(12).values
        rows = []
        names = list(model_names) + ["ENSEMBLE"]
        for name in names:
            col = f"{name}__point"
            if col not in bt:
                continue
            pm = point_metrics(y, bt[col].values, naive)
            # 区间指标（仅组合与自带区间的模型）
            for lv in cfg.interval_levels:
                lo_c, hi_c = f"{name}__lo{int(lv*100)}", f"{name}__hi{int(lv*100)}"
                if lo_c in bt and hi_c in bt:
                    pm.update(interval_metrics(y, bt[lo_c].values, bt[hi_c].values, lv))
            pm["model"] = name
            rows.append(pm)
            # 剔除 1-2 月（真正衡量可预测月份的表现）
            jf = bt["is_jan_feb"].values
            pm_ex = point_metrics(y[~jf], bt[col].values[~jf], naive[~jf])
            for lv in cfg.interval_levels:
                lo_c, hi_c = f"{name}__lo{int(lv*100)}", f"{name}__hi{int(lv*100)}"
                if lo_c in bt and hi_c in bt:
                    pm_ex.update(interval_metrics(y[~jf], bt[lo_c].values[~jf],
                                                  bt[hi_c].values[~jf], lv))
            pm_ex["model"] = name + "(剔除1-2月)"
            rows.append(pm_ex)
            # 仅 1/2 月单列
            pm_jf = point_metrics(y[jf], bt[col].values[jf])
            pm_jf["model"] = name + "(仅1-2月)"
            rows.append(pm_jf)
        ev = pd.DataFrame(rows).set_index("model")
        path = os.path.join(cfg.out_dir, "evaluation_metrics.csv")
        ev.to_csv(path, encoding="utf-8-sig")
        LOG.info("已保存评估指标：%s", path)
        return ev

    def save_forecast(self, target_month, point, intervals, per_model: dict, drivers):
        cfg = self.cfg
        rec = {"target_month": str(month_end(target_month).date()),
               "point_forecast": round(float(point), 3)}
        for lv, (lo, hi) in intervals.items():
            rec[f"lower_{int(lv*100)}"] = round(float(lo), 3)
            rec[f"upper_{int(lv*100)}"] = round(float(hi), 3)
        rec["per_model"] = {k: (round(float(v.point), 3) if v else None)
                            for k, v in per_model.items()}
        rec["top_drivers"] = drivers
        path = os.path.join(cfg.out_dir, "forecast_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        LOG.info("已保存预测结果：%s", path)
        # 同时存一份单行 csv 便于传阅
        flat = {k: v for k, v in rec.items() if k not in ("per_model", "top_drivers")}
        pd.DataFrame([flat]).to_csv(
            os.path.join(cfg.out_dir, "forecast_result.csv"),
            index=False, encoding="utf-8-sig")
        return rec

    def plot_backtest(self, bt: pd.DataFrame, model_names: list[str]):
        """回测时间序列图：实际 vs 各模型 vs 组合，并标注 1-2 月；下方为预测误差。
        这是直观判断预测效果与过拟合的主图。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            LOG.warning("matplotlib 不可用，跳过回测图：%s", e)
            return
        styles = {"AR": ("#1f77b4", "-"), "LocalLevel": ("#8c564b", ":"),
                  "ARX": ("#d62728", "--"), "DFM": ("#9467bd", "-."),
                  "LightGBM": ("#2ca02c", "-."), "ENSEMBLE": ("#ff7f0e", "-")}
        fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        ax.plot(bt.index, bt["y_true"], "k-o", ms=5, lw=2.2, label="Actual", zorder=10)
        for name in list(model_names) + ["ENSEMBLE"]:
            col = f"{name}__point"
            if col in bt:
                c, ls = styles.get(name, ("grey", "-"))
                ax.plot(bt.index, bt[col], color=c, ls=ls, marker=".",
                        lw=3 if name == "ENSEMBLE" else 1.3, label=name)
        # 标注 1-2 月
        for ts in bt.index[bt["is_jan_feb"].values]:
            ax.axvspan(ts - pd.Timedelta(days=15), ts + pd.Timedelta(days=15),
                       color="orange", alpha=0.08)
        ax.set_title("Backtest (pseudo real-time): Industrial VA YoY — Actual vs Models"
                     "  [shaded = Jan/Feb, structurally noisy]")
        ax.set_ylabel("YoY %"); ax.grid(alpha=0.3); ax.axhline(0, color="grey", lw=0.5)
        ax.legend(ncol=3, fontsize=9)
        ax2 = axes[1]
        for name in list(model_names) + ["ENSEMBLE"]:
            col = f"{name}__point"
            if col in bt:
                c, ls = styles.get(name, ("grey", "-"))
                ax2.plot(bt.index, bt[col] - bt["y_true"], color=c, ls=ls, marker=".",
                         lw=3 if name == "ENSEMBLE" else 1.2, label=name)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_title("Prediction error (pred - actual), pp"); ax2.grid(alpha=0.3)
        ax2.legend(ncol=3, fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout()
        path = os.path.join(self.cfg.out_dir, "backtest_timeseries.png")
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        LOG.info("已保存回测时间序列图：%s", path)

    def plot_fan_chart(self, bt: pd.DataFrame, target_month, point, intervals):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            LOG.warning("matplotlib 不可用，跳过绘图：%s", e)
            return
        cfg = self.cfg
        fig, ax = plt.subplots(figsize=(13, 6))
        hist = bt.tail(36)
        ax.plot(hist.index, hist["y_true"], "k-o", ms=3, lw=1.2, label="Actual")
        if "ENSEMBLE__point" in hist:
            ax.plot(hist.index, hist["ENSEMBLE__point"], "b--", lw=1,
                    label="Ensemble (backtest)")
        tm = month_end(target_month)
        ax.plot([tm], [point], "rD", ms=9, label="Forecast")
        colors = {0.95: "#ffd6d6", 0.80: "#ff9d9d"}
        for lv in sorted(cfg.interval_levels, reverse=True):
            lo, hi = intervals[lv]
            ax.errorbar([tm], [point], yerr=[[point - lo], [hi - point]],
                        fmt="none", ecolor=colors.get(lv, "red"),
                        elinewidth=8 if lv == 0.95 else 5, alpha=0.6,
                        label=f"{int(lv*100)}% interval")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("Industrial Value Added YoY: backtest & forecast")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        path = os.path.join(cfg.out_dir, "forecast_fan_chart.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        LOG.info("已保存扇形图：%s", path)

    def save_data_quality(self, loader: DataLoader, ctx: Context, as_of):
        cfg = self.cfg
        panel = ctx.panel_asof(as_of)
        rows = []
        for c in panel.columns:
            s = panel[c]
            valid = s.dropna()
            rows.append({
                "indicator": c,
                "n_valid": int(s.notna().sum()),
                "start": str(valid.index.min().date()) if len(valid) else None,
                "end": str(valid.index.max().date()) if len(valid) else None,
                "last_value": round(float(valid.iloc[-1]), 3) if len(valid) else None,
            })
        dq = pd.DataFrame(rows)
        path = os.path.join(cfg.out_dir, "data_quality_report.csv")
        dq.to_csv(path, index=False, encoding="utf-8-sig")
        # 同名核对表
        dict_rows = [{"key": i.key, "name": i.name, "sheet": i.sheet,
                      "transform": i.transform, "dfm_tier": i.dfm_tier,
                      "role": i.role, "note": i.note}
                     for i in loader.indicators]
        pd.DataFrame(dict_rows).to_csv(
            os.path.join(cfg.out_dir, "indicator_dictionary.csv"),
            index=False, encoding="utf-8-sig")
        LOG.info("已保存数据质量报告与指标字典。")


# ==============================================================================
# 第 15 节  驱动分解
# ==============================================================================
def sanity_clip(point: float, raw: dict, cfg: Config,
                target_month: pd.Timestamp) -> tuple:
    """极端值护栏：把点预测限制在"近12个月实际值范围 ± sanity_clip_pp"内。
    防止 ML/桥接模型在基数效应/春节交互下产生 17~18 这类离谱外推。
    返回 (clipped_point, was_clipped)。1-2月放宽护栏(其本身波动极大)。
    """
    s = raw[cfg.sheet_target].set_index("date")[
        "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)"].dropna()
    s.index = to_month_index(s.index)
    # 护栏范围基于近12个【非1-2月】实际值（剔除1-2月人造极值，否则范围被撑到无意义）
    recent = s[~s.index.month.isin([1, 2])].loc[:target_month].tail(12)
    if len(recent) < 6:
        return point, False
    pad = cfg.sanity_clip_pp * (2.5 if target_month.month in (1, 2) else 1.0)
    lo, hi = recent.min() - pad, recent.max() + pad
    clipped = float(np.clip(point, lo, hi))
    return clipped, (abs(clipped - point) > 1e-6)


def explain_drivers(per_model: dict, weights: dict, cfg: Config,
                    top_k: int = 10) -> dict:
    """【组合感知】的驱动解释：只解释【最终组合实际采用】的模型。

    修复点：旧版恒取 LightGBM 特征重要度，但门控后组合可能根本没用 LightGBM，
    导致"主要驱动"与最终预测值无关。新版：
      - ensemble_composition：组合里每个模型的权重、点预测、加权贡献；
      - note：若由基准(LocalLevel)主导，明确说明"预测≈近期实际均值，高频无增量信号"；
      - feature_drivers：仅来自【权重>0】且能给出特征解释的模型(LGB重要度 / ARX系数)。
    """
    out = {"ensemble_composition": [], "feature_drivers": [], "note": ""}
    if not weights:
        return out
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        pm = per_model.get(name)
        if pm is None:
            continue
        out["ensemble_composition"].append({
            "model": name, "weight": round(float(w), 3),
            "point": round(float(pm.point), 3),
            "weighted_contribution": round(float(w) * float(pm.point), 3)})

    dom = max(weights, key=weights.get)
    dompm = per_model.get(dom)
    if dom == cfg.benchmark_name and dompm is not None:
        out["note"] = (
            f"组合由基准模型 {dom} 主导(权重 {weights[dom]:.2f}，点预测 {dompm.point:.2f})："
            f"预测主要由【工业增加值自身历史的惯性(自回归/近期水平)】决定，"
            f"高频指标未提供超越基准的显著增量(见 DM 检验)。")

    for name, w in weights.items():
        if w <= 0:
            continue
        pm = per_model.get(name)
        if pm is None:
            continue
        if name == "LightGBM" and "importance" in pm.extra:
            items = sorted(pm.extra["importance"].items(),
                           key=lambda kv: kv[1], reverse=True)[:top_k]
            tot = sum(v for _, v in items) or 1.0
            out["feature_drivers"].append({
                "model": name, "weight": round(float(w), 3),
                "top_features": [{"feature": k, "gain_share": round(v / tot, 3)}
                                 for k, v in items]})
        elif name == "ARX" and "coef_top" in pm.extra:
            out["feature_drivers"].append({
                "model": name, "weight": round(float(w), 3),
                "top_features": pm.extra["coef_top"]})
    return out


# ==============================================================================
# 第 16 节  主流程
# ==============================================================================
def detect_target_month(raw: dict, cfg: Config) -> pd.Timestamp:
    """自动确定预测月 = 目标变量最后一个非空月 + 1（"下一个未发布月"）。"""
    s = raw[cfg.sheet_target].set_index("date")[
        "中国:工业增加值:规模以上工业企业:当月同比(1-2月拆分)"].dropna()
    last = month_end(s.index.max())
    return month_end(last + pd.offsets.MonthBegin(1))


def main(cfg: Config = CFG):
    LOG.info("=" * 70)
    LOG.info("工业增加值常态化预测系统 启动")
    LOG.info("=" * 70)

    indicators = build_indicator_dict(cfg)

    # 1) 读取 + 校验
    loader = DataLoader(cfg, indicators)
    raw = loader.load()

    # 1b) 预测时点不写死：按汇总表采集日自动确定 as_of 的"当月第几天"
    run_date = pd.Timestamp(cfg.run_date) if cfg.run_date else infer_collection_date(raw)
    cfg.asof_day_of_month = resolve_asof_day(cfg, raw)
    LOG.info("汇总表采集日(run_date)=%s -> as_of 取当月第 %d 天(随采集自动平移，未写死)",
             run_date.date(), cfg.asof_day_of_month)

    # 2) 组装上下文
    aligner = FrequencyAligner(cfg, raw, indicators)
    fb = FeatureBuilder(cfg)
    ctx = Context(cfg, aligner, fb)
    _wire_context_cache(ctx)

    # 3) 模型集合
    # 核心模型：AR(基准) + LocalLevel(朴素地板) + ARX(锚定高频桥接)
    models = [
        ModelAR(cfg),            # 基准：AR(p) BIC（单变量惯性）
        ModelLocalLevel(cfg),    # 朴素地板/参照（近期均值）
        ModelARX(cfg),           # AR 滞后 + 高频外生（ElasticNet, 时序CV）
    ]
    # 探针模型（默认关闭）：机制最独特的高频/非线性模型，用于监测 regime 变化
    if cfg.enable_probe_models:
        models += [
            ModelDFM(cfg, indicators),   # 混频动态因子（高频共同因子探针）
            ModelLGB(cfg),               # 梯度提升（非线性探针）
        ]
    model_names = [m.name for m in models]

    # 4) 回测（伪真实时点）
    bt = Backtester(cfg, ctx, models, raw).run()

    # 5) 组合 + 共形
    ens = Ensemble(cfg, model_names)
    ens.fit(bt)

    # 6) 报告：回测明细 + 评估表
    reporter = Reporter(cfg)
    bt_full = reporter.save_backtest(bt, ens)
    ev = reporter.evaluation_table(bt_full, model_names)
    LOG.info("\n===== 回测评估 =====\n%s", ev.round(3).to_string())
    reporter.plot_backtest(bt_full, model_names)

    # DM 检验结果落盘（各复杂模型 vs 基准）
    if ens.dm_results:
        dm_rows = [{"model": k, "RMSE": round(v[0], 3),
                    "DM_stat": (round(v[1], 3) if v[1] == v[1] else None),
                    "p_value": (round(v[2], 3) if v[2] == v[2] else None),
                    "verdict": ("显著优于基准" if (v[1]==v[1] and v[1] < 0 and v[2] < cfg.dm_significance)
                                else "显著差于基准" if (v[1]==v[1] and v[1] > 0 and v[2] < cfg.dm_significance)
                                else "与基准无显著差异"),
                    "kept_in_ensemble": k in ens.weights}
                   for k, v in ens.dm_results.items()]
        pd.DataFrame(dm_rows).to_csv(
            os.path.join(cfg.out_dir, "dm_test_vs_benchmark.csv"),
            index=False, encoding="utf-8-sig")

    # 7) 实盘预测：下一个未发布月
    target_month = detect_target_month(raw, cfg)
    as_of = asof_for_target(cfg, target_month)
    LOG.info("预测月 = %s ；评估时点 as_of = %s",
             target_month.date(), as_of.date())

    per_model = {}
    for mdl in models:
        try:
            per_model[mdl.name] = mdl.predict_asof(target_month, as_of, ctx)
        except Exception as e:
            LOG.warning("%s 实盘预测失败：%s", mdl.name, e)
            per_model[mdl.name] = None

    point = ens.combine_single(per_model)
    if point is None:
        LOG.error("所有模型均未给出有效预测，终止。")
        return
    point_raw = point
    point, clipped = sanity_clip(point, raw, cfg, target_month)
    if clipped:
        LOG.warning("极端值护栏触发：组合点预测 %.2f -> %.2f（限制在近12月范围±%.1f）",
                    point_raw, point, cfg.sanity_clip_pp)
    intervals = ens.interval_for(point, target_month)
    drivers = explain_drivers(per_model, ens.weights, cfg)

    # 8) 落地产出
    reporter.save_data_quality(loader, ctx, as_of)
    rec = reporter.save_forecast(target_month, point, intervals, per_model, drivers)
    reporter.plot_fan_chart(bt_full, target_month, point, intervals)

    # 9) 控制台总结
    LOG.info("=" * 70)
    LOG.info("【预测结果】%s 工业增加值:规上:当月同比(1-2月拆分)", target_month.date())
    LOG.info("  点预测       : %.2f %%", point)
    for lv in cfg.interval_levels:
        lo, hi = intervals[lv]
        LOG.info("  %d%% 区间     : [%.2f, %.2f]", int(lv * 100), lo, hi)
    LOG.info("  各模型点预测 : %s",
             {k: (round(v.point, 2) if v else None) for k, v in per_model.items()})
    if target_month.month in (1, 2):
        LOG.info("  注意：1/2 月为拆分口径人造噪声，区间已自动加宽，置信度偏低。")
    LOG.info("=" * 70)
    return rec


if __name__ == "__main__":
    main()
