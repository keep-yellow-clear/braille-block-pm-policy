"""자치구 선정, 변수 계산(H1·H2·V1·G1), 점수화, 종합지수."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammaln

from config import (VARS, DIRECTION, WEIGHTS, AXES, TRIM,
                    MIN_MONTHLY_TOWS, VALID_MONTH_THR, MIN_VALID_MONTHS)


def monthly_table(tows):
    """자치구×월 집계: 전체 견인(T), 점자블록 견인(O), 반복 점자블록 견인(R).
    R = 같은 달 같은 주소에서 2건 이상 나온 점자블록 견인 건수."""
    months = sorted(tows["month"].unique())
    T = tows.groupby(["gu", "month"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    b = tows[tows["braille"]]
    O = b.groupby(["gu", "month"]).size()
    per_addr = b.groupby(["gu", "month", "addr"]).size()
    R = per_addr[per_addr >= 2].groupby(["gu", "month"]).sum()
    m = T.stack().rename("T").to_frame()
    m["O"] = O.reindex(m.index, fill_value=0)
    m["R"] = R.reindex(m.index, fill_value=0)
    return m.reset_index()


def select_districts(m, all_districts=None):
    """all_districts: 서울 25개 구 목록 (견인이 0건인 구를 빠뜨리지 않기 위해).
    ① 월평균 견인 < MIN_MONTHLY_TOWS 인 구 제외 (견인 사업이 사실상 운영되지 않음)
    ② 유효 월 = 그 달 견인 >= 그 구 최대 월 견인 x VALID_MONTH_THR (보고가 끊긴 달 제외)
    ③ 유효 월 < MIN_VALID_MONTHS 인 구 제외"""
    n_months = m["month"].nunique()
    m = m.copy()
    m["valid"] = (m["T"] >= VALID_MONTH_THR * m.groupby("gu")["T"].transform("max")) & (m["T"] > 0)
    g = m.groupby("gu").agg(전체견인=("T", "sum"), 점자블록견인=("O", "sum"), 유효월=("valid", "sum"))
    if all_districts is not None:   # 견인 기록이 한 건도 없는 구도 제외 목록에 남김
        g = g.reindex(sorted(set(all_districts) | set(g.index)), fill_value=0)
    g["월평균견인"] = g["전체견인"] / n_months
    low = g["월평균견인"] < MIN_MONTHLY_TOWS
    few = ~low & (g["유효월"] < MIN_VALID_MONTHS)
    g["제외 이유"] = np.select([low, few],
                             [f"월평균 견인 {MIN_MONTHLY_TOWS}건 미만 (견인 사업 사실상 미운영)",
                              f"유효 월 {MIN_VALID_MONTHS}개월 미만"], default="")
    keep = g.index[g["제외 이유"] == ""]
    excluded = g[g["제외 이유"] != ""].sort_values("전체견인")
    return m[m["gu"].isin(keep) & m["valid"]], excluded


def build_variables(mv, visual, parking):
    """mv: 분석 대상 구의 유효 월 행.  visual: 구별 시각장애인 수.  parking: 주차구역 표."""
    # 같은 달 도시 평균 점자블록 비율 -> 기대 점자블록 견인 건수
    city = mv.groupby("month")[["O", "T"]].sum()
    mv = mv.assign(E=mv["T"] * mv["month"].map(city["O"] / city["T"]))
    x = mv.groupby("gu").agg(O=("O", "sum"), T=("T", "sum"), R=("R", "sum"),
                             E=("E", "sum"), nm=("month", "count"))

    # H1 점자블록 견인 지수 = (O + k) / (E + k), 감마-포아송 경험적 베이즈
    def nll_k(log_k):
        k, o, e = np.exp(log_k), x["O"].values, x["E"].values
        return -np.sum(gammaln(o + k) - gammaln(k) - gammaln(o + 1)
                       + k * np.log(k / (k + e)) + o * np.log(e / (k + e)))
    k = float(np.exp(minimize_scalar(nll_k, bounds=(-5, 8), method="bounded").x))
    x["h1_raw"] = x["O"] / x["E"]
    x["h1"] = (x["O"] + k) / (x["E"] + k)

    # H2 상습 지점 비중 = (R + κμ) / (O + κ), 베타-이항 경험적 베이즈 (μ, κ 주변우도 추정)
    def nll_beta(p):
        mu, kap = 1 / (1 + np.exp(-p[0])), np.exp(p[1])
        a, b = mu * kap, (1 - mu) * kap
        return -np.sum(betaln(x["R"] + a, x["O"] - x["R"] + b) - betaln(a, b))
    r = minimize(nll_beta, [-1.2, 2.0], method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    mu, kappa = float(1 / (1 + np.exp(-r.x[0]))), float(np.exp(r.x[1]))
    x["h2_raw"] = (x["R"] / x["O"]).fillna(0)
    x["h2"] = (x["R"] + kappa * mu) / (x["O"] + kappa)

    # V1 시각장애인 수, G1 주차구역 밀도 = 주차구역 수 ÷ 월평균 견인 × 100
    x["v1"] = visual.reindex(x.index).astype(float)
    x["park"] = parking.groupby("gu").size().reindex(x.index, fill_value=0)
    x["g1"] = x["park"] / (x["T"] / x["nm"]) * 100

    x.attrs.update(k=k, mu=mu, kappa=kappa)
    return x


def to_score(s, direction=1, trim=TRIM, mode="minmax"):
    """0~100 점수. minmax: 양끝 절사 후 min-max, pct: 백분위."""
    if mode == "pct":
        sc = s.rank(pct=True) * 100
    else:
        lo, hi = s.quantile(trim), s.quantile(1 - trim)
        sc = (s.clip(lo, hi) - lo) / (hi - lo) * 100
    return sc if direction == 1 else 100 - sc


def scores_table(x, trim=TRIM, mode="minmax", vars_=VARS):
    return pd.DataFrame({k: to_score(x[k], DIRECTION[k], trim, mode) for k in vars_})


def composite(x, weights=None, vars_=VARS, trim=TRIM, mode="minmax"):
    w = pd.Series(weights or WEIGHTS)[vars_]
    return scores_table(x, trim, mode, vars_) @ (w / w.sum())


def axis_contributions(x):
    """축별 기여 (합 = 종합지수)."""
    s = scores_table(x)
    w = pd.Series(WEIGHTS) / sum(WEIGHTS.values())
    return pd.DataFrame({ax: sum(s[v] * w[v] for v in vs) for ax, vs in AXES.items()})
