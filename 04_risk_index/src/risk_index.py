"""변수(H1, H2, V1, G1) 계산, 0~100 점수화, 종합지수."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import betabinom
from config import VARS, DIRECTION, WEIGHTS, AXES, TRIM, VALID_MONTH_THR


def monthly_tables(tows: pd.DataFrame):
    """구x월 표: 전체 견인 T, 점자블록 견인 B, 반복 점자블록 견인 R(같은 달 같은 주소 2건 이상)."""
    months = sorted(tows.ym.unique()); gus = sorted(tows["구"].unique())
    piv = lambda d: (d.pivot_table(index="구", columns="ym", values="번호", aggfunc="size")
                     .reindex(index=gus, columns=months).fillna(0))
    b = tows[tows.b]
    size = b.groupby(["구", "ym", "주소"])["번호"].transform("size")
    return piv(tows), piv(b), piv(b[size >= 2])


def fit_poisson_gamma(O, E) -> float:
    """H1 보정: 평균 1인 감마 사전분포의 모양 모수 k를 주변우도 최대화로 추정."""
    def nll(lk):
        k = np.exp(lk)
        return -np.sum(gammaln(O + k) - gammaln(k) - gammaln(O + 1)
                       + k * np.log(k / (k + E)) + O * np.log(E / (k + E)))
    return float(np.exp(minimize(nll, [1.0], method="Nelder-Mead").x[0]))


def fit_beta_binomial(R, N):
    """H2 보정: 베타-이항 사전분포의 평균 mu와 집중도 kappa를 주변우도 최대화로 추정."""
    def nll(p):
        mu, kap = 1 / (1 + np.exp(-p[0])), np.exp(p[1])
        return -np.sum(betabinom.logpmf(R, N, mu * kap, (1 - mu) * kap))
    r = minimize(nll, [0, np.log(10)], method="Nelder-Mead")
    return float(1 / (1 + np.exp(-r.x[0]))), float(np.exp(r.x[1]))


def build_variables(tows, v1, parking, thr=VALID_MONTH_THR) -> pd.DataFrame:
    T, B, R = monthly_tables(tows)
    valid = (T.ge(thr * T.max(axis=1), axis=0) & (T > 0)).astype(int)   # 보고가 사실상 중단된 달 제외
    Tv, Bv, Rv = T * valid, B * valid, R * valid
    city = (Bv.sum() / Tv.sum()).fillna(0)          # 같은 달 도시 평균 점자블록 비율
    E = Tv.mul(city, axis=1)                         # 기대 점자블록 견인
    a = pd.DataFrame({"T": Tv.sum(1), "O": Bv.sum(1), "E": E.sum(1), "Rep": Rv.sum(1), "nm": valid.sum(1)})
    a["sir"] = a.O / a.E
    k = fit_poisson_gamma(a.O.values, a.E.values)
    a["h1"] = (a.O + k) / (a.E + k)
    mu, kap = fit_beta_binomial(a.Rep.values.astype(int), a.O.values.astype(int))
    a["h2raw"] = np.where(a.O > 0, a.Rep / a.O, 0)
    a["h2"] = (a.Rep + kap * mu) / (a.O + kap)
    a["avg"] = a["T"] / a.nm
    a = a.join(v1).join(parking).fillna(0)
    a["g1"] = a.park / a.avg * 100                   # 월평균 견인 100건당 주차구역 수
    a.attrs.update(k=k, mu=mu, kappa=kap, valid=valid)
    return a


def to_score(v, direction, trim=TRIM, mode="win"):
    """0~100 점수(높을수록 위험). win: 양끝 절사 후 min-max, pct: 백분위."""
    v = np.asarray(v, float)
    if mode == "pct":
        s = pd.Series(v).rank(method="min").sub(1).div(len(v) - 1).mul(100).values
    else:
        lo, hi = np.percentile(v, trim * 100), np.percentile(v, 100 - trim * 100)
        s = 100 * (np.clip(v, lo, hi) - lo) / (hi - lo) if hi > lo else np.full(len(v), 50.0)
    return s if direction > 0 else 100 - s


def composite(x, weights=WEIGHTS, trim=TRIM, mode="win", vars_=VARS):
    w = sum(weights[k] for k in vars_)
    tot = sum(weights[k] * to_score(x[k], DIRECTION[k], trim, mode) for k in vars_) / w
    return pd.Series(tot, index=x.index)


def scores_table(x) -> pd.DataFrame:
    return pd.DataFrame({k: to_score(x[k], DIRECTION[k]) for k in VARS}, index=x.index)


def axis_contributions(x, weights=WEIGHTS) -> pd.DataFrame:
    s = scores_table(x)
    return pd.DataFrame({ax: sum(weights[k] * s[k] for k in ks) for ax, ks in AXES.items()})
