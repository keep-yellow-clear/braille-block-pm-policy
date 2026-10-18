"""가중치·방법 변경에 대한 순위 안정성, 상관, 다중공선성."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import VARS, DIRECTION, WEIGHTS, MC_DRAWS, MC_CONC, SEED
from risk_index import composite, to_score, scores_table


def data_driven_weights(x):
    s = scores_table(x); S = s.values + 1e-9; P = S / S.sum(0)
    e = -(P * np.log(P)).sum(0) / np.log(len(x))
    entropy = pd.Series((1 - e) / (1 - e).sum(), index=VARS)
    critic = s.std(ddof=1) * (1 - s.corr()).sum()
    return entropy, critic / critic.sum()


def scenarios(x) -> pd.DataFrame:
    ent, cri = data_driven_weights(x)
    W = lambda a, b, c, d: dict(h1=a, h2=b, v1=c, g1=d)
    alts = {"기본(변수 동일)": composite(x),
            "축 균등": composite(x, W(1/6, 1/6, 1/3, 1/3)),
            "엔트로피": composite(x, ent.to_dict()), "CRITIC": composite(x, cri.to_dict()),
            "위해 중시": composite(x, W(.35, .35, .15, .15)),
            "취약성 중시": composite(x, W(.2, .2, .4, .2)),
            "대응공백 중시": composite(x, W(.2, .2, .2, .4)),
            "백분위 변환": composite(x, mode="pct"),
            "절사 0%": composite(x, trim=0), "절사 10%": composite(x, trim=.10)}
    for k in VARS:
        alts[f"{k.upper()} 제외"] = composite(x, vars_=[v for v in VARS if v != k])
    return pd.DataFrame(alts, index=x.index)


def monte_carlo(x):
    rng = np.random.default_rng(SEED)
    S = np.column_stack([to_score(x[k], DIRECTION[k]) for k in VARS])
    alpha = np.array([WEIGHTS[k] for k in VARS]) * MC_CONC
    ranks = np.zeros((MC_DRAWS, len(x)))
    for i in range(MC_DRAWS):
        ranks[i] = (-(S @ rng.dirichlet(alpha))).argsort().argsort() + 1
    mc = pd.DataFrame(ranks, columns=x.index)
    base = composite(x)
    res = pd.DataFrame({"기본순위": base.rank(ascending=False, method="min").astype(int),
                        "평균순위": mc.mean().round(1), "최소": mc.min().astype(int), "최대": mc.max().astype(int),
                        "p5": mc.quantile(.05), "p95": mc.quantile(.95),
                        "top5확률": (mc <= 5).mean().round(2), "top10확률": (mc <= 10).mean().round(2),
                        "n": x.O.astype(int)}).sort_values("기본순위")
    return res


def summary_stats(x, alts):
    base = alts["기본(변수 동일)"]
    rho = {k: float(spearmanr(base, v)[0]) for k, v in alts.items()}
    corr = x[VARS].corr(); vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=VARS)
    return rho, corr, x[VARS].corr("spearman"), vif
