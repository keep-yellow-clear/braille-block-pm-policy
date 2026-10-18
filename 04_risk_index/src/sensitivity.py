"""가중치·방법 변경에 대한 순위 안정성, 상관, 다중공선성."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import VARS, WEIGHTS, MC_DRAWS, MC_CONC, SEED
from risk_index import composite, scores_table


def data_driven_weights(x):
    s = scores_table(x)
    P = (s.values + 1e-12) / (s.values + 1e-12).sum(0)
    e = -(P * np.log(P)).sum(0) / np.log(len(x))
    entropy = pd.Series((1 - e) / (1 - e).sum(), index=VARS)
    critic = s.std(ddof=1) * (1 - s.corr()).sum()
    return entropy, critic / critic.sum()


def scenarios(x):
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
    weights = pd.DataFrame({"동일 가중치(기본)": pd.Series(WEIGHTS), "CRITIC": cri, "엔트로피": ent,
                            "축 균등": pd.Series(W(1/6, 1/6, 1/3, 1/3))}).T
    return pd.DataFrame(alts, index=x.index), weights


def monte_carlo(x):
    """동일 가중치를 중심으로 한 디리클레 분포에서 가중치를 MC_DRAWS번 뽑아 순위 분포 계산."""
    S = scores_table(x).values
    alpha = np.array([WEIGHTS[k] for k in VARS]) * MC_CONC
    W = np.random.default_rng(SEED).dirichlet(alpha, MC_DRAWS)
    ranks = (-(S @ W.T)).argsort(0).argsort(0) + 1          # (구 수, 반복 수)
    base = composite(x).rank(ascending=False, method="min").astype(int)
    res = pd.DataFrame({"기본순위": base.values, "평균순위": ranks.mean(1).round(1),
                        "최소": ranks.min(1), "최대": ranks.max(1),
                        "p5": np.percentile(ranks, 5, axis=1).astype(int),
                        "p95": np.percentile(ranks, 95, axis=1).astype(int),
                        "top5확률": (ranks <= 5).mean(1).round(2),
                        "top10확률": (ranks <= 10).mean(1).round(2),
                        "n": x["O"].astype(int).values}, index=x.index).sort_values("기본순위")
    rho = [spearmanr(base.values, ranks[:, j])[0] for j in range(ranks.shape[1])]
    res.attrs.update(rho_mean=float(np.mean(rho)), rho_min=float(np.min(rho)))
    return res


def summary_stats(x, alts):
    base = alts["기본(변수 동일)"]
    rho = {k: float(spearmanr(base, v)[0]) for k, v in alts.items()}
    corr = x[VARS].corr()
    vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=VARS)
    return rho, corr, x[VARS].corr("spearman"), vif
