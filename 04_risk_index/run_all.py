"""전체 파이프라인: 데이터 로드 -> 변수/지수 -> 민감도 -> CSV·PNG 저장.

사용 예:
  python run_all.py            # data/ 폴더의 파일로 바로 실행 (인자 불필요)
  python run_all.py --tows "data/tow_*.xlsx" --disability data/disability_by_gu.csv \
                    --parking data/parking_zones.csv --out results
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
from config import VARS
from load_data import load_tows, load_visual_disability, load_parking
from risk_index import build_variables, composite, scores_table, axis_contributions
from sensitivity import scenarios, monte_carlo, summary_stats
import figures

ap = argparse.ArgumentParser()
ROOT = Path(__file__).parent
ap.add_argument("--tows", default=str(ROOT / "data" / "tow_*.xlsx"), help="견인 현황 xlsx (glob 패턴)")
ap.add_argument("--disability", default=str(ROOT / "data" / "disability_by_gu.csv"))
ap.add_argument("--parking", default=str(ROOT / "data" / "parking_zones.csv"))
ap.add_argument("--out", default=str(ROOT / "results"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

x = build_variables(load_tows(a.tows), load_visual_disability(a.disability), load_parking(a.parking))
x["종합지수"] = composite(x); x["순위"] = x["종합지수"].rank(ascending=False, method="min").astype(int)
contrib = axis_contributions(x)
table = x[["O", "T", "nm", "h1", "h2", "v1", "g1", "park", "종합지수", "순위"]].join(contrib.add_suffix(" 기여"))
table = table.rename(columns={"O": "점자블록 견인(유효 월)", "T": "전체 견인(유효 월)", "nm": "유효 월 수", "park": "주차구역 수"})
table.sort_values("순위").round(3).to_csv(out / "risk_index.csv", encoding="utf-8-sig")
alts = scenarios(x); res = monte_carlo(x); rho, corr, scorr, vif = summary_stats(x, alts)
alts.rank(ascending=False, method="min").to_csv(out / "scenario_ranks.csv", encoding="utf-8-sig")
res.to_csv(out / "monte_carlo.csv", encoding="utf-8-sig")
corr.round(3).to_csv(out / "correlation.csv", encoding="utf-8-sig")

figures.setup_fonts()
figures.fig_ranking(x, out / "1_ranking_contribution.png"); figures.fig_heatmap(x, out / "2_variable_scores.png")
figures.fig_stability(res, out / "3_rank_stability.png"); figures.fig_shrinkage(x, out / "4_shrinkage.png")
p = x.attrs
print(f"k={p['k']:.2f} mu={p['mu']:.3f} kappa={p['kappa']:.2f}")
print(table.sort_values("순위")[["종합지수", "순위"]].round(1).head(10).to_string())
print("Spearman vs 기본:", {k: round(v, 3) for k, v in rho.items()})
