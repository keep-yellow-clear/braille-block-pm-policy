"""전체 파이프라인: 데이터 로드 -> 자치구 선정 -> 변수/지수 -> 민감도 -> 동 단위 주차구역 공백 -> CSV·PNG 저장.

입력은 저장소 공용 data/ 폴더에서 찾습니다 (경로·이름 규칙은 src/config.py 의 INPUT_PATTERNS):
  data/raw/        견인 현황 xlsx(2025), 장애인 현황 csv, 전동킥보드 주차구역 현황 csv
  data/processed/  주차구역_도로명주소_법정동.csv

사용 예:
  python run_all.py            # 인자 없이 실행
  python run_all.py --tows "../data/raw/*견인*2025*.xlsx" --out results   # 경로 직접 지정
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import figures
from config import VARS, YEAR
from load_data import find_inputs, load_tows, load_visual_disability, load_parking
from parking_gap import dong_gap
from risk_index import monthly_table, select_districts, build_variables, composite, scores_table, axis_contributions
from sensitivity import scenarios, monte_carlo, summary_stats

ap = argparse.ArgumentParser()
ap.add_argument("--tows", help="견인 현황 xlsx (glob 패턴)")
ap.add_argument("--disability", help="장애인 현황(장애유형별·연령별) csv")
ap.add_argument("--parking", help="전동킥보드 주차구역 현황 csv")
ap.add_argument("--dong-map", help="도로명 주소 주차구역의 법정동 보완표 csv")
ap.add_argument("--out", default=str(ROOT / "results"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
csv = dict(encoding="utf-8-sig")

# 1. 데이터
tows = load_tows(find_inputs("tows", a.tows))
visual = load_visual_disability(find_inputs("disability", a.disability)[0])
parking = load_parking(find_inputs("parking", a.parking)[0], find_inputs("dong_map", a.dong_map)[0])

# 2. 자치구 선정 -> 변수 -> 종합지수
mv, excluded = select_districts(monthly_table(tows), all_districts=visual.index)
x = build_variables(mv, visual, parking)
x["종합지수"] = composite(x)
x["순위"] = x["종합지수"].rank(ascending=False, method="min").astype(int)
contrib = axis_contributions(x)
scores = scores_table(x)
table = (x[["O", "T", "nm", "h1_raw", "h1", "h2_raw", "h2", "v1", "park", "g1", "종합지수", "순위"]]
         .join(scores.add_suffix(" 점수")).join(contrib.add_suffix(" 기여"))
         .rename(columns={"O": "점자블록 견인(유효 월)", "T": "전체 견인(유효 월)", "nm": "유효 월 수",
                          "park": "주차구역 수"}))
table.index.name = "구"
table = table.sort_values("순위")
table.round(3).to_csv(out / "risk_index.csv", **csv)
excluded.round(1).rename_axis("구").to_csv(out / "excluded_districts.csv", **csv)

# 3. 민감도
alts, weights = scenarios(x)
res = monte_carlo(x)
rho, corr, scorr, vif = summary_stats(x, alts)
alts.rank(ascending=False, method="min").astype(int).rename_axis("구").to_csv(out / "scenario_ranks.csv", **csv)
weights.round(3).assign(**{"기본 순위와의 스피어만 ρ": [1.0, rho["CRITIC"], rho["엔트로피"], rho["축 균등"]]}) \
    .round(3).rename_axis("방식").to_csv(out / "weight_methods.csv", **csv)
res.rename_axis("구").to_csv(out / "monte_carlo.csv", **csv)
corr.round(3).to_csv(out / "correlation.csv", **csv)

# 4. 동 단위 주차구역 공백
gap, gap_summary = dong_gap(tows, parking)
gap.to_csv(out / "dong_parking_gap.csv", **csv)
parking[["gu", "addr", "detail", "rack", "dong_lot", "dong"]].rename(columns={
    "gu": "구", "addr": "주소", "detail": "상세위치", "rack": "거치대",
    "dong_lot": "법정동(보완 전)", "dong": "법정동(보완 후)"}).to_csv(out / "parking_zones_with_dong.csv", index=False, **csv)

# 5. 그림
figures.setup_fonts()
figures.fig_ranking(table, out / "1_ranking_contribution.png")
figures.fig_heatmap(scores, x["종합지수"], out / "2_variable_scores.png")
figures.fig_stability(res, out / "3_rank_stability.png")
figures.fig_shrinkage(x, out / "4_shrinkage.png")
figures.fig_excluded(excluded, out / "5_excluded_districts.png")
figures.fig_parking_gap(gap, out / "6_dong_parking_gap.png")

# 6. 요약 출력
p = x.attrs
print(f"\n[{YEAR}] 분석 대상 {len(x)}개 구, 제외 {len(excluded)}개 구: {', '.join(excluded.index)}")
print(f"보정 모수: k={p['k']:.3f}  mu={p['mu']:.3f}  kappa={p['kappa']:.2f}")
print(table[["종합지수", "순위"]].round(1).head(10).to_string())
print("Spearman vs 기본:", {k: round(v, 3) for k, v in rho.items()})
print(f"몬테카를로 ρ: 평균 {res.attrs['rho_mean']:.2f}, 최소 {res.attrs['rho_min']:.2f}")
print("VIF:", vif.round(2).to_dict())
print("동 단위 주차구역 공백:", {k: (f"{v:.0%}" if isinstance(v, float) else v) for k, v in gap_summary.items()})
print(f"\n결과 저장: {out}")
