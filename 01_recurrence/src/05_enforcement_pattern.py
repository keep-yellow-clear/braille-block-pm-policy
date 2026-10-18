"""5단계: 관측된 재발 간격에 '단속 주기'가 얼마나 섞여 있는지 점검.

관측 간격 = 실제 재주차까지 시간 + 신고까지 시간 + 견인이 운영되는 날까지 대기
→ 데이터로 '단속 쪽' 패턴(요일, 구별 운영일, 고정 순찰 주기 여부)을 확인한다.

입력: data/raw/ 2025년 견인 현황 원자료 2개 (config.load_tow_2025), data/processed/intervals_main.csv (03단계)
출력: results/enforcement_pattern.xlsx
"""
import numpy as np
import pandas as pd
from config import PROC_DIR, RES_DIR, load_tow_2025

WD = ["월", "화", "수", "목", "금", "토", "일"]

ev = pd.read_csv(PROC_DIR / "intervals_main.csv", parse_dates=["일", "다음견인일"])
r = ev[ev["event"] == 1].copy()

df = load_tow_2025(drop_false_report=False)            # 원자료 직접 읽기 (구별 운영일수는 원래대로 오신고 포함)
df["일"] = df["신고일"].dt.normalize()

# (1) 요일 분포: 견인이 평일에만 이뤄지는가
weekday = (ev["일"].dt.dayofweek.value_counts(normalize=True).sort_index() * 100).round(1)
weekday.index = WD

# (2) 영업일 기준 간격
r["영업일간격"] = np.busday_count(r["일"].values.astype("datetime64[D]"),
                                r["다음견인일"].values.astype("datetime64[D]"))
fri_mon = ((r["gap"] == 3) & (r["일"].dt.dayofweek == 4)).sum() / (r["gap"] == 3).sum()

# (3) 고정 순찰 주기 여부: 7·14일 간격에 몰리는가, 지번별 특정 요일 집중도
gap_dist = (r["gap"].value_counts(normalize=True).sort_index() * 100).round(2).loc[1:30]
top_wd_share = ev.groupby("주소")["일"].apply(lambda s: s.dt.dayofweek.value_counts(normalize=True).max())

# (4) 구별 견인 운영일수
gu_days = df.groupby("구")["일"].nunique().sort_values(ascending=False).rename("2025 견인 발생일수")
gu_days = gu_days[gu_days.index.isin(ev["구"].unique())]

summary = pd.DataFrame({
    "지표": ["평일 견인 비중", "달력일 간격 중앙값", "영업일 간격 중앙값", "1일(다음날) 재발 비중",
           "3일 간격 중 금→월 비중", "지번별 최다 요일 비중(중앙값)", "요일 무관 시 기대값"],
    "값": [f"{weekday.iloc[:5].sum():.1f}%", f"{r['gap'].median():.0f}일", f"{r['영업일간격'].median():.0f}영업일",
          f"{(r['gap'] == 1).mean()*100:.1f}%", f"{fri_mon*100:.1f}%", f"{top_wd_share.median()*100:.0f}%", "20%"]})
print(summary.to_string(index=False)); print(weekday.to_string()); print(gu_days.to_string())

with pd.ExcelWriter(RES_DIR / "enforcement_pattern.xlsx") as w:
    summary.to_excel(w, sheet_name="요약", index=False)
    weekday.rename("견인일 비중(%)").to_frame().to_excel(w, sheet_name="요일분포")
    gap_dist.rename("비중(%)").to_frame().rename_axis("간격(일)").to_excel(w, sheet_name="간격분포_1~30일")
    gu_days.to_frame().to_excel(w, sheet_name="구별 운영일수")
