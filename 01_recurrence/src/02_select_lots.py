"""2단계: 2025년 기준 상습 견인 지번을 선정한다.

핵심 아이디어
- '견인 건수'가 아니라 '견인일수'(견인이 있었던 서로 다른 날짜 수)로 센다.
  하루에 10대를 한꺼번에 견인해도 1일. 일괄 견인 때문에 건수가 부풀려지는 것을 막는다.
- '견인 월수'(12개월 중 견인이 있었던 달 수)로 특정 시기에만 몰린 지번을 걸러낸다.

입력: data/raw/ 2025년 견인 현황 원자료 2개 (config.load_tow_2025)
출력: data/processed/lots_2025.csv (전체 지번 집계)
      data/processed/top_lots_main.csv, top_lots_strict.csv
"""
import pandas as pd
from config import PROC_DIR, SELECTION, load_tow_2025


def lot_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()                                      # load_tow_2025: 2025년·오신고 제외
    d["일"] = d["신고일"].dt.normalize()
    d["월"] = d["신고일"].dt.month
    return (d.groupby("주소")
             .agg(구=("구", "first"), 견인일수=("일", "nunique"),
                  견인월수=("월", "nunique"), 견인건수=("일", "size"))
             .reset_index()
             .sort_values(["견인일수", "견인건수", "주소"], ascending=[False, False, True])  # 동점 순서 고정
             .set_index("주소"))


if __name__ == "__main__":
    df = load_tow_2025()
    g = lot_summary(df)
    g.to_csv(PROC_DIR / "lots_2025.csv", encoding="utf-8-sig")

    print(f"2025년 견인 발생 지번: {len(g):,}곳")
    print(f"견인일수 99백분위 = {g['견인일수'].quantile(.99):.0f}일 (상위 1% 기준선)")
    for key, s in SELECTION.items():
        top = g[(g["견인일수"] >= s["min_days"]) & (g["견인월수"] >= s["min_months"])].copy()
        top.insert(0, "순위", range(1, len(top) + 1))
        top.to_csv(PROC_DIR / f"top_lots_{key}.csv", encoding="utf-8-sig")
        share = top["견인건수"].sum() / g["견인건수"].sum() * 100
        print(f"[{s['label']}] 견인일수≥{s['min_days']} & 월수≥{s['min_months']}: "
              f"{len(top)}곳, 전체 견인건의 {share:.1f}%")
