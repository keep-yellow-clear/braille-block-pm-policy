"""1단계: 2025년 견인 원자료 불러오기 확인.

입력: data/raw/서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx
      data/raw/서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx
읽기·정제 규칙은 config.load_tow_2025() 한 곳에 있고, 02·03·05 단계가 모두 이 함수로 원자료를 직접 읽는다.
이 스크립트는 원자료가 제대로 읽히는지 건수만 확인한다.
"""
from config import RAW_TOW_2025, load_tow_2025

if __name__ == "__main__":
    df = load_tow_2025(drop_false_report=False)
    print("원자료:", *RAW_TOW_2025, sep="\n  ")
    print(f"2025년 견인 {len(df):,}건 (오신고 {(df['유형'] == '오신고').sum()}건 포함)")
    print(df.groupby(df["신고일"].dt.month).size().rename("월별 건수").to_string())
