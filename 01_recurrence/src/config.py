"""공통 설정: 경로, 분석 연도, 상습 지번 선정 기준, 2025년 견인 원자료 불러오기."""
from pathlib import Path
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]   # 01_recurrence/
REPO_ROOT = ANALYSIS_DIR.parent                      # 저장소 최상위 (원자료는 모든 분석이 공유)
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
RES_DIR = ANALYSIS_DIR / "results"
FIG_DIR = RES_DIR / "figures"
for d in (PROC_DIR, RES_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_YEAR = 2025
OBS_END = "2025-12-31"          # 관측 종료일 (이 날까지 재견인 없으면 중도절단)

# 01 재발 주기 분석에 쓰는 원자료: 2025년 견인 현황 2개 파일 (받은 파일명 그대로, 월별 시트)
RAW_TOW_2025 = [
    "서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx",
    "서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx",
]
TOW_COLS = ["번호", "신고일", "구정보", "주소", "유형", "조치일"]

# 상습 지번 선정 기준 (견인일수 = 같은 날 여러 대 견인은 1일로 셈)
SELECTION = {
    "main":   {"label": "본분석(175곳)", "min_days": 24, "min_months": 9},   # 견인일수 상위 1% + 9개월 이상
    "strict": {"label": "확인용(102곳)", "min_days": 30, "min_months": 10},  # 더 엄격한 기준 (민감도 분석)
}

# 차트 색상 (본분석 / 확인용)
COLORS = {"main": "#2a78d6", "strict": "#eb6834"}
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 환경에 맞게 수정


def load_tow_2025(drop_false_report: bool = True) -> pd.DataFrame:
    """data/raw의 2025년 견인 현황 원자료(xlsx 2개 × 월별 시트)를 직접 읽어 한 표로 만든다.

    정제 규칙
      - 각 시트 0행은 빈 제목행 → 1행을 헤더로 읽고 앞 6개 컬럼만 사용
      - 신고일·조치일: 날짜형과 문자열이 섞여 있어 날짜로 통일
      - 주소 공백 정리, '서울시 ' → '서울특별시 ' 통일
      - 구: 구정보 컬럼이 일부 잘려 있어(예: '중구 신') 주소 두 번째 단어에서 다시 추출
      - 오신고는 분석에서 제외 (drop_false_report=True)
    """
    frames = []
    for name in RAW_TOW_2025:
        sheets = pd.read_excel(RAW_DIR / name, sheet_name=None, header=1)
        for sheet, d in sheets.items():
            d = d.iloc[:, :6].copy()
            d.columns = TOW_COLS
            d["원본파일"], d["시트"] = name, sheet
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["주소"])
    df["신고일"] = pd.to_datetime(df["신고일"], errors="coerce")
    df["조치일"] = pd.to_datetime(df["조치일"], errors="coerce")
    df["주소"] = (df["주소"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                  .str.replace(r"^서울시 ", "서울특별시 ", regex=True))
    df["구"] = df["주소"].str.split(" ").str[1]
    df["유형"] = df["유형"].astype(str).str.strip()
    df["연"] = df["신고일"].dt.year
    df = df[df["연"] == ANALYSIS_YEAR]
    if drop_false_report:
        df = df[df["유형"] != "오신고"]
    return df.reset_index(drop=True)
