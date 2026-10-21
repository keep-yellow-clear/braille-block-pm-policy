"""분석 설정값. 기간·가중치·절사·임계값·입력 파일 위치는 여기서만 바꿉니다."""
from pathlib import Path

# ── 입력 데이터: 저장소 공용 data/ 폴더 (03_risk_index/ 의 상위 폴더) ──
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR, PROCESSED_DIR = DATA_DIR / "raw", DATA_DIR / "processed"
# 파일 이름 규칙 (glob). 여러 개가 맞으면 견인 현황은 모두 합치고, 나머지는 첫 번째 파일을 씀
INPUT_PATTERNS = {
    "tows": [RAW_DIR / "*견인*.xlsx", RAW_DIR / "*tow*.xlsx"],
    "disability": [RAW_DIR / "*장애인*.csv", RAW_DIR / "*disability*.csv"],
    "parking": [RAW_DIR / "*주차구역*현황*.csv", RAW_DIR / "*parking*.csv"],
    "dong_map": [PROCESSED_DIR / "주차구역_도로명주소_법정동.csv", PROCESSED_DIR / "*법정동*.csv"],
}

YEAR = "2025"                       # 분석 기간: 2025년 1~12월

VARS = ["h1", "h2", "v1", "g1"]
DIRECTION = {"h1": 1, "h2": 1, "v1": 1, "g1": -1}   # +1: 클수록 위험, -1: 클수록 안전(점수 뒤집음)
WEIGHTS = {"h1": 0.25, "h2": 0.25, "v1": 0.25, "g1": 0.25}  # 전문가 판단(AHP) 전 -> 동일 가중치
AXES = {"위해": ["h1", "h2"], "취약성": ["v1"], "대응 공백": ["g1"]}
LABELS = {"h1": "H1 점자블록 견인 지수", "h2": "H2 상습 지점 비중",
          "v1": "V1 시각장애인 수", "g1": "G1 주차구역 밀도"}

TRIM = 0.05             # 양끝 절사 비율 (winsorization)

# 자치구·월 선정
MIN_MONTHLY_TOWS = 50   # 2025년 월평균 견인이 이보다 적은 구는 견인 사업이 사실상 운영되지 않은 것으로 보고 제외
VALID_MONTH_THR = 0.25  # 유효 월: 그 달 견인 >= (그 구 2025년 최대 월 견인) x 이 값
MIN_VALID_MONTHS = 6    # 유효 월이 이보다 적은 구는 제외

BRAILLE_KEYWORD = "점자"  # 견인 '유형'에 이 문자열이 있으면 점자블록 견인
EXCLUDE_TYPES = []        # 분석에서 뺄 견인 유형 (예: ["오신고"]). 기본은 전체 사용

MC_DRAWS, MC_CONC, SEED = 5000, 32, 42   # 가중치 무작위 변동(디리클레): 반복 수, 집중도, 난수 시드

HOTSPOT_MIN_BRAILLE = 10  # 동 단위 주차구역 공백 요약에서 '견인 다발 동'으로 볼 점자블록 견인 건수
