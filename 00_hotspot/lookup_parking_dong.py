"""주차구역 도로명 주소 → 법정동 조회 (주소정보누리집, 무료·키 불필요).

서울시 전동킥보드 주차구역 현황.csv 의 주소 중 일부(약 100곳)는 '테헤란로 152'처럼 도로명 주소라
주소만으로는 법정동을 알 수 없다. 행정안전부 주소정보누리집(juso.go.kr) 검색으로 법정동을 찾아
data/processed/주차구역_도로명주소_법정동.csv 로 저장한다. hotspot_analysis.py 가 이 파일을 읽는다.

입력: data/raw/서울시 전동킥보드 주차구역 현황.csv
출력: data/processed/주차구역_도로명주소_법정동.csv  (구, 주소, 조회주소, 법정동, 지번주소, 근거)

- 검색 결과 중 도로명·건물번호가 조회 주소와 정확히 같은 건만 채택한다.
- 정확히 일치하는 결과가 없는 주소는 MANUAL 에 근거와 함께 직접 적는다
  (같은 도로의 인접 건물번호 결과, 상세위치 건물명 등).
- 결과 파일은 저장소에 함께 올려 두므로, 분석을 다시 돌릴 때 이 스크립트를 실행할 필요는 없다.

실행: python 00_hotspot/lookup_parking_dong.py
"""
import json, re, time, urllib.request
import pandas as pd
from hotspot_analysis import RAW_DIR, load_parking

OUT = RAW_DIR.parent / "processed" / "주차구역_도로명주소_법정동.csv"
URL = "https://www.juso.go.kr/api/solr/solrKeywordSearch"

# 정확 일치 결과가 없는 주소: 조회주소 → (법정동, 참고 지번주소, 근거)
MANUAL = {
    "서울특별시 종로구 세종로 84-8": ("세종로", "서울특별시 종로구 세종로 84-8", "지번 주소 (세종로 = 법정동 이름)"),
    "서울특별시 마포구 양화로 32": ("합정동", "서울특별시 마포구 합정동 383-4 (양화로 33)",
                                "같은 도로 인접 번호 (양화로 29·33·44 모두 합정동)"),
    "서울특별시 마포구 월드컵로3길 1": ("합정동", "월드컵로3길 14 (마포 한강 2차 푸르지오)", "같은 도로 인접 번호, 상세위치 건물 일치"),
    "서울특별시 강남구 봉은사로 607": ("삼성동", "서울특별시 강남구 삼성동 107-9 (봉은사로 605)",
                                  "같은 도로 인접 번호 (봉은사로 605·609 모두 삼성동)"),
    "서울특별시 송파구 잠실역 46": ("잠실동", "석촌호수로 (잠실동)", '도로명 없음, 상세위치 "석촌호수 서호 입구"로 판단'),
}


def search(keyword: str) -> list:
    body = {"keyword": keyword, "strSearchType": "HSTRY", "strFirstSort": "none", "strAblYn": "N",
            "strAotYn": "N", "strSynnYn": None, "strHstryYn": "Y", "reqFrom": "RN_SEARCH_KOR_WEB",
            "checkMoblieYn": "N", "strFunctionName": "Y", "pageable": {"page": 0, "size": 5},
            "strAroundYn": "N", "isLogAccessEnabled": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return (json.load(r).get("results") or {}).get("content") or []


def norm(s: str) -> str:
    """'…로 지하163 (마곡동)' → '…로 163' 비교용."""
    s = re.sub(r"\s*\(.*$", "", s or "").replace("지하", "")
    return re.sub(r"\s+", " ", s).strip()


def road_query(gu: str, addr: str) -> str:
    a = re.sub(r"\s+", " ", str(addr)).strip()
    a = re.sub(r"^서울(특별)?시\s*", "", a)
    a = re.sub(r"^\S+구\s+", "", a)
    a = re.sub(r"^(\S*?(?:로|길))(\d[\d-]*)$", r"\1 \2", a)     # '동일로1308-1' → '동일로 1308-1'
    return f"서울특별시 {gu} {a}"


if __name__ == "__main__":
    p = load_parking(use_lookup=False)                        # 지번 주소에서 동을 못 읽은 곳만 조회
    todo = p[p["동"].isna()].drop_duplicates("주소")
    rows = []
    for _, r in todo.iterrows():
        q = road_query(r["구"], r["주소"])
        if q in MANUAL:
            dong, jibun, why = MANUAL[q]
        else:
            hit = next((x for x in search(q) if norm(x.get("schKorRoadNm")) == norm(q)), None)
            dong, jibun, why = ((hit["emdNm"], re.sub(r"\s+", " ", hit["schKorJibunNm"]).strip(), "도로명 정확 일치")
                                if hit else (None, None, "조회 실패 — MANUAL에 추가 필요"))
            time.sleep(0.3)
        rows.append({"구": r["구"], "주소": str(r["주소"]).strip(), "조회주소": q,
                     "법정동": dong, "지번주소": jibun, "근거": why})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out["근거"].value_counts().to_string())
    print(f"{len(out)}곳 → {OUT}")
