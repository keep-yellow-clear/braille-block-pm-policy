"""원자료 읽기: 견인 현황(xlsx), 장애인 현황(csv), 주차구역 현황(csv)."""
import glob
import pandas as pd
from config import BRAILLE_KEYWORD


def load_tows(tow_glob: str) -> pd.DataFrame:
    """견인 현황 xlsx 여러 개를 하나로. 헤더 행은 '번호' 셀로 찾는다."""
    rows = []
    for f in sorted(glob.glob(tow_glob)):
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None)
            head = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "번호"][0]
            d = raw.iloc[head + 1:, :6].copy()
            d.columns = ["번호", "신고일", "구", "주소", "유형", "조치일"]
            rows.append(d)
    df = pd.concat(rows, ignore_index=True).dropna(subset=["구"])
    df["dt"] = pd.to_datetime(df["신고일"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df["ym"] = df["dt"].dt.strftime("%y%m")
    df["b"] = df["유형"].astype(str).str.contains(BRAILLE_KEYWORD)
    return df


def load_visual_disability(path: str) -> pd.Series:
    """구별 시각장애인 수(소계 연령). 서울 열린데이터광장 '장애인 현황(장애유형별/연령별)'."""
    d = pd.read_csv(path, encoding="utf-8-sig", header=None, skiprows=4)
    v = d[[1, 3, 10]].copy()
    v.columns = ["gu", "age", "vis"]
    v["vis"] = pd.to_numeric(v["vis"].astype(str).str.replace(",", "").replace("-", "0"),
                             errors="coerce").fillna(0)
    return v[v.age == "소계"].set_index("gu")["vis"].drop("소계").rename("v1")


def load_parking(path: str) -> pd.DataFrame:
    """구별 주차구역 수·거치대 설치 구역 수. 파일에 없는 구는 0으로 채워진다."""
    pk = pd.read_csv(path, encoding="cp949")
    out = pd.DataFrame({"park": pk.groupby("시군구명").size(),
                        "dock": pk[pk["거치대 유무"] == "Y"].groupby("시군구명").size()})
    return out.fillna(0)
