"""원자료 읽기: 견인 현황, 시각장애인 수, 주차구역(+법정동 보완표)."""
import glob
import re

import pandas as pd

from config import YEAR, BRAILLE_KEYWORD, EXCLUDE_TYPES, INPUT_PATTERNS

TOW_COLS = {"신고일": "date", "구정보": "gu", "주소": "addr", "유형": "type"}


def find_inputs(key, override=None):
    """입력 파일 찾기. override(명령줄 인자)가 있으면 그것을, 없으면 config.INPUT_PATTERNS 순서대로."""
    patterns = [override] if override else [str(p) for p in INPUT_PATTERNS[key]]
    for pat in patterns:
        files = sorted(glob.glob(str(pat)))
        if files:
            return files
    raise FileNotFoundError(f"[{key}] 입력 파일을 찾지 못했습니다. 찾은 위치: {patterns}\n"
                            "저장소 data/raw, data/processed 에 파일이 있는지, 또는 명령줄 인자로 경로를 지정했는지 확인하세요.")


def _read_tow_sheet(df):
    """시트마다 제목 행이 있을 수 있어, '구정보'가 있는 행을 머리글로 찾음. 없으면 None."""
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if all(c in row for c in TOW_COLS):
            out = df.iloc[i + 1:].copy()
            out.columns = [str(c).strip() for c in df.iloc[i]]
            return out[list(TOW_COLS)].rename(columns=TOW_COLS)
    return None


def load_tows(files):
    """견인 현황 xlsx(여러 파일·여러 시트)를 하나로 합침. 2025년 행만 남김."""
    frames = [_read_tow_sheet(df) for f in files
              for df in pd.read_excel(f, sheet_name=None, header=None).values()]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError(f"견인 현황 시트(신고일·구정보·주소·유형 열)를 찾지 못했습니다: {files}")
    d = pd.concat(frames, ignore_index=True).dropna(subset=["gu"])
    for c in ["gu", "addr", "type"]:
        d[c] = d[c].astype(str).str.strip()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].dt.strftime("%Y") == YEAR]
    if EXCLUDE_TYPES:
        d = d[~d["type"].isin(EXCLUDE_TYPES)]
    d["month"] = d["date"].dt.strftime("%Y-%m")
    d["braille"] = d["type"].str.contains(BRAILLE_KEYWORD)
    d["dong"] = d["addr"].str.extract(r"서울특별시\s+\S+구\s+(\S+)")[0]  # 법정동 (예: 신림동, 을지로4가)
    print(f"[견인] {len(files)}개 파일, {YEAR}년 {len(d):,}건 (점자블록 {d.braille.sum():,}건)")
    if d.empty:
        raise ValueError(f"{YEAR}년 견인 기록이 없습니다. 입력 파일을 확인하세요: {files}")
    return d.reset_index(drop=True)


def load_visual_disability(path):
    """서울시 장애인 현황(장애유형별·연령별) CSV에서 자치구별 시각장애인 합계."""
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig", dtype=str)
    col = next(c for c in raw.columns
               if raw.iloc[2, c] == "시각" and raw.iloc[3, c] == "합계")
    body = raw.iloc[4:]
    rows = body[(body[2] == "합계") & (body[3] == "소계") & (body[1] != "소계")]
    v = pd.to_numeric(rows.set_index(1)[col].str.replace(",", ""), errors="coerce")
    v.index.name = "gu"
    return v.rename("v1")


def _dong_from_lot_address(gu, addr):
    """지번 주소에서 법정동 추출 (예: '서초동711-4' -> 서초동, '신문로 2가 58' -> 신문로2가).
    도로명 주소는 None."""
    a = re.sub(r"^(서울특별시\s+)?" + gu + r"\s*", "", addr)
    a = re.sub(r"\s+(\d+가)\b", r"\1", a)
    m = re.match(r"^(\D+?(?:동|\d+가))(?=[\s\d]|$)", a)
    return m.group(1) if m else None


def load_parking(path, dong_map_path=None):
    """전동킥보드 주차구역 현황. 도로명 주소는 법정동 보완표로 법정동을 채움."""
    p = pd.read_csv(path, encoding="cp949")
    p.columns = ["no", "gu", "addr", "detail", "rack"]
    p["gu"] = p["gu"].str.strip()
    p["addr"] = p["addr"].astype(str).str.strip()
    p["dong_lot"] = [_dong_from_lot_address(g, a) for g, a in zip(p.gu, p.addr)]
    p["dong"] = p["dong_lot"]
    if dong_map_path:
        m = pd.read_csv(dong_map_path, encoding="utf-8-sig")
        lookup = {(g, a.strip()): d for g, a, d in zip(m["구"], m["주소"], m["법정동"])}
        p["dong"] = [lookup.get((g, a)) or d for g, a, d in zip(p.gu, p.addr, p.dong_lot)]
    print(f"[주차구역] {len(p)}곳, 법정동 빈칸 {p.dong_lot.isna().sum()}곳 -> 보완 후 {p.dong.isna().sum()}곳")
    return p
