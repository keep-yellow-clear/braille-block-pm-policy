"""00 핫스팟 분석: 점자블록 위 전동킥보드는 어디에 몰려 있나 (2025).

입력 (data/raw, 받은 파일명 그대로)
  - 서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx    월별 시트
  - 서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx   월별 시트
  - 서울시 전동킥보드 주차구역 현황.csv               (cp949)
  + data/processed/주차구역_도로명주소_법정동.csv      도로명 주소 → 법정동 (lookup_parking_dong.py, 주소정보누리집 조회)

분석
  - 견인 사유(유형)에 '점자'가 들어간 건 = '점자블록 또는 교통약자 엘리베이터' 사유 견인
  - 지번(주소) 단위 반복 횟수, 법정동·자치구 단위 집계, 월별 추이
  - 주차구역 수: 지번 주소는 주소에서, 도로명 주소는 조회 결과에서 법정동을 읽어 329곳 모두 동별 집계
  - 주차구역 공백: 자치구별 '점자블록 견인 비중 vs 주차구역 비중', 주차구역 0곳 법정동에 몰린 견인

출력
  00_hotspot_parking/figures/fig1_자치구별_비교.png
  00_hotspot_parking/figures/fig2_법정동_Top20.png
  00_hotspot_parking/figures/fig3_반복지번_Top30_월별.png
  00_hotspot_parking/figures/fig4_지번_집중도.png
  00_hotspot_parking/figures/fig5_월별_추이.png
  00_hotspot_parking/figures/fig6_주차구역공백_자치구.png
  00_hotspot_parking/figures/fig7_주차구역공백_법정동.png
  00_hotspot_parking/results/hotspot_summary.xlsx        (그림에 쓴 표 전부)

실행: python 00_hotspot_parking/hotspot_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, ticker

# ---------------------------------------------------------------- 경로·설정
HERE = Path(__file__).resolve().parent                 # 00_hotspot_parking/
RAW_DIR = HERE.parent / "data" / "raw"
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
for d in (FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_TOW_2025 = [
    "서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx",
    "서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx",
]
RAW_PARKING = "서울시 전동킥보드 주차구역 현황.csv"
LOOKUP = HERE.parent / "data" / "processed" / "주차구역_도로명주소_법정동.csv"   # lookup_parking_dong.py 결과
TOW_COLS = ["번호", "신고일", "구정보", "주소", "유형", "조치일"]
YEAR = 2025
LOW_TOW_GU = 300            # 연 견인 300건 미만 구 = 견인 제도가 사실상 운영되지 않는 구

C_BR = "#2a78d6"            # 점자블록 사유 (01 분석과 같은 파랑)
C_PARK = "#eb6834"          # 주차구역
C_MUTED = "#b9b8b3"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 환경에 맞게 수정

try:
    fm.fontManager.addfont(FONT_PATH)
    FONT = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    FONT = "Malgun Gothic"  # Windows 기본 한글 폰트 (macOS는 'AppleGothic')
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.dpi": 200})


# ---------------------------------------------------------------- 원자료 읽기
def load_tow() -> pd.DataFrame:
    """2025년 견인 현황 원자료 2개(월별 시트)를 한 표로. 정제 규칙은 README의 원자료 절 참고."""
    frames = []
    for name in RAW_TOW_2025:
        for sheet, d in pd.read_excel(RAW_DIR / name, sheet_name=None, header=1).items():
            d = d.iloc[:, :6].copy()        # 0행 빈 제목행 → 1행이 헤더, 앞 6개 컬럼만
            d.columns = TOW_COLS
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["주소"])
    df["신고일"] = pd.to_datetime(df["신고일"], errors="coerce")
    df["주소"] = (df["주소"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                  .str.replace(r"^서울(특별)?시 ", "", regex=True))       # '마포구 서교동 406' 형태
    df["유형"] = df["유형"].astype(str).str.strip()
    df = df[(df["신고일"].dt.year == YEAR) & (df["유형"] != "오신고")].copy()
    tok = df["주소"].str.split(" ")
    df["구"] = tok.str[0]                   # 구정보 컬럼은 일부 잘려 있어 주소에서 추출
    df["동"] = tok.str[1]
    df["지번있음"] = tok.str.len() >= 3      # '금천구 가산동'처럼 동 이름만 적힌 건 구분
    df["월"] = df["신고일"].dt.month
    df["점자"] = df["유형"].str.contains("점자")
    return df


def load_parking(use_lookup: bool = True) -> pd.DataFrame:
    """주차구역 현황. 주소 첫 단어가 '…동'/'…가'이면 동 이름으로 읽는다 ('신문로 2가' → '신문로2가').
    도로명 주소는 lookup_parking_dong.py 가 만든 data/processed/주차구역_도로명주소_법정동.csv 로 법정동을 채운다."""
    p = None
    for enc in ("cp949", "utf-8-sig"):
        try:
            p = pd.read_csv(RAW_DIR / RAW_PARKING, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    p["구"] = p["시군구명"].astype(str).str.strip()
    addr = (p["주소"].astype(str).str.replace(r"^서울(특별)?시\s*", "", regex=True)
            .apply(lambda s: s.strip()))
    addr = addr.str.replace(r"^\S+구\s+", "", regex=True)               # 주소 앞에 구 이름이 붙은 경우
    addr = addr.str.replace(r"^(\S+)\s+(\d+가)", r"\1\2", regex=True)    # '신문로 2가' → '신문로2가'
    addr = addr.str.replace(r"^(\S+?(?:동|가))(\d)", r"\1 \2", regex=True)  # '창동723-11' → '창동 723-11'
    first = addr.str.split(" ").str[0]
    p["동"] = first.where(first.str.match(r".+(동|가)$"), np.nan)
    p["동"] = p["동"].str.replace(r"\d+동$", "동", regex=True)          # 행정동 표기(봉천1동 등) → 법정동
    p["동출처"] = np.where(p["동"].notna(), "지번 주소", None)
    if use_lookup and LOOKUP.exists():
        lk = pd.read_csv(LOOKUP, encoding="utf-8-sig").dropna(subset=["법정동"])
        key = lk.set_index(["구", "주소"])["법정동"]
        fill = pd.Series([key.get((g, str(a).strip())) for g, a in zip(p["구"], p["주소"])], index=p.index)
        need = p["동"].isna() & fill.notna()
        p.loc[need, "동"] = fill[need]
        p.loc[need, "동출처"] = "도로명 주소 조회"
    return p


# ---------------------------------------------------------------- 집계
def build_tables(tow: pd.DataFrame, park: pd.DataFrame) -> dict:
    br = tow[tow["점자"]]
    park_gu = park.groupby("구").size()
    park_dong = park.dropna(subset=["동"]).groupby(["구", "동"]).size()

    gu = (tow.groupby("구").agg(전체견인=("주소", "size"), 점자블록견인=("점자", "sum")))
    gu["점자블록비중"] = gu["점자블록견인"] / gu["전체견인"]
    gu["주차구역"] = park_gu.reindex(gu.index).fillna(0).astype(int)
    for g in park_gu.index.difference(gu.index):            # 견인 0건 구도 표에 남김
        gu.loc[g] = [0, 0, np.nan, park_gu[g]]
    gu["견인운영"] = gu["전체견인"] >= LOW_TOW_GU
    gu = gu.sort_values(["점자블록견인", "전체견인"], ascending=False)

    dong = br.groupby(["구", "동"]).size().rename("점자블록견인").to_frame()
    dong["전체견인"] = tow.groupby(["구", "동"]).size().reindex(dong.index)
    dong["주차구역"] = park_dong.reindex(dong.index).fillna(0).astype(int)
    dong = dong.sort_values("점자블록견인", ascending=False)

    lot = br[br["지번있음"]]
    addr = (lot.groupby("주소").agg(구=("구", "first"), 동=("동", "first"), 점자블록견인=("주소", "size"),
                                  견인월수=("월", "nunique"), 견인일수=("신고일", "nunique"),
                                  첫견인=("신고일", "min"), 마지막견인=("신고일", "max")))
    addr["전체견인"] = tow.groupby("주소").size().reindex(addr.index)
    addr = addr.sort_values(["점자블록견인", "견인일수", "전체견인"], ascending=False)
    months = (lot.groupby(["주소", "월"]).size().unstack(fill_value=0)
              .reindex(columns=range(1, 13), fill_value=0))

    monthly = tow.groupby("월").agg(전체견인=("주소", "size"), 점자블록견인=("점자", "sum"))
    monthly["점자블록비중"] = monthly["점자블록견인"] / monthly["전체견인"]

    n = addr["점자블록견인"]
    kpi = pd.Series({
        "2025 전체 견인(오신고 제외)": len(tow),
        "점자블록 사유 견인": len(br),
        "점자블록 견인 지번 수(지번 있는 주소)": len(addr),
        "1회만 견인된 지번": int((n == 1).sum()),
        "2회 이상 반복 지번": int((n >= 2).sum()),
        "반복 지번이 차지한 견인 비중": n[n >= 2].sum() / n.sum(),
        "상습 지번(3회 이상·3개월 이상)": int(((n >= 3) & (addr["견인월수"] >= 3)).sum()),
        "상위 10% 지번이 차지한 견인 비중": n.iloc[:int(round(len(n) * .1))].sum() / n.sum(),
        "상위 20개 동이 차지한 견인 비중": dong["점자블록견인"].iloc[:20].sum() / len(br),
        "주차구역 수(전체)": len(park),
        "주차구역 중 법정동 배정(지번 주소)": int((park["동출처"] == "지번 주소").sum()),
        "주차구역 중 법정동 배정(도로명 주소 조회)": int((park["동출처"] == "도로명 주소 조회").sum()),
    }, name="값")
    # 주차구역 공백: 견인이 운영되는 구만 (운영 안 되는 구는 0건이 '위반 없음'이 아님)
    act = gu[gu["견인운영"]].copy()
    act["점자블록견인_비중"] = act["점자블록견인"] / act["점자블록견인"].sum()
    act["주차구역_비중"] = act["주차구역"] / act["주차구역"].sum()
    act["비중차(견인-주차)"] = act["점자블록견인_비중"] - act["주차구역_비중"]
    act["주차구역1곳당_점자블록견인"] = act["점자블록견인"] / act["주차구역"].replace(0, np.nan)
    gap_gu = act.sort_values("비중차(견인-주차)", ascending=False)

    bins = pd.cut(dong["주차구역"], [-1, 0, 2, np.inf], labels=["0곳", "1–2곳", "3곳 이상"])
    gap_bucket = dong.groupby(bins, observed=False).agg(동수=("점자블록견인", "size"), 점자블록견인=("점자블록견인", "sum"))
    gap_bucket["견인비중"] = gap_bucket["점자블록견인"] / gap_bucket["점자블록견인"].sum()
    gap_dong = dong[dong["주차구역"] == 0].sort_values("점자블록견인", ascending=False)

    kpi["주차구역 0곳 동에서 난 점자블록 견인 비중"] = gap_bucket.loc["0곳", "견인비중"]
    return dict(kpi=kpi, gu=gu, dong=dong, addr=addr, months=months, monthly=monthly,
                gap_gu=gap_gu, gap_bucket=gap_bucket, gap_dong=gap_dong)


# ---------------------------------------------------------------- 그림
def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=13, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color="#6b6a66", va="bottom")


def fig_gu(gu: pd.DataFrame):
    """자치구별: 점자블록 견인 건수 / 점자블록 비중 / 주차구역 수 (같은 y축, 세 패널)."""
    g = gu.iloc[::-1]
    y = np.arange(len(g))
    on = g["견인운영"].values
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True, gridspec_kw={"wspace": .12})
    panels = [("점자블록견인", "점자블록 사유 견인(건)", C_BR, "{:,.0f}"),
              ("점자블록비중", "그 구 전체 견인 중 점자블록 비중", C_BR, "{:.1%}"),
              ("주차구역", "PM 주차구역(곳)", C_PARK, "{:,.0f}")]
    for ax, (col, lab, c, f) in zip(axes, panels):
        v = g[col].fillna(0).values
        ax.barh(y, v, color=np.where(on, c, C_MUTED), height=.7)
        for yi, vi, o in zip(y, v, on):
            ax.text(vi, yi, " " + f.format(vi), va="center", fontsize=8.5, color="#3a3936" if o else "#8a8984")
        ax.set_xlabel(lab); ax.grid(axis="x", color="#ecebe7", lw=.8); ax.set_axisbelow(True)
        ax.set_xlim(0, v.max() * 1.22 if v.max() > 0 else 1)
        if col == "점자블록비중":
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    axes[0].set_yticks(y); axes[0].set_yticklabels(g.index)
    fig.suptitle("자치구별 점자블록 사유 견인과 주차구역 (2025)", x=.06, ha="left", fontsize=14)
    fig.text(.06, .925, f"회색 = 2025년 견인이 {LOW_TOW_GU}건 미만인 구. 견인 제도가 사실상 운영되지 않아 0건이 '위반 없음'을 뜻하지 않음",
             fontsize=9.5, color="#6b6a66")
    fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.08)
    fig.savefig(FIG_DIR / "fig1_자치구별_비교.png"); plt.close(fig)


def fig_dong(dong: pd.DataFrame, top=20):
    d = dong.iloc[:top].iloc[::-1]
    labels = [f"{g} {n}" for g, n in d.index]
    y = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 7.2), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1.2], "wspace": .08})
    a1.barh(y, d["점자블록견인"], color=C_BR, height=.7)
    for yi, v in zip(y, d["점자블록견인"]):
        a1.text(v, yi, f" {v}", va="center", fontsize=9)
    a2.barh(y, d["주차구역"], color=C_PARK, height=.7)
    for yi, v in zip(y, d["주차구역"]):
        a2.text(v, yi, f" {v}", va="center", fontsize=9, color="#3a3936" if v else "#b23b1a")
    a1.set_yticks(y); a1.set_yticklabels(labels)
    a1.set_xlabel("점자블록 사유 견인(건)"); a2.set_xlabel("동 안 PM 주차구역(곳)")
    a2.set_xlim(0, max(d["주차구역"].max(), 1) * 1.6)
    a2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
    for a in (a1, a2):
        a.grid(axis="x", color="#ecebe7", lw=.8); a.set_axisbelow(True)
    share = dong["점자블록견인"].iloc[:top].sum() / dong["점자블록견인"].sum()
    fig.suptitle(f"점자블록 사유 견인이 많은 법정동 Top {top} — 전체의 {share:.0%}", x=.06, ha="left", fontsize=14)
    zero = int((dong["주차구역"].iloc[:top] == 0).sum())
    fig.text(.06, .925, f"오른쪽: 같은 동 안의 PM 주차구역 수 (329곳 모두 법정동 배정). Top {top} 중 {zero}개 동은 주차구역 0곳",
             fontsize=9.5, color="#6b6a66")
    fig.subplots_adjust(left=.17, right=.97, top=.9, bottom=.09)
    fig.savefig(FIG_DIR / "fig2_법정동_Top20.png"); plt.close(fig)


def fig_top_addr(addr: pd.DataFrame, months: pd.DataFrame, top=30):
    a = addr.iloc[:top]
    m = months.loc[a.index].values
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(11, 10), sharey=True,
                                  gridspec_kw={"width_ratios": [4, 1.3], "wspace": .05})
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(4, m.max()), aspect="auto")
    for (i, j), v in np.ndenumerate(m):
        if v:
            ax.text(j, i, v, ha="center", va="center", fontsize=8.5,
                    color="white" if v >= m.max() * .6 else "#0b2a4f")
    ax.set_xticks(range(12)); ax.set_xticklabels([f"{i}월" for i in range(1, 13)])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(a))); ax.set_yticklabels([f"{i+1}. {s}" for i, s in enumerate(a.index)], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 12), minor=True); ax.set_yticks(np.arange(-.5, len(a)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=2); ax.tick_params(which="minor", length=0)
    y = np.arange(len(a))
    axn.barh(y, a["점자블록견인"], color=C_BR, height=.65)
    for yi, v, mo, al in zip(y, a["점자블록견인"], a["견인월수"], a["전체견인"]):
        axn.text(v, yi, f" {v}건 · {mo}개월 · 전체 {al}", va="center", fontsize=8)
    axn.set_xlim(0, a["점자블록견인"].max() * 2.6); axn.set_xticks([])
    axn.spines["bottom"].set_visible(False); axn.spines["left"].set_visible(False)
    axn.set_title("점자블록 견인 · 견인 월수 · 전체 사유 견인", fontsize=9, color="#52514e", loc="left")
    fig.suptitle(f"같은 지번에서 점자블록 사유 견인이 되풀이된 곳 Top {top} (2025, 월별 건수)",
                 x=.03, ha="left", fontsize=14)
    fig.subplots_adjust(left=.22, right=.98, top=.9, bottom=.02)
    fig.savefig(FIG_DIR / "fig3_반복지번_Top30_월별.png"); plt.close(fig)


def fig_concentration(addr: pd.DataFrame, kpi: pd.Series):
    n = addr["점자블록견인"].sort_values(ascending=False).values
    dist = pd.Series(np.clip(n, None, 10)).value_counts().sort_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": .28})
    a1.bar(dist.index, dist.values, color=[C_MUTED] + [C_BR] * (len(dist) - 1), width=.75)
    for x, v in dist.items():
        a1.text(x, v, f"{v}", ha="center", va="bottom", fontsize=9)
    a1.set_xticks(dist.index); a1.set_xticklabels([str(i) if i < 10 else "10+" for i in dist.index])
    a1.set_xlabel("지번 하나에서 점자블록 사유로 견인된 횟수"); a1.set_ylabel("지번 수")
    a1.grid(axis="y", color="#ecebe7", lw=.8); a1.set_axisbelow(True)
    title(a1, "지번별 반복 횟수", f"2회 이상 반복된 지번 {kpi['2회 이상 반복 지번']:.0f}곳이 견인의 {kpi['반복 지번이 차지한 견인 비중']:.0%}")

    x = np.arange(1, len(n) + 1) / len(n)
    yv = np.cumsum(n) / n.sum()
    a2.plot(np.r_[0, x], np.r_[0, yv], color=C_BR, lw=2)
    a2.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--")
    for p in (.1, .2):
        v = yv[int(round(len(n) * p)) - 1]
        a2.plot(p, v, "o", ms=8, color=C_BR, mec="#fcfcfb", mew=1.5)
        a2.annotate(f"상위 {p:.0%} 지번 → 견인 {v:.0%}", (p, v), xytext=(10, -4), textcoords="offset points", fontsize=9.5)
    a2.xaxis.set_major_formatter(ticker.PercentFormatter(1)); a2.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.02)
    a2.set_xlabel("견인 많은 지번부터 누적한 지번 비율"); a2.set_ylabel("누적 점자블록 견인 비율")
    a2.grid(color="#ecebe7", lw=.8)
    title(a2, "누적 집중도", "점선 = 모든 지번이 똑같이 견인될 때")
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.13)
    fig.savefig(FIG_DIR / "fig4_지번_집중도.png"); plt.close(fig)


def fig_monthly(monthly: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"hspace": .35})
    x = monthly.index
    a1.bar(x, monthly["점자블록견인"], color=C_BR, width=.7)
    for xi, v in zip(x, monthly["점자블록견인"]):
        a1.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("건"); a1.grid(axis="y", color="#ecebe7", lw=.8); a1.set_axisbelow(True)
    title(a1, "월별 점자블록 사유 견인 (2025)")
    a2.plot(x, monthly["점자블록비중"], color=C_BR, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=1.5)
    a2.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=1))
    a2.set_ylim(0, monthly["점자블록비중"].max() * 1.3); a2.grid(axis="y", color="#ecebe7", lw=.8)
    title(a2, "전체 견인 중 점자블록 비중", "단속량 변화를 걸러 낸 값")
    a2.set_xticks(x); a2.set_xticklabels([f"{i}월" for i in x])
    fig.subplots_adjust(left=.09, right=.98, top=.93, bottom=.07)
    fig.savefig(FIG_DIR / "fig5_월별_추이.png"); plt.close(fig)


def fig_gap_gu(gap: pd.DataFrame):
    """자치구별 점자블록 견인 비중 vs 주차구역 비중 (견인 운영 구, 비중 차이 큰 순)."""
    g = gap.iloc[::-1]
    y = np.arange(len(g))
    h = .38
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.barh(y + h / 2, g["점자블록견인_비중"], height=h, color=C_BR, label="서울 점자블록 사유 견인 중 그 구의 비중")
    ax.barh(y - h / 2, g["주차구역_비중"], height=h, color=C_PARK, label="서울 PM 주차구역 중 그 구의 비중")
    xmax = max(g["점자블록견인_비중"].max(), g["주차구역_비중"].max())
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.text(r["점자블록견인_비중"], yi + h / 2, f" {r['점자블록견인_비중']:.1%}", va="center", fontsize=8.5)
        ax.text(r["주차구역_비중"], yi - h / 2, f" {r['주차구역_비중']:.1%}", va="center", fontsize=8.5)
        per = r["주차구역1곳당_점자블록견인"]
        txt = "주차구역 0곳" if pd.isna(per) else (f"{per:.0f}건" if per >= 10 else f"{per:.1f}건" if per >= .1 else f"{per:.2f}건")
        ax.text(xmax * 1.28, yi, txt, va="center", ha="right", fontsize=9,
                color="#b23b1a" if (pd.isna(per) or per >= 20) else "#52514e")
    ax.text(xmax * 1.28, len(g) - .3, "주차구역 1곳당\n점자블록 견인", ha="right", va="bottom", fontsize=8.5, color="#6b6a66")
    ax.set_yticks(y); ax.set_yticklabels(g.index)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, xmax * 1.3); ax.grid(axis="x", color="#ecebe7", lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5)
    top2 = gap.iloc[:2]
    fig.suptitle(f"주차구역 공백 — {'·'.join(top2.index)}는 점자블록 견인의 {top2['점자블록견인_비중'].sum():.0%}, "
                 f"주차구역은 {top2['주차구역_비중'].sum():.0%}", x=.06, ha="left", fontsize=14)
    fig.text(.06, .925, f"2025년 견인 {LOW_TOW_GU}건 이상 {len(g)}개 구 기준. 위로 갈수록 견인 비중에 비해 주차구역이 모자란 구. "
             "주차구역 수는 공개 데이터 전체(구 단위)", fontsize=9.5, color="#6b6a66")
    fig.subplots_adjust(left=.1, right=.97, top=.86, bottom=.06)
    fig.savefig(FIG_DIR / "fig6_주차구역공백_자치구.png"); plt.close(fig)


def fig_gap_dong(gap_dong: pd.DataFrame, bucket: pd.DataFrame, top=15):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.6), gridspec_kw={"width_ratios": [1, 1.6], "wspace": .35})
    # (a) 동의 주차구역 수별 점자블록 견인 비중
    b = bucket["견인비중"]
    x = np.arange(len(b))
    a1.bar(x, b.values, color=[C_PARK, "#f3b08f", C_MUTED], width=.65)
    for xi, (k, r) in zip(x, bucket.iterrows()):
        a1.text(xi, r["견인비중"], f"{r['견인비중']:.0%}\n({int(r['점자블록견인']):,}건·{int(r['동수'])}개 동)",
                ha="center", va="bottom", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels([f"주차구역\n{k}" for k in b.index])
    a1.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0)); a1.set_ylim(0, b.max() * 1.3)
    a1.grid(axis="y", color="#ecebe7", lw=.8); a1.set_axisbelow(True)
    title(a1, "동 안 주차구역 수별 점자블록 견인", "점자블록 견인이 1건 이상 있었던 법정동 기준")
    # (b) 주차구역 0곳인데 견인 많은 동
    d = gap_dong.iloc[:top].iloc[::-1]
    y = np.arange(len(d))
    a2.barh(y, d["점자블록견인"], color=C_PARK, height=.7)
    for yi, (idx, r) in zip(y, d.iterrows()):
        a2.text(r["점자블록견인"], yi, f" {r['점자블록견인']}건 (전체 견인 {r['전체견인']:,})", va="center", fontsize=8.5)
    a2.set_yticks(y); a2.set_yticklabels([f"{g} {n}" for g, n in d.index])
    a2.set_xlim(0, d["점자블록견인"].max() * 1.6); a2.set_xlabel("점자블록 사유 견인(건)")
    a2.grid(axis="x", color="#ecebe7", lw=.8); a2.set_axisbelow(True)
    title(a2, f"주차구역이 한 곳도 없는 동 Top {top}", "신설 주차구역 우선 검토 대상")
    fig.suptitle(f"점자블록 견인의 {bucket.loc['0곳', '견인비중']:.0%}가 PM 주차구역이 한 곳도 없는 동에서 일어났다 (2025)",
                 x=.06, ha="left", fontsize=14)
    fig.text(.06, .925, "주차구역 329곳 모두 법정동 배정 (지번 주소는 주소에서, 도로명 주소는 주소정보누리집 조회로)",
             fontsize=9.5, color="#6b6a66")
    fig.subplots_adjust(left=.06, right=.97, top=.78, bottom=.1)
    fig.savefig(FIG_DIR / "fig7_주차구역공백_법정동.png"); plt.close(fig)


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    tow, park = load_tow(), load_parking()
    T = build_tables(tow, park)

    fig_gu(T["gu"])
    fig_dong(T["dong"])
    fig_top_addr(T["addr"], T["months"])
    fig_concentration(T["addr"], T["kpi"])
    fig_monthly(T["monthly"])
    fig_gap_gu(T["gap_gu"])
    fig_gap_dong(T["gap_dong"], T["gap_bucket"])

    with pd.ExcelWriter(RES_DIR / "hotspot_summary.xlsx") as w:
        T["kpi"].to_frame().to_excel(w, sheet_name="핵심수치")
        T["gu"].to_excel(w, sheet_name="자치구별")
        T["dong"].reset_index().to_excel(w, sheet_name="법정동별", index=False)
        top = T["addr"].iloc[:30].join(T["months"].add_suffix("월"))
        top.assign(첫견인=top["첫견인"].dt.date, 마지막견인=top["마지막견인"].dt.date) \
           .to_excel(w, sheet_name="반복지번_Top30")
        T["monthly"].to_excel(w, sheet_name="월별")
        T["gap_gu"].to_excel(w, sheet_name="주차구역공백_자치구")
        T["gap_bucket"].to_excel(w, sheet_name="주차구역공백_동구간")
        T["gap_dong"].reset_index().to_excel(w, sheet_name="주차구역0곳_동", index=False)

    k = T["kpi"]
    for i, v in k.items():
        print(f"{i}: {v:.1%}" if v < 1 else f"{i}: {v:,.0f}")
    print("그림 저장:", FIG_DIR)
