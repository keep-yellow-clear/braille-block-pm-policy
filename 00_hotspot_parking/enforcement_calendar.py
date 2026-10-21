"""00 단속 캘린더: 점자블록 사유 견인은 언제 멈추고, 언제 몰리나 (2025).

입력 (data/raw, 받은 파일명 그대로)
  - 서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx, (25.7~12월).xlsx   월별 시트
  - 기온 현황/2025_OBS_ASOS_TIM_20261012182520.csv                    기상청 ASOS 서울(108) 시간자료 (cp949)
  - 공휴일: Python holidays 패키지 (대한민국 공휴일·대체공휴일·임시공휴일)

분석
  - 날짜별 견인 건수(전체 / 점자블록 사유)를 요일·공휴일·기상과 결합
  - 강수는 낮 시간(07–20시) 합계. 12/31은 기상자료가 없어 모형에서 제외
  - 평일 대상 음이항 회귀: 점자블록 견인 ~ 월 + 요일 + 강수구간 + 폭염 + 적설 + 기온편차 + 전날 큰비
    → 발생률비(IRR)와 95% 신뢰구간
  - 월×요일 예측값(비 없는 평일 기준)으로 집중 단속 캘린더 작성

출력
  00_hotspot_parking/figures/cal1_요일별_평균.png
  00_hotspot_parking/figures/cal2_구별_운영률.png
  00_hotspot_parking/figures/cal3_일별_흐름_강수.png
  00_hotspot_parking/figures/cal4_기상요일_효과.png
  00_hotspot_parking/figures/cal5_집중단속_캘린더.png
  00_hotspot_parking/results/calendar_summary.xlsx

실행: python 00_hotspot_parking/enforcement_calendar.py   (statsmodels, holidays 필요)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates  # noqa: F401
from matplotlib import font_manager as fm, ticker
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------- 경로·설정
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent / "data" / "raw"
FIG_DIR, RES_DIR = HERE / "figures", HERE / "results"
for d in (FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_TOW_2025 = [
    "서울특별시_전동킥보드_견인_현황(25.1~6월).xlsx",
    "서울특별시_전동킥보드_견인_현황(25.7~12월).xlsx",
]
RAW_WEATHER = Path("기온 현황") / "2025_OBS_ASOS_TIM_20261012182520.csv"
TOW_COLS = ["번호", "신고일", "구정보", "주소", "유형", "조치일"]
YEAR = 2025
LOW_TOW_GU = 300
WD = ["월", "화", "수", "목", "금", "토", "일"]

C_BR, C_ALL, C_RAIN, C_MUTED = "#2a78d6", "#8a8984", "#1f9e89", "#b9b8b3"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 환경에 맞게 수정
try:
    fm.fontManager.addfont(FONT_PATH)
    FONT = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    FONT = "Malgun Gothic"
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200})


# ---------------------------------------------------------------- 원자료 읽기
def load_tow() -> pd.DataFrame:
    frames = []
    for name in RAW_TOW_2025:
        for _, d in pd.read_excel(RAW_DIR / name, sheet_name=None, header=1).items():
            d = d.iloc[:, :6].copy()
            d.columns = TOW_COLS
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["주소"])
    df["신고일"] = pd.to_datetime(df["신고일"], errors="coerce")
    df["주소"] = (df["주소"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                  .str.replace(r"^서울(특별)?시 ", "", regex=True))
    df["유형"] = df["유형"].astype(str).str.strip()
    df = df[(df["신고일"].dt.year == YEAR) & (df["유형"] != "오신고")].copy()
    df["구"] = df["주소"].str.split(" ").str[0]
    df["일"] = df["신고일"].dt.normalize()
    df["점자"] = df["유형"].str.contains("점자")
    return df


def load_weather() -> pd.DataFrame:
    """ASOS 시간자료 → 일별: 낮(07–20시) 강수 합, 하루 강수 합, 평균·최고기온, 적설 여부."""
    w = pd.read_csv(RAW_DIR / RAW_WEATHER, encoding="cp949")
    w.columns = ["지점", "지점명", "일시", "기온", "강수", "풍속", "적설"]
    t = pd.to_datetime(w["일시"])
    w["일"], w["시"] = t.dt.normalize(), t.dt.hour
    w[["강수", "적설"]] = w[["강수", "적설"]].fillna(0)
    day = w[w["시"] >= 1]                                   # 00시 값은 전날 24시 관측이라 그날 합계에서 제외
    out = day.groupby("일").agg(rain=("강수", "sum"), tavg=("기온", "mean"), tmax=("기온", "max"),
                               snow=("적설", "max"))
    out["rd"] = day[day["시"].between(7, 20)].groupby("일")["강수"].sum()
    return out.fillna({"rd": 0})


def day_table(tow: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    kr = holidays.KR(years=YEAR)
    days = pd.DataFrame(index=pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", name="일"))
    days["all"] = tow.groupby("일").size().reindex(days.index, fill_value=0)
    days["br"] = tow[tow["점자"]].groupby("일").size().reindex(days.index, fill_value=0)
    days["dow"] = days.index.dayofweek
    days["hol"] = [kr.get(d, "") for d in days.index.date]
    days["dt"] = np.select([days["hol"] != "", days["dow"] == 6, days["dow"] == 5],
                           ["공휴일", "일요일", "토요일"], "평일")
    days = days.join(wx)
    days["prev_rain"] = days["rain"].shift(1)
    return days


# ---------------------------------------------------------------- 모형
def fit_models(days: pd.DataFrame):
    d = days[(days["dt"] == "평일") & days["rd"].notna()].copy()     # 12/31 기상 결측 제외
    d["month"] = d.index.month
    d["dowc"] = pd.Categorical(d["dow"].map(dict(enumerate(["mon", "tue", "wed", "thu", "fri"]))),
                               categories=["mon", "tue", "wed", "thu", "fri"])
    d["rcat"] = pd.cut(d["rd"], [-1, 0.05, 5, 20, np.inf], labels=["r0", "r1", "r2", "r3"])
    d["heat"] = (d["tmax"] >= 33).astype(int)
    d["snow"] = (d["snow"] > 0).astype(int)
    d["tanom"] = d["tavg"] - d.groupby("month")["tavg"].transform("mean")
    d["prev_heavy"] = ((d["prev_rain"] >= 10) & (d["rd"] < 0.05)).astype(int)
    f = "~ C(month) + C(dowc) + C(rcat) + heat + snow + tanom + prev_heavy"
    nb_br = smf.negativebinomial("br " + f, d).fit(disp=0, maxiter=200)
    nb_all = smf.negativebinomial("all " + f, d).fit(disp=0, maxiter=200)
    return d, nb_br, nb_all


LABELS = {"C(dowc)[T.tue]": "화요일", "C(dowc)[T.wed]": "수요일", "C(dowc)[T.thu]": "목요일",
          "C(dowc)[T.fri]": "금요일", "C(rcat)[T.r1]": "낮 비 0.1–5mm", "C(rcat)[T.r2]": "낮 비 5–20mm",
          "C(rcat)[T.r3]": "낮 비 20mm 이상", "heat": "폭염(최고 33℃↑)", "snow": "적설",
          "tanom": "기온 편차(+1℃)", "prev_heavy": "전날 큰비 후 갠 날"}


def irr_table(m, name):
    ci = m.conf_int()
    t = pd.DataFrame({"IRR": np.exp(m.params), "하한": np.exp(ci[0]), "상한": np.exp(ci[1]), "p": m.pvalues})
    t = t.loc[list(LABELS)].rename(index=LABELS)
    t.insert(0, "모형", name)
    return t


def calendar_grid(d, m):
    """비 없는 평일 기준 월×요일 예측 일평균."""
    rows = [{"month": mo, "dowc": wd, "rcat": "r0", "heat": 0, "snow": 0, "tanom": 0.0, "prev_heavy": 0}
            for mo in range(1, 13) for wd in ["mon", "tue", "wed", "thu", "fri"]]
    g = pd.DataFrame(rows)
    g["dowc"] = pd.Categorical(g["dowc"], categories=["mon", "tue", "wed", "thu", "fri"])
    g["rcat"] = pd.Categorical(g["rcat"], categories=["r0", "r1", "r2", "r3"])
    g["pred"] = np.asarray(m.predict(g))
    emp = d.groupby(["month", "dow"])["br"].agg(["mean", "size"])
    g["실제평균"] = [emp.loc[(mo, i), "mean"] for mo, i in zip(g["month"], g["dowc"].cat.codes)]
    q = g["pred"].quantile([.5, .75, .9]).values
    g["등급"] = np.digitize(g["pred"], q)            # 0 완화, 1 보통, 2 집중(상위 25%), 3 최우선(상위 10%)
    return g, q


# ---------------------------------------------------------------- 그림
def top_of(fig, extra=0.0):
    """그림 제목·부제 아래 여백을 인치 단위로 맞춘 subplots_adjust top 값."""
    return 1 - (1.0 + extra) / fig.get_figheight()


def title(ax_or_fig, text, sub=None, fig=False):
    if fig:
        h = ax_or_fig.get_figheight()
        ax_or_fig.suptitle(text, x=.06, y=1 - .2 / h, ha="left", va="top", fontsize=14)
        if sub:
            ax_or_fig.text(.06, 1 - .58 / h, sub, fontsize=9.5, color="#6b6a66", va="top")
    else:
        ax_or_fig.set_title(text, loc="left", fontsize=12.5, pad=20 if sub else 8)
        if sub:
            ax_or_fig.text(0, 1.02, sub, transform=ax_or_fig.transAxes, fontsize=9.5, color="#6b6a66")


def fig_dow(days):
    g = days.copy()
    g["k"] = np.where(g["hol"] != "", "공휴일", g["dow"].map(dict(enumerate(WD))))
    t = g.groupby("k")[["br", "all"]].mean().reindex(WD + ["공휴일"])
    n = g.groupby("k").size().reindex(t.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": .22})
    for ax, col, c, lab in ((axes[0], "br", C_BR, "점자블록 사유 견인"), (axes[1], "all", C_ALL, "전체 견인 (모든 사유)")):
        colors = [c if i < 5 else C_MUTED for i in range(len(t))]
        ax.bar(range(len(t)), t[col], color=colors, width=.7)
        for i, v in enumerate(t[col]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(t))); ax.set_xticklabels([f"{k}\n({n[k]}일)" for k in t.index])
        ax.grid(axis="y", color="#ecebe7", lw=.8); ax.set_axisbelow(True); ax.set_ylabel("하루 평균 건수")
        title(ax, lab, "공휴일은 요일과 따로 묶음")
    title(fig, "요일별 하루 평균 견인 (2025) — 주말·공휴일에는 보도 단속이 멈춘다", fig=True)
    fig.subplots_adjust(left=.06, right=.98, top=top_of(fig, .25), bottom=.14)
    fig.savefig(FIG_DIR / "cal1_요일별_평균.png"); plt.close(fig)


def fig_gu_op(tow, days):
    dt = days["dt"].replace({"일요일": "일·공휴일", "공휴일": "일·공휴일", "토요일": "토"})
    n_dt = dt.value_counts()
    t = tow.copy(); t["dt"] = t["일"].map(dt)
    gu_all = t.groupby("구").size()
    gus = gu_all[gu_all >= LOW_TOW_GU].index
    br = t[t["점자"]].groupby("구").size().reindex(gus, fill_value=0).sort_values(ascending=False)
    op = (t.groupby(["구", "dt"])["일"].nunique().unstack(fill_value=0)
          .reindex(index=br.index, columns=["평일", "토", "일·공휴일"], fill_value=0) / n_dt[["평일", "토", "일·공휴일"]])
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(op.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(op.values):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > .6 else "#0b2a4f")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{c}\n({n_dt[c]}일)" for c in op.columns]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(op))); ax.set_yticklabels([f"{g} ({br[g]}건)" for g in op.index])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3), minor=True); ax.set_yticks(np.arange(-.5, len(op)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=2); ax.tick_params(which="minor", length=0)
    title(fig, "구별 견인 운영률 — 그날 그 구에서 견인이 1건이라도 기록된 날의 비율",
          f"2025년 견인 {LOW_TOW_GU}건 이상 {len(op)}개 구, 점자블록 사유 견인 많은 순 (괄호: 점자블록 견인 건수)", fig=True)
    fig.subplots_adjust(left=.2, right=.97, top=top_of(fig, .45), bottom=.03)
    fig.savefig(FIG_DIR / "cal2_구별_운영률.png"); plt.close(fig)


def fig_daily(days):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": .12})
    x = days.index
    a1.bar(x, days["br"], color=np.where(days["dt"] == "평일", C_BR, C_MUTED), width=1)
    a1.set_ylabel("점자블록 견인(건)"); a1.grid(axis="y", color="#ecebe7", lw=.8); a1.set_axisbelow(True)
    a2.bar(x, days["rd"].fillna(0), color=C_RAIN, width=1)
    a2.set_ylabel("낮 강수(mm)"); a2.invert_yaxis(); a2.grid(axis="y", color="#ecebe7", lw=.8)
    a2.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    a2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{matplotlib.dates.num2date(v).month}월"))
    title(fig, "한 해 동안의 흐름 — 날짜별 점자블록 사유 견인과 낮(07–20시) 강수량",
          "막대 사이 빈칸이 주말·공휴일. 아래 막대는 위에서 아래로 그린 강수량", fig=True)
    fig.subplots_adjust(left=.07, right=.98, top=top_of(fig), bottom=.07)
    fig.savefig(FIG_DIR / "cal3_일별_흐름_강수.png"); plt.close(fig)


def fig_irr(irr):
    order = list(LABELS.values())
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for name, off, c in (("점자블록 견인", .15, C_BR), ("전체 견인", -.15, C_ALL)):
        s = irr[irr["모형"] == name].loc[order]
        sig = (s["p"] < .05).values
        ax.errorbar(s["IRR"], y + off, xerr=[s["IRR"] - s["하한"], s["상한"] - s["IRR"]], fmt="none",
                    ecolor=c, elinewidth=2)
        ax.scatter(s["IRR"], y + off, s=48, color=np.where(sig, c, "#fcfcfb"), edgecolor=c, lw=1.6, zorder=3,
                   label=name)
    ax.axvline(1, color="#8a8984", lw=1); ax.set_xscale("log")
    ax.set_xticks([.5, .6, .8, 1, 1.25, 1.5]); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    ax.set_yticks(y); ax.set_yticklabels(order); ax.grid(axis="x", color="#ecebe7", lw=.8)
    ax.set_xlabel("발생률비(IRR)  ← 견인 줄어듦 | 견인 늘어남 →")
    ax.legend(frameon=False, loc="lower right")
    title(ax, "기상·요일이 견인 건수를 몇 배로 바꾸나 (평일, 음이항 회귀)",
          "기준: 월요일·비 없는 날, 월별 계절 차이 통제. 채운 점 = p<0.05, 가로선 = 95% 신뢰구간")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cal4_기상요일_효과.png"); plt.close(fig)


def fig_calendar(g):
    m = g.pivot(index="dowc", columns="month", values="pred").values
    tier = g.pivot(index="dowc", columns="month", values="등급").values
    cmap = ListedColormap(["#e3e2dd", "#bcd6f3", "#5b9be3", "#1b4f99"])
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.imshow(tier, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), aspect="auto")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10, color="white" if tier[i, j] >= 2 else "#2b2a27")
    ax.set_xticks(range(12)); ax.set_xticklabels([f"{i}월" for i in range(1, 13)]); ax.xaxis.tick_top()
    ax.set_yticks(range(5)); ax.set_yticklabels(WD[:5])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 12), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=3); ax.tick_params(which="minor", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors[::-1]]
    ax.legend(handles, ["최우선 (상위 10%)", "집중 (상위 25%)", "보통", "완화"], ncol=4, frameon=False,
              loc="upper left", bbox_to_anchor=(0, -.04), fontsize=9.5)
    title(fig, "집중 단속 캘린더 — 비 없는 평일 기준 점자블록 견인 예측 일평균",
          "칸 안 숫자는 음이항 모형 예측값. 실제 평균은 칸마다 3~5일뿐이라 흔들림이 커서 색은 예측값으로 칠함", fig=True)
    fig.subplots_adjust(left=.05, right=.98, top=top_of(fig, .45), bottom=.14)
    fig.savefig(FIG_DIR / "cal5_집중단속_캘린더.png"); plt.close(fig)


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    tow, wx = load_tow(), load_weather()
    days = day_table(tow, wx)
    d, nb_br, nb_all = fit_models(days)
    irr = pd.concat([irr_table(nb_br, "점자블록 견인"), irr_table(nb_all, "전체 견인")])
    grid, q = calendar_grid(d, nb_br)

    fig_dow(days); fig_gu_op(tow, days); fig_daily(days); fig_irr(irr); fig_calendar(grid)

    by_dt = days.groupby("dt").agg(일수=("br", "size"), 점자블록_일평균=("br", "mean"), 전체_일평균=("all", "mean"),
                                   점자블록_합계=("br", "sum"))
    by_dt["점자블록_비중"] = by_dt["점자블록_합계"] / by_dt["점자블록_합계"].sum()
    with pd.ExcelWriter(RES_DIR / "calendar_summary.xlsx") as w:
        by_dt.to_excel(w, sheet_name="날짜유형별")
        irr.round(4).to_excel(w, sheet_name="음이항_IRR")
        grid.drop(columns=["rcat", "heat", "snow", "tanom", "prev_heavy"]).round(3) \
            .to_excel(w, sheet_name="캘린더_예측", index=False)
        days.reset_index().to_excel(w, sheet_name="일별", index=False)

    print(by_dt.round(3).to_string())
    print(f"모형 대상 평일 {len(d)}일, 음이항 alpha(점자블록) = {nb_br.params['alpha']:.3f}")
    print(irr.round(3).to_string())
    print("캘린더 등급 경계(예측 일평균):", np.round(q, 2))
    print("그림 저장:", FIG_DIR)
