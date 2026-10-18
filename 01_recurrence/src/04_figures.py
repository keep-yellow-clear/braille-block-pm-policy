"""4단계: 결과 그림 2장 생성.
- fig1_재발곡선.png : Kaplan-Meier 누적 재발확률 (본분석 vs 확인용)
- fig2_위험비.png   : 반복사건 Cox(Andersen-Gill) 위험비 forest plot
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, ticker
from config import RES_DIR, FIG_DIR, SELECTION, COLORS, FONT_PATH

try:
    fm.fontManager.addfont(FONT_PATH)
    font = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    font = "Malgun Gothic"   # Windows 기본 한글 폰트 (macOS는 'AppleGothic')
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

res = pd.read_pickle(RES_DIR / "survival_results.pkl")
km_df, cox, kms = res["km"], res["cox"], res["kms"]

# ---- Fig 1: 누적 재발확률 ----
fig, ax = plt.subplots(figsize=(8, 4.8))
for key, km in kms.items():
    sf, ci = km.survival_function_.iloc[:, 0], km.confidence_interval_
    ax.step(sf.index, 1 - sf, where="post", color=COLORS[key], lw=2, label=SELECTION[key]["label"])
    ax.fill_between(sf.index, 1 - ci.iloc[:, 0], 1 - ci.iloc[:, 1], step="post", color=COLORS[key], alpha=.15, lw=0)
main = km_df.set_index("분석").loc[SELECTION["main"]["label"]]
for t in (7, 14, 30):
    ax.axvline(t, color="#d9d8d3", lw=1, zorder=0)
    ax.annotate(f"{t}일: {main[f'{t}일 내 재발']*100:.0f}%", (t, main[f"{t}일 내 재발"]),
                xytext=(6, -14), textcoords="offset points", fontsize=10, color="#0b0b0b")
ax.set_xlim(0, 45); ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
ax.set_xlabel("견인 후 경과일"); ax.set_ylabel("다시 견인됐을 확률(누적)")
ax.grid(axis="y", color="#ecebe7", lw=.8); ax.legend(frameon=False, loc="lower right")
ax.set_title("상습 지번에서 견인 후 재발까지 걸린 날 (2025, Kaplan-Meier)", loc="left", fontsize=13)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_재발곡선.png", dpi=200)

# ---- Fig 2: 위험비 forest plot ----
ag = cox[cox["모형"] == "Andersen-Gill"]
order = ["누적 견인 순번(log)", "그날 견인 대수(log)", "점자블록 위반 포함", "주말 견인",
         "유형: 횡단보도(기준 차도)", "유형: 보호구역", "유형: 일반보도", "유형: 기타",
         "2분기(기준 1분기)", "3분기", "4분기"]
y = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(8, 5.6))
for (key, off) in (("main", .15), ("strict", -.15)):
    s = ag[ag["분석"] == SELECTION[key]["label"]].set_index("변수").loc[order]
    ax.errorbar(s["HR"], y + off, xerr=[s["HR"] - s["HR 하한"], s["HR 상한"] - s["HR"]], fmt="o", ms=6,
                color=COLORS[key], elinewidth=2, capsize=0, mec="#fcfcfb", mew=1.5, label=SELECTION[key]["label"])
ax.axvline(1, color="#8a8984", lw=1); ax.set_xscale("log")
ax.set_xticks([.4, .6, .8, 1, 1.25, 1.5]); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
ax.set_yticks(y); ax.set_yticklabels(order); ax.grid(axis="x", color="#ecebe7", lw=.8)
ax.set_xlabel("위험비(HR)  ← 재발이 늦어짐 | 재발이 빨라짐 →")
ax.set_title("재발 속도에 영향을 주는 요인 (반복사건 Cox, Andersen-Gill)", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_위험비.png", dpi=200)
print("그림 저장:", FIG_DIR)
