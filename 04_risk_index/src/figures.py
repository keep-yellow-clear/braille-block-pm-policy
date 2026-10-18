"""그림 4장 (PNG)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from config import VARS, AXES
from risk_index import axis_contributions, scores_table, composite

INK, MUT = "#222222", "#666666"
COL = {"위해": "#D55E00", "취약성": "#0072B2", "대응 공백": "#E69F00"}   # Okabe-Ito


def setup_fonts():
    import glob
    for f in glob.glob("/usr/share/fonts/**/NotoSansCJK-*.ttc", recursive=True):
        fm.fontManager.addfont(f)
    plt.rcParams.update({"font.family": next((f for f in ["Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "NanumGothic"]
                                             if f in {x.name for x in fm.fontManager.ttflist}), "sans-serif"),
                         "axes.unicode_minus": False, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})


def fig_ranking(x, out):
    contrib = axis_contributions(x); tot = contrib.sum(axis=1)
    order = tot.sort_values(ascending=False).index; contrib = contrib.loc[order]
    fig, ax = plt.subplots(figsize=(9, 9.2)); y = np.arange(len(order))[::-1]; left = np.zeros(len(order))
    for a in AXES:
        ax.barh(y, contrib[a], left=left, color=COL[a], edgecolor="white", linewidth=.6, height=.72); left += contrib[a].values
    for i, g in enumerate(order): ax.text(tot[g] + .8, y[i], f"{tot[g]:.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(list(order), fontsize=10); ax.set_xlim(0, 100)
    ax.set_xlabel("종합 위험지수 (0~100, 높을수록 위험)", color=MUT)
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=COL[a], label=a) for a in AXES], loc="lower right", frameon=False)
    fig.suptitle("자치구별 위험지수 — " + "·".join(g[:-1] for g in order[:4]) + "가 상위", x=.01, ha="left", fontsize=14, fontweight="bold")
    ax.set_title("막대 = 위해(H1·H2) + 취약성(V1) + 대응 공백(G1) 기여분의 합 (변수 동일 가중치)", loc="left", fontsize=10, color=MUT)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, facecolor="white"); plt.close(fig)


def fig_heatmap(x, out):
    S = scores_table(x); tot = composite(x); order = tot.sort_values(ascending=False).index
    names = ["H1\n점자블록\n견인 지수", "H2\n상습 지점\n비중", "V1\n시각장애인\n수", "G1\n주차구역\n부족"]
    M = S.loc[order, VARS].values
    fig, ax = plt.subplots(figsize=(6.8, 9.4)); im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 68 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9); ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{g}  ({tot[g]:.0f})" for g in order], fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label("변수 점수 (0~100, 높을수록 위험)", color=MUT); cb.outline.set_visible(False)
    fig.suptitle("어떤 변수가 순위를 만들었나 — 변수별 점수", x=.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(.01, .925, "행 순서 = 종합 위험지수 순, 괄호 안 = 종합점수. 색이 진할수록 그 변수에서 위험이 큼", fontsize=9, color=MUT)
    fig.tight_layout(rect=(0, 0, 1, .915)); fig.savefig(out, facecolor="white"); plt.close(fig)


def fig_stability(res, out):
    fig, ax = plt.subplots(figsize=(9, 9)); yy = np.arange(len(res))[::-1]
    for i, (g, r) in enumerate(res.iterrows()):
        ax.plot([r["최소"], r["최대"]], [yy[i]] * 2, color=INK, alpha=.2, lw=6, solid_capstyle="round")
        ax.plot([r.p5, r.p95], [yy[i]] * 2, color=INK, lw=6, solid_capstyle="round")
        ax.plot(r["기본순위"], yy[i], "o", color="white", markeredgecolor=INK, markersize=6, zorder=5)
    ax.axvline(5.5, color=INK, ls="--", lw=.8); ax.text(5.7, len(res) - .4, "상위 5위 경계", fontsize=8.5, va="bottom")
    ax.set_yticks(yy); ax.set_yticklabels([f"{g}  (n={int(r.n)})" for g, r in res.iterrows()], fontsize=10)
    ax.set_xlim(.5, 25.5); ax.set_xticks([1, 5, 10, 15, 20, 25]); ax.set_xlabel("순위 (1 = 가장 위험)", color=MUT)
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    top = [g for g, r in res.iterrows() if r["top5확률"] >= .95]
    fig.suptitle(f'가중치를 흔들어도 {"·".join(g[:-1] for g in top)}은 상위 5위 안', x=.01, ha="left", fontsize=14, fontweight="bold")
    ax.set_title("가중치 5,000회 무작위 변동: 진한 선 = 5~95% 구간, 연한 선 = 최소~최대, 흰 점 = 기본 순위, n = 점자블록 견인 건수",
                 loc="left", fontsize=9, color=MUT)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, facecolor="white"); plt.close(fig)


def fig_shrinkage(x, out):
    xx = x.sort_values("O"); fig, ax = plt.subplots(figsize=(9, 9)); yy = np.arange(len(xx))
    for i, (g, r) in enumerate(xx.iterrows()):
        ax.plot([r.sir, r.h1], [i, i], color="#999999", lw=2, zorder=1)
        ax.plot(r.sir, i, "o", mfc="white", mec=INK, ms=7, zorder=3); ax.plot(r.h1, i, "o", color=COL["위해"], ms=7, zorder=4)
    ax.axvline(1, color=MUT, ls="--", lw=.8); ax.text(1.02, len(xx) - .3, "도시 평균 = 1", fontsize=8.5, color=MUT)
    ax.set_yticks(yy); ax.set_yticklabels([f"{g}  (n={int(r.O)})" for g, r in xx.iterrows()], fontsize=10)
    ax.set_xlabel("점자블록 견인 지수 (같은 달 도시 평균 대비, 1.0 = 평균)", color=MUT)
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="원값"),
                       plt.Line2D([], [], marker="o", ls="", color=COL["위해"], label="표본 크기 보정 후")], loc="lower right", frameon=False)
    fig.suptitle("표본이 작은 구는 도시 평균 쪽으로 보정됩니다", x=.01, ha="left", fontsize=14, fontweight="bold")
    ax.set_title("위에서부터 점자블록 견인 건수(n)가 적은 구. 같은 달끼리만 비교해 자료 기간 차이를 보정함", loc="left", fontsize=9.5, color=MUT)
    ax.invert_yaxis(); fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, facecolor="white"); plt.close(fig)
