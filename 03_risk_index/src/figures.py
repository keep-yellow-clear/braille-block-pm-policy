"""결과 그림 (PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from config import VARS, AXES, MC_DRAWS, YEAR, HOTSPOT_MIN_BRAILLE

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # 위해, 취약성, 대응 공백 (고정 순서)
ACCENT = "#eb6834"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SHORT = {"h1": "H1\n점자블록\n견인 지수", "h2": "H2\n상습 지점\n비중",
         "v1": "V1\n시각장애인\n수", "g1": "G1\n주차구역\n부족"}
DPI = 200


def setup_fonts():
    """OS별 한글 글꼴 지정 (맑은 고딕 / Apple SD Gothic Neo / Noto Sans CJK / 나눔고딕)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "NanumBarunGothic"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[주의] 한글 글꼴을 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다 (나눔고딕 설치 권장).")
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": AXIS, "font.size": 10})


def _title(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.18 / h, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.55 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def _clean(ax):
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_ranking(table, path):
    """table: 종합지수, 순위, '<축> 기여' 열."""
    t = table.sort_values("순위")
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.86, bottom=0.08)
    _title(fig, f"{YEAR}년 자치구별 종합 위험지수 — {'·'.join(t.index[:3])} 순",
           "막대 = 위해(H1·H2) + 취약성(V1) + 대응 공백(G1) 기여분의 합, 변수 동일 가중치. 0~100, 높을수록 위험")
    y = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for ax_name, color in zip(AXES, SERIES):
        v = t[f"{ax_name} 기여"].values
        ax.barh(y, v, left=left, height=0.66, color=color, label=ax_name, edgecolor=SURFACE, linewidth=1.5)
        left += v
    for yi, total in zip(y, t["종합지수"]):
        ax.text(total + 1, yi, f"{total:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("종합 위험지수")
    _clean(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_heatmap(scores, total, path):
    """scores: 변수별 0~100 점수, total: 종합지수."""
    order = total.sort_values(ascending=False).index
    M = scores.loc[order, VARS].values
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    fig.subplots_adjust(left=0.2, right=0.88, top=0.8, bottom=0.03)
    _title(fig, "어떤 변수가 순위를 만들었나 — 변수별 점수",
           "행 = 종합 위험지수 순 (괄호 = 종합점수). 색이 진할수록 그 변수에서 위험이 큼")
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] >= 55 else INK)
    ax.set_xticks(range(len(VARS)), [SHORT[v] for v in VARS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [f"{g} ({total[g]:.0f})" for g in order])
    ax.set_xticks(np.arange(-.5, len(VARS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.3, 0.025, 0.35]))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    cb.set_label("변수 점수 (0~100)", color=INK2, fontsize=8.5)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_stability(res, path):
    """res: monte_carlo() 결과."""
    r = res.sort_values("기본순위")
    sure = r.index[r["top5확률"] >= 0.95].tolist()
    n = len(r)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.84, bottom=0.08)
    _title(fig, f"가중치를 흔들어도 {'·'.join(sure)}는 상위 5위 안",
           f"가중치 {MC_DRAWS:,}회 무작위 변동: 진한 선 = 5~95% 구간, 연한 선 = 최소~최대, 흰 점 = 기본 순위")
    y = np.arange(n)[::-1]
    ax.hlines(y, r["최소"], r["최대"], color=AXIS, linewidth=5, capstyle="round")
    ax.hlines(y, r["p5"], r["p95"], color=INK2, linewidth=5, capstyle="round")
    ax.scatter(r["기본순위"], y, s=46, color="white", edgecolor=INK, linewidth=1.4, zorder=3)
    ax.axvline(5.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(5.6, y.max() + 0.75, "상위 5위 경계", fontsize=8.5, color=INK2, va="bottom")
    for yi, p in zip(y, r["top5확률"]):
        if p > 0:
            ax.text(n + 0.6, yi, f"{p:.0%}", va="center", fontsize=8.5, color=INK2)
    ax.text(n + 0.6, y.max() + 0.75, "상위 5위\n확률", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(y, [f"{g} (n={k})" for g, k in zip(r.index, r["n"])])
    ax.set_xticks([1, 4, 8, 12, 16] if n <= 18 else [1, 5, 10, 15, 20, 25])
    ax.set_xlim(0.3, n + 2.2)
    ax.set_ylim(-0.7, y.max() + 1.6)
    ax.set_xlabel("순위 (1 = 가장 위험)    n = 점자블록 견인 건수")
    _clean(ax)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_shrinkage(x, path):
    """x: build_variables() 결과 (h1_raw, h1, O)."""
    d = x.sort_values("O")
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.84, bottom=0.08)
    _title(fig, "표본이 작은 구는 도시 평균 쪽으로 보정됩니다",
           "점자블록 견인 지수 (같은 달 도시 평균 대비, 1.0 = 평균). 위에서부터 점자블록 견인 건수(n)가 적은 구")
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d["h1_raw"], d["h1"], color=AXIS, linewidth=2)
    ax.scatter(d["h1_raw"], y, s=48, facecolor=SURFACE, edgecolor=INK, linewidth=1.3, zorder=3, label="원값")
    ax.scatter(d["h1"], y, s=48, color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=4, label="표본 크기 보정 후")
    ax.axvline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(1.03, -0.6, "도시 평균 = 1", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"{g} (n={int(k)})" for g, k in zip(d.index, d["O"])])
    ax.set_xlim(0, max(d["h1_raw"].max(), d["h1"].max()) * 1.08)
    ax.set_xlabel("점자블록 견인 지수")
    _clean(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_excluded(ex, path):
    """ex: select_districts()의 제외 구 표."""
    t = ex.reset_index()
    cells = [[r["gu"], f"{int(r['전체견인']):,}", f"{r['월평균견인']:.0f}", int(r["유효월"]), r["제외 이유"]]
             for _, r in t.iterrows()]
    labels = ["자치구", f"{YEAR}년 견인", "월평균 견인", "유효 월", "제외 이유"]
    row_h, top, bottom = 0.34, 1.05, 0.2
    table_h = row_h * (len(cells) + 1.3)
    height = top + table_h + bottom
    fig = plt.figure(figsize=(9, height))
    _title(fig, f"{YEAR}년 분석에서 빠진 자치구 {len(cells)}곳",
           "구 이름 표기 문제가 아니라 견인 기록 자체가 거의 없음. 위험이 낮다는 뜻이 아니라 판단할 자료가 없다는 뜻")
    ax = fig.add_axes([0.02, bottom / height, 0.96, table_h / height])
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=labels, colWidths=[0.12, 0.13, 0.13, 0.1, 0.52],
                  bbox=[0, 0, 1, 1], cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    for (r, _), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_facecolor("#f0efec" if r == 0 else SURFACE)
        if r == 0:
            cell.set_text_props(color=INK2, fontweight="bold")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_parking_gap(gap, path, top_n=25):
    """gap: parking_gap.dong_gap() 결과. 점자블록 견인 상위 동과 주차구역 수."""
    t = gap.head(top_n).iloc[::-1]
    zero = (t["주차구역(보완 후)"] == 0).values
    fig, ax = plt.subplots(figsize=(9, 8.6))
    fig.subplots_adjust(left=0.17, right=0.9, top=0.86, bottom=0.07)
    _title(fig, f"점자블록 견인 상위 {top_n}개 동 중 {zero.sum()}곳은 킥보드 주차구역이 없습니다",
           f"{YEAR}년 점자블록 견인 건수. 막대 끝 숫자 = 주차구역 수(도로명 주소 보완 후), 괄호 = 보완 전")
    y = np.arange(len(t))
    ax.barh(y, t["점자블록 견인"], height=0.66, color=np.where(zero, ACCENT, AXIS))
    for yi, (b, zo, zn) in enumerate(zip(t["점자블록 견인"], t["주차구역(보완 전)"], t["주차구역(보완 후)"])):
        label = f"{zn}곳" + (f" ({zo})" if zo != zn else "")
        ax.text(b + 1.5, yi, label, va="center", fontsize=9,
                color=INK if zn == 0 else INK2, fontweight="bold" if zn == 0 else "normal")
    ax.set_yticks(y, [f"{g} {d}" for g, d in t.index])
    ax.set_xlim(0, t["점자블록 견인"].max() * 1.15)
    ax.set_xlabel(f"점자블록 견인 건수 ({YEAR})")
    _clean(ax)
    ax.legend(handles=[Patch(color=ACCENT, label="주차구역 0곳"), Patch(color=AXIS, label="주차구역 있음")],
              loc="lower right", frameon=False, fontsize=9)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
