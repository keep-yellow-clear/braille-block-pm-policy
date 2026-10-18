"""
점자블록 위 공유 PM 방치 — 기대처벌 모형 + 손익분기·민감도 분석
-----------------------------------------------------------------
Becker(1968) 기대처벌 모형:  이용자는  B > p·F  이면 위반(점자블록 위 반납)한다.
  B : 위반 이득(정상 주차구역까지 우회하는 시간·요금 절약분, 원)
  p : 위반 1건이 적발돼 이용자에게 과태료가 실제 부과될 확률
  F : 과태료(원)
억지 조건:  p·F >= B   →   손익분기 적발확률  p* = B / F

신고포상이 적발확률에 미치는 영향:
  p = p0 + (1 - p0)·ρ·q
  p0 : 단속공무원·견인 등 기존 경로의 적발확률
  ρ  : 위반 1건당 시민이 신고할 확률 (포상금 R의 증가함수)
  q  : 신고가 과태료 부과로 확정되는 비율(증거 요건 충족률)
역산:  ρ* = (p* - p0) / ((1 - p0)·q)

모든 수치는 아래 PARAMS에서 바꿀 수 있는 '가정'이며, 근거는 03_penalty_model/REPORT.md 표에 기재.
원자료(data/raw/)는 사용하지 않는다. p0를 실제 값으로 바꿀 때 REPORT.md 6-1 참고.

출력: 03_penalty_model/results/breakeven_table.csv
      03_penalty_model/results/figures/fig1_breakeven.png, fig2_sensitivity.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import csv, os

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 03_penalty_model/
OUT = os.path.join(ANALYSIS_DIR, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- 한글 폰트 ---------------------------------------------------------
for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Medium" in f:
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]  # 없으면 다음 폰트 사용
plt.rcParams["axes.unicode_minus"] = False

# ---- 파라미터(가정) ----------------------------------------------------
PARAMS = dict(
    wage_per_hour=10_320,      # 2026 최저시급 → 시간가치 하한 대용
    pm_fare_per_min=150,       # 공유 PM 분당 요금(가정, 업체별 상이)
    detour_min=[1, 3, 5, 10],  # 지정주차구역까지 추가 소요(분) 시나리오
    p0=0.005,                  # 기존 적발확률 0.5% (가정 — 데이터로 대체 필요)
    q=0.6,                     # 신고→부과 확정률 (가정)
    report_cost_min=2,         # 신고 1건 소요시간(분)
    reward_ratio=[0.2, 0.3, 0.4],  # 국내 신고포상금 관행(과태료의 20~40%)
)
FINES = [30_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]  # 1,000,000 = 법정 상한(교통약자법 §33②)

def benefit(detour_min, p=PARAMS):
    """위반 이득 B = 우회 시간가치 + 우회 중 추가 요금."""
    return detour_min * (p["wage_per_hour"] / 60 + p["pm_fare_per_min"])

def p_star(B, F):
    return B / F

def rho_star(pstar, p0=PARAMS["p0"], q=PARAMS["q"]):
    return max(0.0, (pstar - p0) / ((1 - p0) * q))

# ---- 1) 역산 표 --------------------------------------------------------
rows = []
for d in PARAMS["detour_min"]:
    B = benefit(d)
    for F in FINES:
        ps = p_star(B, F)
        rs = rho_star(ps)
        rows.append(dict(detour_min=d, B=round(B), fine=F,
                         p_star_pct=round(ps * 100, 2),
                         rho_star_pct=round(rs * 100, 2),
                         report_mult_vs_p0=round(ps / PARAMS["p0"], 1)))
with open(os.path.join(OUT, "breakeven_table.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# 신고자 참여조건: R·q >= c_report  →  최소 포상금
c_report = PARAMS["report_cost_min"] * PARAMS["wage_per_hour"] / 60
R_min = c_report / PARAMS["q"]

# ---- 스타일 (보고서용 · 무채색 + 단일 강조색) --------------------------
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
SEQ = ["#c9d6e8", "#8fa9cc", "#4f72a6", "#1d3f73"]   # 단일 청색 계열, 밝음→어두움
ACCENT = "#b5482c"

def style(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#9a9a9a")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- 2) 손익분기 곡선 --------------------------------------------------
F_grid = np.linspace(20_000, 1_000_000, 400)
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
style(ax)
for d, c in zip(PARAMS["detour_min"], SEQ):
    B = benefit(d)
    ax.plot(F_grid / 10_000, p_star(B, F_grid) * 100, color=c, lw=2)
    ax.text(10.8, p_star(B, 100_000) * 100,
            f"우회 {d}분 (B≈{B:,.0f}원)", color=INK, fontsize=8.5, va="center",
            bbox=dict(fc="white", ec="none", pad=1))
ax.axhline(PARAMS["p0"] * 100, color=ACCENT, lw=1.2, ls="--")
ax.text(98, PARAMS["p0"] * 100 + 0.08, f"현재 적발확률 가정 p0={PARAMS['p0']*100:.1f}%",
        color=ACCENT, fontsize=8.5, ha="right")
for F in [5, 10, 50, 100]:
    ax.axvline(F, color=GRID, lw=0.8, zorder=0)
ax.set_xlim(0, 100); ax.set_ylim(0, 8)
ax.set_xlabel("과태료 F (만원)", color=INK, fontsize=10)
ax.set_ylabel("억지에 필요한 최소 적발확률 p* (%)", color=INK, fontsize=10)
ax.set_title("손익분기 적발확률  p* = B / F", loc="left", color=INK, fontsize=12, fontweight="bold")
ax.text(0, -0.2, "곡선 아래 영역 = 위반이 '이득'인 구간. 점선보다 곡선이 위에 있으면 현 단속만으로는 억지 불가.",
        transform=ax.transAxes, fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_breakeven.png")); plt.close(fig)

# ---- 3) 민감도 히트맵: 순억지 여유 (p·F − B) / B -----------------------
p_axis = np.array([0.25, 0.5, 1, 2, 3, 5, 10]) / 100
F_axis = np.array(FINES)
B_ref = benefit(3)
R = np.array([[p * F / B_ref for F in F_axis] for p in p_axis])   # 기대비용/이득 비율
# 3단계 판정: <1 억지 실패, 1~3 경계(손익분기 근처), >=3 확실한 억지
cls = np.digitize(R, [1, 3])
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#ecd2c8", "#f2f2ef", "#c9d6e8"])
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
im = ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
for i in range(len(p_axis)):
    for j in range(len(F_axis)):
        v = p_axis[i] * F_axis[j]
        ax.text(j, i, f"{v:,.0f}원", ha="center", va="center", fontsize=8.5, color=INK)
for k in range(len(F_axis) + 1):
    ax.axvline(k - 0.5, color="white", lw=2)
for k in range(len(p_axis) + 1):
    ax.axhline(k - 0.5, color="white", lw=2)
ax.set_xticks(range(len(F_axis)), [f"{f//10_000}만" for f in F_axis])
ax.set_yticks(range(len(p_axis)), [f"{p*100:g}%" for p in p_axis])
ax.set_xlabel("과태료 F", color=INK); ax.set_ylabel("적발확률 p", color=INK)
ax.tick_params(colors=MUTED, length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(f"기대비용 p·F (칸 안 숫자) vs 위반이득 B≈{B_ref:,.0f}원 (우회 3분 기준)",
             loc="left", color=INK, fontsize=11.5, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#ecd2c8", label="억지 실패 (p·F < B)"),
                   Patch(fc="#f2f2ef", ec="#cccccc", label="경계 (B ~ 3B)"),
                   Patch(fc="#c9d6e8", label="억지 (p·F ≥ 3B)")],
          loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_sensitivity.png")); plt.close(fig)

# ---- 콘솔 요약 -------------------------------------------------------
print(f"신고자 최소 포상금 R_min = 신고비용 {c_report:,.0f}원 / q {PARAMS['q']} = {R_min:,.0f}원")
for r in rows:
    if r["detour_min"] in (3, 10) and r["fine"] in (50_000, 100_000, 200_000):
        print(r)
