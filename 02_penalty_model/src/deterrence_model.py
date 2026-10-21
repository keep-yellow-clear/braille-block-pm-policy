"""
점자블록 위 공유 PM 방치 — 기대처벌 모형 + 신고포상금 산정
-----------------------------------------------------------------
Becker(1968) 기대처벌 모형:  이용자는  B > p·F  이면 위반(점자블록 위 반납)한다.
  B : 위반 이득(정상 주차구역까지 우회하는 시간·요금 절약분, 원)
  p : 위반 1건이 적발돼 이용자에게 과태료가 실제 부과될 확률
  F : 과태료(원). 교통약자법 시행령(대통령령 제34871호, 2024.9.15 시행)
      별표3 제2호가 제11조②항(장애인을 위한 보도 이용방해·훼손) 위반의
      과태료를 50만 원으로 정한다 (법 제33조②항의 상한 100만 원 이내).
      이 저장소는 이 금액을 고정값으로 쓰고, 과태료 자체의 금액 산정은
      다루지 않는다. 아래 분석은 포상금 산정에 집중한다.
억지 조건:  p·F >= B   →   손익분기 적발확률  p* = B / F

신고포상이 적발확률에 미치는 영향:
  p = p0 + (1 - p0)·ρ·q
  p0 : 단속공무원·견인 등 기존 경로의 적발확률
  ρ  : 위반 1건당 시민이 신고할 확률 (포상금 R의 증가함수)
  q  : 신고가 과태료 부과로 확정되는 비율(증거 요건 충족률)
역산:  ρ* = (p* - p0) / ((1 - p0)·q)

포상금 R은 과태료 F의 일정 비율(국내 신고포상금 관행 20~40%)로 정하고,
신고자가 참여할 최소 조건(R·q >= 신고 1건의 시간비용)을 충족하는지 확인한다.

모든 수치는 아래 PARAMS에서 바꿀 수 있는 '가정'이며, 근거는 02_penalty_model/REPORT.md 표에 기재.
원자료(data/raw/)는 사용하지 않는다. p0를 실제 값으로 바꿀 때 REPORT.md 6-1 참고.

출력: 02_penalty_model/results/breakeven_table.csv   (우회 시나리오별 p*, ρ*)
      02_penalty_model/results/reward_table.csv       (포상금 비율별 R, 참여조건 충족 여부)
      02_penalty_model/results/figures/fig1_breakeven.png, fig2_reward.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import csv, os

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 02_penalty_model/
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
# 과태료 고정값. 근거: 교통약자법 시행령(대통령령 제34871호, 2024.9.15 시행) 별표3 제2호
FINE = 500_000

def benefit(detour_min, p=PARAMS):
    """위반 이득 B = 우회 시간가치 + 우회 중 추가 요금."""
    return detour_min * (p["wage_per_hour"] / 60 + p["pm_fare_per_min"])

def p_star(B, F=FINE):
    return B / F

def rho_star(pstar, p0=PARAMS["p0"], q=PARAMS["q"]):
    return max(0.0, (pstar - p0) / ((1 - p0) * q))

# ---- 1) 손익분기 표 (과태료 50만 원 고정) ------------------------------
rows = []
for d in PARAMS["detour_min"]:
    B = benefit(d)
    ps = p_star(B)
    rs = rho_star(ps)
    rows.append(dict(detour_min=d, B=round(B), fine=FINE,
                     p_star_pct=round(ps * 100, 2),
                     rho_star_pct=round(rs * 100, 2),
                     report_mult_vs_p0=round(ps / PARAMS["p0"], 1)))
with open(os.path.join(OUT, "breakeven_table.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# ---- 2) 포상금 표 (비율별 R, 참여조건 충족 여부) -----------------------
# 신고자 참여조건: R·q >= c_report  →  최소 포상금
c_report = PARAMS["report_cost_min"] * PARAMS["wage_per_hour"] / 60
R_min = c_report / PARAMS["q"]

reward_rows = []
for ratio in PARAMS["reward_ratio"]:
    R = FINE * ratio
    reward_rows.append(dict(reward_ratio_pct=round(ratio * 100, 1),
                            reward_amount=round(R),
                            r_min=round(R_min),
                            meets_participation=R >= R_min,
                            margin=round(R - R_min)))
with open(os.path.join(OUT, "reward_table.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=reward_rows[0].keys()); w.writeheader(); w.writerows(reward_rows)

# ---- 스타일 (보고서용 · 무채색 + 단일 강조색) --------------------------
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
SEQ = ["#c9d6e8", "#8fa9cc", "#4f72a6", "#1d3f73"]   # 단일 청색 계열, 밝음→어두움
ACCENT = "#b5482c"

def style(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#9a9a9a")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- 3) 손익분기 그림: 우회 시간별 p*, ρ* (과태료 50만 원 고정) --------
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
style(ax)
x = np.arange(len(rows))
w = 0.35
ax.bar(x - w/2, [r["p_star_pct"] for r in rows], width=w, color=SEQ[2], label="손익분기 적발확률 p*")
ax.bar(x + w/2, [r["rho_star_pct"] for r in rows], width=w, color=ACCENT, label="필요 신고확률 ρ*")
ax.axhline(PARAMS["p0"] * 100, color=MUTED, lw=1.2, ls="--")
ax.text(len(rows) - 0.5, PARAMS["p0"] * 100 + 0.15,
        f"현재 적발확률 가정 p0={PARAMS['p0']*100:.1f}%", color=MUTED, fontsize=8.5, ha="right")
ax.set_xticks(x, [f"우회 {r['detour_min']}분" for r in rows])
ax.set_ylabel("확률 (%)", color=INK, fontsize=10)
ax.set_title(f"손익분기 적발확률과 필요 신고확률  (과태료 F = {FINE:,}원 고정)",
             loc="left", color=INK, fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.text(0, -0.18, "과태료는 교통약자법 시행령 별표3 제2호 기준 50만 원으로 고정. 우회시간이 길수록 억지에 더 높은 신고확률이 필요하다.",
        transform=ax.transAxes, fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_breakeven.png")); plt.close(fig)

# ---- 4) 포상금 그림: 비율별 포상금 금액 vs 참여 최소조건 ---------------
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
style(ax)
x = np.arange(len(reward_rows))
colors = [SEQ[2] if r["meets_participation"] else "#ecd2c8" for r in reward_rows]
ax.bar(x, [r["reward_amount"] for r in reward_rows], color=colors, width=0.5)
for i, r in enumerate(reward_rows):
    ax.text(i, r["reward_amount"] + max(reward_rows, key=lambda r: r["reward_amount"])["reward_amount"] * 0.02,
            f"{r['reward_amount']:,}원", ha="center", fontsize=9, color=INK)
ax.axhline(R_min, color=ACCENT, lw=1.2, ls="--")
ax.text(0.02, 0.92, f"신고자 참여 최소조건 R_min 약 {R_min:,.0f}원 (점선, 세 비율 모두 상회)",
        transform=ax.transAxes, color=ACCENT, fontsize=8.5, ha="left", va="top")
ax.set_xticks(x, [f"{r['reward_ratio_pct']:g}%" for r in reward_rows])
ax.set_xlabel("포상금 비율 (과태료 대비)", color=INK, fontsize=10)
ax.set_ylabel("포상금 R (원)", color=INK, fontsize=10)
ax.set_title(f"포상금 비율별 금액  (과태료 F = {FINE:,}원 고정)",
             loc="left", color=INK, fontsize=12, fontweight="bold")
ax.text(0, -0.18, "점선(R_min) 위에 있으면 신고자가 신고 비용을 회수할 수 있는 수준의 포상금이다.",
        transform=ax.transAxes, fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_reward.png")); plt.close(fig)

# ---- 콘솔 요약 -------------------------------------------------------
print(f"과태료 F = {FINE:,}원 (교통약자법 시행령 별표3 제2호 기준, 고정)")
print(f"신고자 최소 포상금 R_min = 신고비용 {c_report:,.0f}원 / q {PARAMS['q']} = {R_min:,.0f}원")
for r in rows:
    print(r)
for r in reward_rows:
    print(r)
