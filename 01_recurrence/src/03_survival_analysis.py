"""3단계: 재발 주기 생존분석 (Kaplan-Meier + 반복사건 Cox).

① 재발 간격 만들기
   지번별 견인일을 시간순으로 줄 세운 뒤, '이번 견인일 → 다음 견인일'까지의 날 수(gap)를 구한다.
   마지막 견인 이후 연말(12/31)까지 재견인이 없으면 '중도절단'(event=0)으로 둔다.

② Kaplan-Meier: "견인 후 t일이 지났을 때 아직 재견인되지 않았을 확률" S(t)
   S(t) = Π (1 - d_i / n_i)   (d_i: i일째에 재발한 간격 수, n_i: i일째까지 아직 재발 안 한 간격 수)

③ 반복사건 Cox: 재발을 빠르게/느리게 하는 요인의 위험비(HR)
   h(t | x) = h0(t) · exp(β·x),  HR = exp(β)
   - Andersen-Gill: 모든 재발을 같은 기준위험으로 보고, 누적 견인 순번을 변수로 넣음
   - PWP-GT: 1번째·2번째·…재발마다 기준위험을 따로 둠(순번 층화)
   - 두 모형 모두 지번별 군집 robust 표준오차 사용(같은 지번 기록끼리의 상관 보정)

입력: data/raw/ 2025년 견인 현황 원자료 2개 (config.load_tow_2025), data/processed/top_lots_*.csv (02단계)
출력: data/processed/intervals_{main,strict}.csv, results/survival_results.xlsx, results/survival_results.pkl
"""
import warnings
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import median_survival_times
from config import PROC_DIR, RES_DIR, OBS_END, SELECTION, load_tow_2025

warnings.filterwarnings("ignore")
END = pd.Timestamp(OBS_END)


def type_group(t: str) -> str:
    """위반 유형을 6개 군으로 묶음 (차도가 기준범주)."""
    if "점자" in t: return "점자블록"
    if "차도" in t: return "차도"
    if "횡단" in t: return "횡단보도"
    if "보호구역" in t: return "보호구역"
    if "보도" in t: return "일반보도"
    return "기타"


def build_intervals(df: pd.DataFrame, lots) -> pd.DataFrame:
    d = df[df["주소"].isin(lots)].copy()
    d["일"] = d["신고일"].dt.normalize()
    d["유형군"] = d["유형"].map(type_group)
    ev = (d.groupby(["주소", "일"])
            .agg(구=("구", "first"), 대수=("유형", "size"),
                 점자=("유형군", lambda s: int((s == "점자블록").any())),
                 유형군=("유형군", lambda s: s.mode().iloc[0]))
            .reset_index().sort_values(["주소", "일"]))
    ev["다음견인일"] = ev.groupby("주소")["일"].shift(-1)
    ev["event"] = ev["다음견인일"].notna().astype(int)              # 1=재발 관측, 0=중도절단
    ev["gap"] = (ev["다음견인일"].fillna(END) - ev["일"]).dt.days
    ev = ev[ev["gap"] > 0].copy()                                    # 12/31 견인은 관측기간 0일 → 제외
    ev["순번"] = ev.groupby("주소").cumcount() + 1
    ev["분기"] = "Q" + ev["일"].dt.quarter.astype(str)
    ev["주말"] = (ev["일"].dt.dayofweek >= 5).astype(int)
    return ev


def km_summary(ev, label):
    km = KaplanMeierFitter().fit(ev["gap"], ev["event"], label=label)
    F = lambda t: 1 - float(km.survival_function_at_times(t).iloc[0])
    lo, hi = median_survival_times(km.confidence_interval_).values[0]
    row = {"분석": label, "지번 수": ev["주소"].nunique(), "재발 간격 수": int(ev["event"].sum()),
           "중도절단": int((ev["event"] == 0).sum()), "재발 간격 중앙값(일)": km.median_survival_time_,
           "중앙값 95% CI": f"{lo:.0f}–{hi:.0f}"}
    for t in (3, 7, 14, 30):
        row[f"{t}일 내 재발"] = round(F(t), 4)
    return km, row


def design_matrix(ev):
    X = ev.copy()
    X["log_대수"] = np.log(X["대수"])
    X["log_순번"] = np.log(X["순번"])
    X["층"] = X["순번"].clip(upper=11)                               # PWP: 1~10번째, 11번째 이상
    X = pd.get_dummies(X, columns=["유형군", "분기"], dtype=int)
    covs = ["log_대수", "점자", "주말",
            "유형군_횡단보도", "유형군_보호구역", "유형군_일반보도", "유형군_기타",
            "분기_Q2", "분기_Q3", "분기_Q4"]                         # 기준범주: 차도, 1분기 / 점자블록은 '점자' 플래그로 반영
    return X, [c for c in covs if c in X.columns]


LABELS = {"log_순번": "누적 견인 순번(log)", "log_대수": "그날 견인 대수(log)", "점자": "점자블록 위반 포함",
          "주말": "주말 견인", "유형군_횡단보도": "유형: 횡단보도(기준 차도)", "유형군_보호구역": "유형: 보호구역",
          "유형군_일반보도": "유형: 일반보도", "유형군_기타": "유형: 기타", "분기_Q2": "2분기(기준 1분기)",
          "분기_Q3": "3분기", "분기_Q4": "4분기"}


def cox_table(m, model, label):
    s = m.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "HR 하한", "HR 상한", "p"]
    s.index = [LABELS.get(i, i) for i in s.index]
    s.insert(0, "모형", model); s.insert(0, "분석", label)
    return s.reset_index(names="변수")


if __name__ == "__main__":
    df = load_tow_2025()                               # 원자료 직접 읽기 (2025년, 오신고 제외)

    km_rows, cox_tabs, kms = [], [], {}
    for key, s in SELECTION.items():
        lots = pd.read_csv(PROC_DIR / f"top_lots_{key}.csv")["주소"]
        ev = build_intervals(df, lots)
        ev.to_csv(PROC_DIR / f"intervals_{key}.csv", index=False, encoding="utf-8-sig")

        km, row = km_summary(ev, s["label"]); kms[key] = km; km_rows.append(row)

        X, covs = design_matrix(ev)
        ag = CoxPHFitter().fit(X[["gap", "event", "주소", "log_순번"] + covs], "gap", "event",
                               cluster_col="주소", robust=True)
        pwp = CoxPHFitter().fit(X[["gap", "event", "주소", "층"] + covs], "gap", "event",
                                strata=["층"], cluster_col="주소", robust=True)
        cox_tabs += [cox_table(ag, "Andersen-Gill", s["label"]), cox_table(pwp, "PWP-GT(순번 층화)", s["label"])]
        print(f"[{s['label']}] 일치도 AG={ag.concordance_index_:.3f}, PWP={pwp.concordance_index_:.3f}")

    km_df = pd.DataFrame(km_rows)
    cox_df = pd.concat(cox_tabs, ignore_index=True)
    print(km_df.to_string()); print(cox_df.round(3).to_string())

    with pd.ExcelWriter(RES_DIR / "survival_results.xlsx") as w:
        km_df.to_excel(w, sheet_name="KM 요약", index=False)
        cox_df.round(4).to_excel(w, sheet_name="반복사건 Cox", index=False)
        for key, km in kms.items():
            sf = km.survival_function_.iloc[:, 0]
            pd.DataFrame({"경과일": sf.index, "미재발확률 S(t)": sf.values, "누적 재발확률 1-S(t)": 1 - sf.values}) \
              .query("경과일 <= 60").round(4).to_excel(w, sheet_name=f"KM곡선_{key}", index=False)
    pd.to_pickle({"km": km_df, "cox": cox_df, "kms": kms}, RES_DIR / "survival_results.pkl")
