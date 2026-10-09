"""08. 전환 규칙 비교: 현행 60분 규칙 vs 모형 기반 동적 규칙 (CLAUDE.md 6절 Algorithm 3, 9절 해석 원칙).

대상: 테스트 기간(10~12월) 즉시호출, Stage 1(배차 전). 상태 불명 제외.
점검: s = 0, 5, 10, ... 분마다 아직 배차·취소되지 않은 콜(T1 >= s)을 본다.
  규칙 A(현행): s = 60에 아직 미배차면 전환 대상.
  규칙 B(동적): 06단계 모형으로 '다음 30분 내 최종 포기 확률'을 계산해 처음으로 θ 이상이 된 시점에 전환 대상.
               (피처는 그 시점에 알 수 있는 값만. 부하는 점검 시점 값으로 고정)
지표(규칙별, 전체·야간)
  하루 평균 전환 대상 건수(운영 부담)
  재현율 = 배차 전 최종 포기 콜 중 취소 '전에' 전환 대상이 된 비율, 미리 잡은 시간 = 취소 시각 - 전환 시점(중앙값)
  정밀도 = 전환 대상 중 실제 배차 전 최종 포기 비율, 불필요 전환 = 전환 대상 중 전환 후 10분 안에 배차된 비율
  60분 전에 이미 떠난 비율 = 배차 전 최종 포기 중 취소가 접수 후 60분 전(규칙 A가 구조적으로 못 잡는 몫)
추가: 배차 전 최종 포기 콜 중, 취소 직전 점검 시점에 모형이 '다음 10분 내 배차 확률 >= 50%'로 본 비율
      규칙 B의 전환 시점 분포(접수 직후 0분에 표시된 비율 등), 60분 전에 떠난 비율의 연간(1~12월) 값
해석: 조기 탐지 성능과 운영 부담만 보고한다. "전환하면 포기가 X% 줄어든다" 같은 인과 주장은 하지 않는다.
      바우처·임차택시 공급량과 비용 자료가 없어, 전환 대상을 실제로 처리할 수 있는지는 알 수 없다(한계).
출력: outputs/tables/policy_compare.csv, policy_theta_curve.csv, outputs/figures/fig_07_policy.png
"""
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from utils import (COLOR, MODELS, NIGHT_HOURS, apply_style, clean, draw_figures, fig_title, fig_width, save_fig,
                   save_table)

hm = importlib.import_module("06_hazard_model")      # 06단계와 같은 데이터 준비·예측 함수를 그대로 쓴다

RULE_A_MIN = 60
HORIZON_BINS = 6        # 다음 30분
DISP_BINS = 2           # 다음 10분
THETAS = np.round(np.arange(0.02, 0.601, 0.005), 3)


def score_checks(bst, feats, test, load):
    """모든 점검 시점(콜 x s)의 '다음 30분 최종 포기 확률'과 '다음 10분 배차 확률'."""
    parts = []
    for b in range(F.N_BINS):
        surv = test[test.T1 >= b * F.BIN]
        if not len(surv):
            break
        Fp, _ = hm.predict_cif_model(bst, feats, surv, load, s_bin=b, n_bins=HORIZON_BINS)
        parts.append(pd.DataFrame({"i": surv.index.to_numpy(), "b": b, "p_ab30": Fp[:, HORIZON_BINS, 1],
                                   "p_d10": Fp[:, DISP_BINS, 0]}))
    return pd.concat(parts, ignore_index=True).sort_values(["i", "b"], kind="stable").reset_index(drop=True)


def first_flag(chk, theta, n):
    """규칙 B: 콜마다 처음으로 p_ab30 >= θ가 된 점검 시점(분). 없으면 NaN."""
    hit = chk[chk.p_ab30.to_numpy() >= theta]
    f = np.full(n, np.nan)
    if len(hit):
        u, idx = np.unique(hit["i"].to_numpy(), return_index=True)
        f[u] = hit["b"].to_numpy()[idx] * F.BIN
    return f


def metrics(test, flag, mask, n_days):
    """flag: 콜별 전환 시점(분, NaN = 전환 안 됨). mask: 집단(전체/야간)."""
    T, E = test.T1.to_numpy(), test.E1_ab.to_numpy()
    fl = ~np.isnan(flag) & mask
    ab = (E == 2) & mask
    caught = ab & fl & (flag < T)
    lead = (T - flag)[caught]
    disp10 = fl & (E == 1) & ((T - flag) <= 10)
    return {"하루 평균 전환 대상(건)": fl.sum() / n_days,
            "재현율%(배차 전 최종 포기를 취소 전에 잡음)": caught.sum() / ab.sum() * 100,
            "미리 잡은 시간 중앙값(분)": float(np.median(lead)) if len(lead) else np.nan,
            "정밀도%(전환 대상 중 배차 전 최종 포기)": (fl & (E == 2)).sum() / max(fl.sum(), 1) * 100,
            "불필요 전환%(전환 후 10분 안에 배차)": disp10.sum() / max(fl.sum(), 1) * 100,
            "전환 대상 콜 수": int(fl.sum()), "배차 전 최종 포기 콜 수": int(ab.sum())}


def main():
    t0 = time.time()
    imm, load = hm.prepare()
    bst = lgb.Booster(model_str=(MODELS / "lgbm_stage1.txt").read_text(encoding="utf-8"))
    feats = bst.feature_name()
    test = imm[(imm.split == "test") & (imm.E1_ab != 0)].reset_index(drop=True)
    n_days = test.t_request.dt.normalize().nunique()
    night = test.t_request.dt.hour.isin(NIGHT_HOURS).to_numpy()
    groups = {"전체": np.ones(len(test), bool), "야간(20~01시)": night}
    print(f"  테스트 콜 {len(test):,} ({n_days}일), 야간 {night.sum():,}")

    chk = score_checks(bst, feats, test, load)
    print(f"  점검 시점 {len(chk):,}개 예측 ({time.time() - t0:.0f}초)")

    flag_a = np.where(test.T1.to_numpy() >= RULE_A_MIN, float(RULE_A_MIN), np.nan)

    # θ 곡선
    curve = []
    for th in THETAS:
        f = first_flag(chk, th, len(test))
        for g, m in groups.items():
            curve.append({"θ": th, "집단": g, **metrics(test, f, m, n_days)})
    curve = pd.DataFrame(curve)
    save_table(curve.round(3), "policy_theta_curve.csv")

    # 규칙 A와 하루 전환 건수가 같아지는 θ(전체 기준, 같은 θ를 야간에도 적용)
    a_all = metrics(test, flag_a, groups["전체"], n_days)
    c_all = curve[curve["집단"] == "전체"]
    th_match = float(c_all.iloc[(c_all["하루 평균 전환 대상(건)"] - a_all["하루 평균 전환 대상(건)"]).abs().argmin()]["θ"])
    flag_b = first_flag(chk, th_match, len(test))

    T, E = test.T1.to_numpy(), test.E1_ab.to_numpy()
    rows = []
    for g, m in groups.items():
        for rule, fl, th in [("A 현행: 60분 경과 & 미배차", flag_a, np.nan),
                             (f"B 동적: 다음 30분 최종 포기 확률 ≥ θ (규칙 A와 같은 건수)", flag_b, th_match)]:
            rows.append({"구분": "1 같은 운영 부담에서 비교", "집단": g, "규칙": rule, "θ": th, **metrics(test, fl, m, n_days)})
    for g, m in groups.items():
        for th in [0.05, 0.10, 0.15, 0.20, 0.30]:
            rows.append({"구분": "2 θ별 규칙 B", "집단": g, "규칙": "B 동적", "θ": th,
                         **metrics(test, first_flag(chk, th, len(test)), m, n_days)})
    # 연간(1~12월) 즉시호출 전체의 배차 전 최종 포기(보고서 결과 3이 인용하는 기준)
    hr_all = imm.t_request.dt.hour
    ab_year = {"전체": imm[imm.E1_ab == 2], "야간(20~01시)": imm[(imm.E1_ab == 2) & hr_all.isin(NIGHT_HOURS)]}
    for g, m in groups.items():
        ab = (E == 2) & m
        rows.append({"구분": "3 구조적 한계", "집단": g, "규칙": "A 현행",
                     "지표": "배차 전 최종 포기 중 접수 후 60분 전에 이미 떠난 비율% [평가 기간 10~12월]",
                     "값": (T[ab] < RULE_A_MIN).mean() * 100})
        rows.append({"구분": "3 구조적 한계", "집단": g, "규칙": "A 현행",
                     "지표": "배차 전 최종 포기 중 접수 후 60분 전에 이미 떠난 비율% [연간 1~12월]",
                     "값": (ab_year[g].T1 < RULE_A_MIN).mean() * 100})
        # 규칙 B(같은 운영 부담 θ)가 전환 대상으로 표시하는 시점: 대부분 접수 직후다
        fb = flag_b[~np.isnan(flag_b) & m]
        fc = flag_b[~np.isnan(flag_b) & m & (E == 2) & (flag_b < T)]
        for name, val in [("전환 대상 중 접수 직후(0분)에 표시된 비율%", (fb == 0).mean() * 100),
                          ("전환 대상 중 5~10분에 표시된 비율%", ((fb >= 5) & (fb <= 10)).mean() * 100),
                          ("전환 대상 중 15~30분에 표시된 비율%", ((fb >= 15) & (fb <= 30)).mean() * 100),
                          ("전환 대상 중 30분 넘어 표시된 비율%", (fb > 30).mean() * 100),
                          ("전환 시점 p90(분)", np.percentile(fb, 90)),
                          ("취소 전에 잡은 최종 포기 중 접수 직후(0분)에 표시된 비율%", (fc == 0).mean() * 100)]:
            rows.append({"구분": "5 동적 규칙의 전환 시점(같은 운영 부담 θ)", "집단": g, "규칙": "B 동적", "θ": th_match,
                         "지표": name, "값": val})
        # 취소 직전 점검 시점(취소가 일어난 5분 구간의 시작)에서 '다음 10분 배차 확률 >= 50%'
        last = chk.merge(pd.DataFrame({"i": np.where(ab)[0], "b": np.floor(T[ab] / F.BIN).astype(int)}), on=["i", "b"])
        rows.append({"구분": "4 곧 배차될 상황에서 떠남", "집단": g, "규칙": "모형",
                     "지표": "배차 전 최종 포기 중 취소 직전 점검에서 '다음 10분 배차 확률 ≥ 50%'였던 비율%",
                     "값": (last.p_d10 >= 0.5).mean() * 100})
        rows.append({"구분": "4 곧 배차될 상황에서 떠남", "집단": g, "규칙": "모형",
                     "지표": "같은 콜들의 취소 직전 '다음 10분 배차 확률' 중앙값%", "값": last.p_d10.median() * 100})
    t = pd.DataFrame(rows).sort_values("구분", kind="stable")
    save_table(t.round(3), "policy_compare.csv")
    pd.set_option("display.width", 250)
    print(t.drop(columns=["전환 대상 콜 수", "배차 전 최종 포기 콜 수"], errors="ignore").round(2).to_string(index=False))

    draw_figures(fig_policy, test, curve, a_all, metrics(test, flag_a, groups["야간(20~01시)"], n_days), th_match,
                 flag_a, flag_b, groups)
    print(f"[08_policy_simulation] 완료 {time.time() - t0:.0f}초")


def fig_policy(test, curve, a_all, a_night, th_match, flag_a, flag_b, groups, baselines=None):
    """baselines(08b가 넘김): {"R0": df, "R2": df, "R3": df, "R1": (x, y)} 단순 규칙의 (가) 패널용 곡선·점.
    각 df는 '하루 평균 전환 대상(건)'과 재현율 열을 가진다. 없으면 08 단독 실행 결과(A·B만)를 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    ann = not clean()
    h_in = (3.5 if ann else 3.2) + ((1.0 if ann else 0.35) if baselines else 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fig_width(), h_in),
                                 gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.25]})
    if ann:
        fig.subplots_adjust(left=0.095, right=0.98, top=0.66 if baselines else 0.76, bottom=0.37 if baselines else 0.29)
    else:
        fig.subplots_adjust(left=0.12, right=0.98, top=0.72 if baselines else 0.78, bottom=0.16)
    blue, gray = COLOR["board"], COLOR["ink2"]
    pct = FuncFormatter(lambda v, _: f"{v:.0f}%")
    rc, dc = "재현율%(배차 전 최종 포기를 취소 전에 잡음)", "하루 평균 전환 대상(건)"

    if baselines:      # 단순 규칙: B보다 아래에 그린다. 색은 기준 팔레트의 다른 슬롯(포기 주황과 겹치지 않게)
        s = baselines["R0"].sort_values(dc)
        a1.plot(s[dc], s[rc], color=COLOR["muted"], lw=1.3, ls=(0, (2, 2)), label="R0 무작위", zorder=1)
        s = baselines["R2"].sort_values(dc)
        a1.plot(s[dc], s[rc], color=COLOR["series"][6], lw=1.6, label="R2 룩업표(구×시각)", zorder=2)
        s = baselines["R3"].sort_values(dc)
        a1.plot(s[dc], s[rc], color=COLOR["series"][2], lw=1.8, label="R3 모형, 접수 시점만", zorder=2)
        x1, y1 = baselines["R1"]
        a1.plot([x1], [y1], "D", ms=5.5, color=COLOR["series"][4], mec=COLOR["surface"], mew=1.0, zorder=4,
                label="R1 야간 일괄")

    c = curve[curve["집단"] == "전체"].sort_values(dc)
    a1.plot(c[dc], c[rc], color=blue, lw=2, label="B 동적 규칙(θ를 바꿔 가며)")
    m = c.loc[(c["θ"] - th_match).abs().idxmin()]
    a1.plot([m[dc]], [m[rc]], "o", ms=6, color=blue, mec=COLOR["surface"], mew=1.2, zorder=4,
            label="B 동적 규칙(A와 같은 하루 건수)")
    a1.plot([a_all[dc]], [a_all[rc]], "o", ms=6, color=gray, mec=COLOR["surface"], mew=1.2, zorder=4,
            label="A 현행 60분 규칙")
    if ann and not baselines:
        a1.annotate(f"A 현행 60분 규칙\n하루 {a_all[dc]:.0f}건, 재현율 {a_all[rc]:.0f}%", xy=(a_all[dc], a_all[rc]),
                    xytext=(a_all[dc] * 1.45, max(a_all[rc] - 9, 2)), fontsize=7, color=COLOR["ink"],
                    arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))
        a1.annotate(f"B (θ = {th_match:.3f})\n같은 건수, 재현율 {m[rc]:.0f}%", xy=(m[dc], m[rc]),
                    xytext=(m[dc] * 1.45, m[rc] - 12), fontsize=7, color=COLOR["ink"],
                    arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))
    a1.set_xlim(0, min(c[dc].max(), a_all[dc] * 4))
    a1.set_ylim(0, 100)
    a1.yaxis.set_major_formatter(pct)
    a1.set_xlabel("하루 평균 전환 대상(건)")
    a1.set_ylabel("배차 전 최종 포기를\n미리 잡은 비율" if not ann else "배차 전 최종 포기를 미리 잡은 비율")
    a1.set_title("(가) 운영 부담 대비 조기 탐지", loc="left", fontsize=8.5 if ann else None, pad=4)
    a1.grid(axis="y"); a1.set_axisbelow(True); a1.tick_params(length=0)

    T, E = test.T1.to_numpy(), test.E1_ab.to_numpy()
    hr = test.t_request.dt.hour.to_numpy()
    rows = []
    for h in range(24):
        ab = (E == 2) & (hr == h)
        for name, fl in [("A", flag_a), ("B", flag_b)]:
            caught = ab & ~np.isnan(fl) & (fl < T)
            rows.append({"hour": h, "rule": name, "recall": caught.sum() / max(ab.sum(), 1) * 100, "n": int(ab.sum())})
    r = pd.DataFrame(rows).pivot(index="hour", columns="rule", values="recall")
    a2.axvspan(19.5, 23.5, color="#f0efec", lw=0, zorder=0)
    a2.axvspan(-0.5, 1.5, color="#f0efec", lw=0, zorder=0)
    a2.plot(r.index, r["A"], color=gray, lw=2, label="A 현행 60분 규칙")
    a2.plot(r.index, r["B"], color=blue, lw=2, label="B 동적 규칙(같은 하루 건수)")
    a2.set_xlim(-0.5, 23.5); a2.set_xticks(range(0, 24, 3)); a2.set_ylim(0, 100)
    a2.yaxis.set_major_formatter(pct)
    a2.set_xlabel("접수 시간대(시)")
    a2.set_title("(나) 시간대별 조기 탐지 비율(음영 = 야간)" if ann else "(나) 시간대별 조기 탐지 비율", loc="left",
                 fontsize=8.5 if ann else None, pad=4)
    if ann:
        a2.legend(loc="upper center", fontsize=6.8, ncol=2)
    else:
        h1, l1 = a1.get_legend_handles_labels()
        fig.legend(h1, l1, loc="upper left", bbox_to_anchor=(0.1, 1.0), ncol=3 if baselines else 2,
                   handlelength=1.6, columnspacing=1.0)
    if ann and baselines:
        h1, l1 = a1.get_legend_handles_labels()
        fig.legend(h1, l1, loc="upper left", bbox_to_anchor=(0.08, 0.86), ncol=4, fontsize=6.8, handlelength=1.4, columnspacing=1.0)
    a2.grid(axis="y"); a2.set_axisbelow(True); a2.tick_params(length=0)

    b_all = curve[(curve["집단"] == "전체") & (curve["θ"] == m["θ"])].iloc[0]
    earlier = b_all["미리 잡은 시간 중앙값(분)"] > a_all["미리 잡은 시간 중앙값(분)"]
    if baselines:      # 08b: 단순 규칙과의 비교 결과를 제목에 정직하게 반영한다(모형 접수 시점 1회 vs 5분 재점검 포함)
        mt = baselines["match"]
        fig_title(fig, f"같은 하루 {a_all[dc]:.0f}건에서 접수 시점 모형은 현행 규칙보다 {mt['R3'] / max(a_all[rc], 1e-9):.1f}배, "
                       f"룩업표보다 {mt['R3'] / max(mt['R2'], 1e-9):.1f}배 더 잡는다",
                  f"재현율: 현행 {a_all[rc]:.1f}%, 룩업표 {mt['R2']:.1f}%, 모형(접수 시점만) {mt['R3']:.1f}%, "
                  f"동적(5분 재점검) {b_all[rc]:.1f}%. 재점검으로 {b_all[rc] - mt['R3']:+.1f}%p", y_sub=0.9)
    else:
        fig_title(fig, f"같은 운영 부담으로 모형 기반 규칙은 포기를 {b_all[rc] / max(a_all[rc], 1e-9):.1f}배 더 많이"
                       + (", 더 일찍 잡아낸다" if earlier else " 잡아낸다"),
                  f"하루 {a_all[dc]:.0f}건 기준 재현율: 현행 {a_all[rc]:.1f}% → 동적 {b_all[rc]:.1f}%, "
                  f"미리 잡은 시간 중앙값: 현행 {a_all['미리 잡은 시간 중앙값(분)']:.0f}분 → 동적 {b_all['미리 잡은 시간 중앙값(분)']:.0f}분 "
                  f"(θ = {th_match:.3f})", y_sub=0.9)
    note = ("주: 테스트 기간(2025년 10~12월) 즉시호출의 배차 전 단계. 동적 규칙 = 5분마다 06단계 모형으로 '다음 30분 내 최종 포기 확률'을\n"
            "    계산해 처음 θ 이상이 된 시점에 전환 대상으로 표시. 재현율 = 배차 전 최종 포기 콜을 취소 전에 전환 대상으로 잡은 비율.\n"
            "    (나)의 동적 규칙은 서울 전체에 같은 θ를 쓰므로 시간대별 전환 건수는 현행 규칙과 다름. 회색 음영 = 야간(20~01시).\n"
            "    조기 탐지 성능과 운영 부담만 비교한 것이며, 전환이 실제로 포기를 줄인다는 인과 효과는 추정하지 않음.\n"
            "    바우처·임차택시 공급량과 비용 자료가 없어 전환 대상을 실제로 처리할 수 있는지는 알 수 없음.")
    if baselines:
        note += ("\n    (가)의 단순 규칙: R0 = 접수 시점 무작위 표시, R1 = 접수 20~01시 호출 전부(점 하나), R2 = 학습 기간(1~8월) (출발구×접수 시각)별\n"
                 "    배차 전 최종 포기율 표(같은 구 쪽으로 수축)가 τ 이상이면 접수 시점에 표시, R3 = 06단계 모형이 접수 시점에 한 번만 계산한\n"
                 "    '다음 30분 내 최종 포기 확률'이 θ 이상이면 표시. 설정 격자는 학습·검증 자료 또는 고정 값으로 정했고 테스트 결과로 고르지 않음.")
    save_fig(fig, "fig_07_policy.png", note=note)
    plt.close(fig)


if __name__ == "__main__":
    main()
