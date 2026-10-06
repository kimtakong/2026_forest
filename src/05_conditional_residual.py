"""05. 조건부 잔여대기: 이미 s분 기다린 사람에게 다음 30분은 어떻게 되나 (CLAUDE.md 5절).

두 가지 '기다리는 중' 집합을 구분한다(2026-10-06 정정).
  [배차 전] 아직 배차·취소되지 않은 콜(floor(T1) >= s). Stage 1 사건 E1_ab(1 배차, 2 최종 포기, 4 재접수 취소).
            정책 해석(전환·안내)은 이 집합으로 한다. -> conditional_predispatch.csv, fig_04
  [승차·취소 전] 아직 승차·취소되지 않은 콜(floor(T_all) >= s). 이미 배차되어 차량을 기다리는 콜이 섞여 있다.
            사건 E_all_ab(1 승차, 2 최종 포기, 4 재접수 취소, 3 기타 실패). -> conditional_residual.csv, 부록 fig_04b
            s가 커질수록 이 집합 안의 '이미 배차된 콜' 비중이 커져 다음 30분 승차 확률이 오른다(배차가 빨라지는 것이 아님).
1분 이산 Aalen-Johansen(survival.py)으로 P(다음 30분 안에 원인 j | s분 시점 대기 중) = (F_j(s+30) - F_j(s)) / S(s).
대상: 즉시호출, 분모 = 콜 기준.
"""
import time

import numpy as np
import pandas as pd

from survival import conditional, conditional_time_to, discrete_cif
from utils import (CALLS, COLOR, DAY_HOURS, HOUR_GROUPS, NIGHT_HOURS, apply_style, clean, draw_figures, fig_title,
                   fig_width, load_counts, save_fig, save_table)

S_LIST = (0, 15, 30, 45, 60, 90)                 # 승차·취소 전 집합(기존)
S_PRE = (0, 5, 10, 15, 20, 30, 45, 60)           # 배차 전 집합
M = 30
CAUSES = [1, 2, 4, 3]
DAY, NIGHT = "주간(10~14시)", "야간(20~01시)"


def time_to_resolution(curve, s, level=0.5):
    c = curve.set_index("t")["S"]
    r = c.loc[s:] / c.loc[s]
    hit = r[r <= 1 - level]
    return float(hit.index[0] - s) if len(hit) else np.nan


def already_dispatched(d, s):
    """승차·취소 전 집합(floor(T_all) >= s) 가운데 s분 전에 이미 배차된 콜의 비율(%)과 집합 크기."""
    w = d[np.floor(d.T_all) >= s]
    return ((w.E1 == 1) & (w.T1 < s)).mean() * 100, len(w)


def residual_table(imm):
    """[승차·취소 전] 집합. 이미 배차되어 차량을 기다리는 콜을 포함한다."""
    groups = {"전체": imm}
    hr = imm.t_request.dt.hour
    groups.update({g: imm[hr.isin(h)] for g, h in HOUR_GROUPS.items()})
    n_all = len(imm)
    rows = []
    for g, d in groups.items():
        bases = {"기본": (d.E_all_ab.values, CAUSES), "민감도: 취소를 한 사건으로": (d.E_all.values, [1, 2, 3])}
        for basis, (E, causes) in bases.items():
            curve = discrete_cif(d.T_all.values, E, causes)
            for s in S_LIST:
                p = conditional(curve, s, M, causes)
                row = {"집단": g, "집단 콜 수": len(d), "즉시호출 대비%": len(d) / n_all * 100, "기준": basis,
                       "s(이미 기다린 분)": s, "s분 시점 승차·취소 전 콜 수": p["위험집합"],
                       "그중 이미 배차된 비율%": already_dispatched(d, s)[0],
                       f"다음 {M}분 승차%": p["P_1"] * 100}
                if basis == "기본":
                    row[f"다음 {M}분 최종 포기%"] = p["P_2"] * 100
                    row[f"다음 {M}분 재접수 취소%"] = p["P_4"] * 100
                else:
                    row[f"다음 {M}분 취소%"] = p["P_2"] * 100
                row[f"다음 {M}분 기타 실패%"] = p["P_3"] * 100
                row[f"{M}분 뒤에도 대기%"] = p["P_대기중"] * 100
                row[f"naive 다음 {M}분 승차%"] = p["naive_P_1"] * 100
                row["승차 50% 도달 남은 시간(분)"] = conditional_time_to(curve, s, 1)
                row["대기 종료 50% 남은 시간(분)"] = time_to_resolution(curve, s)
                rows.append(row)
    t = pd.DataFrame(rows).round(2)
    save_table(t, "conditional_residual.csv")
    return t


def predispatch_table(imm):
    """[배차 전] 집합: s분 시점에 아직 배차·취소되지 않은 콜."""
    hr = imm.t_request.dt.hour
    groups = {"전체": imm, DAY: imm[hr.isin(DAY_HOURS)], NIGHT: imm[hr.isin(NIGHT_HOURS)]}
    rows = []
    for g, d in groups.items():
        curve = discrete_cif(d.T1.values, d.E1_ab.values, [1, 2, 4])
        for s in S_PRE:
            p = conditional(curve, s, M, [1, 2, 4])
            rem = d.loc[(d.E1_ab == 1) & (np.floor(d.T1) >= s), "T1"] - s
            share, n_wait = already_dispatched(d, s)
            rows.append({"집단": g, "집단 콜 수": len(d), "s(이미 기다린 분)": s,
                         "s분 시점 미배차·미취소 콜 수": p["위험집합"],
                         f"다음 {M}분 배차%": p["P_1"] * 100, f"다음 {M}분 최종 포기%": p["P_2"] * 100,
                         f"다음 {M}분 재접수 취소%": p["P_4"] * 100, f"{M}분 뒤에도 미배차%": p["P_대기중"] * 100,
                         "결국 배차된 콜 수": len(rem),
                         "남은 배차 시간 p25(분)": rem.quantile(.25), "남은 배차 시간 p50(분)": rem.quantile(.5),
                         "남은 배차 시간 p75(분)": rem.quantile(.75), "남은 배차 시간 p90(분)": rem.quantile(.9),
                         "s분 시점 승차·취소 전 콜 수": n_wait, "그중 이미 배차된 비율%": share})
    t = pd.DataFrame(rows).round(2)
    save_table(t, "conditional_predispatch.csv")
    show = t[t["집단"] != "전체"][["집단", "s(이미 기다린 분)", "s분 시점 미배차·미취소 콜 수", f"다음 {M}분 배차%",
                                  f"다음 {M}분 최종 포기%", "남은 배차 시간 p75(분)", "남은 배차 시간 p90(분)",
                                  "그중 이미 배차된 비율%"]]
    print(show.round(1).to_string(index=False))
    return t


def _stacked(ax, d, s_list, segs, ann, n_col):
    x = np.arange(len(s_list))
    bottom = np.zeros(len(x))
    GAP = 0.6                                    # 조각 사이 표면색 간격(%p)
    for col, lab, color, tcol in segs:
        v = d[col].values
        ax.bar(x, np.maximum(v - GAP, 0), bottom=bottom + GAP / 2, width=0.62, color=color, label=lab, zorder=2)
        if ann:
            for xi, (b, vv) in enumerate(zip(bottom, v)):
                if tcol is not None and vv >= 6:
                    ax.text(xi, b + vv / 2, f"{vv:.0f}", ha="center", va="center", fontsize=6.5, color=tcol)
        bottom += v
    nr = d[n_col].values
    fmt = lambda v: f"{v / 1e4:.1f}만" if v >= 1e4 else f"{v:,}"
    labels = [f"{s}분\n{fmt(v)}" for s, v in zip(s_list, nr)] if ann else [f"{s}" for s in s_list]
    ax.set_xticks(x, labels, fontsize=6.6 if ann else None)
    ax.set_ylim(0, 100)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)


def fig_predispatch(t):
    """fig_04: 아직 배차되지 않은 콜의 s분 시점 다음 30분 결과."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    ann = not clean()
    cnt = load_counts()
    segs = [(f"다음 {M}분 배차%", "배차", COLOR["board"], "white"),
            (f"다음 {M}분 최종 포기%", "최종 포기", COLOR["cancel"], "white"),
            (f"다음 {M}분 재접수 취소%", "재접수 취소", "#f5b494", None),
            (f"{M}분 뒤에도 미배차%", "계속 미배차", "#e1e0d9", None)]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width(), 3.9 if ann else 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    if ann:
        fig.subplots_adjust(left=0.095, right=0.985, top=0.73, bottom=0.29)
    else:
        fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.16)
    for i, (ax, g, n) in enumerate(zip(axes, [DAY, NIGHT], [cnt["imm_day_n"], cnt["imm_night_n"]])):
        d = t[t["집단"] == g].set_index("s(이미 기다린 분)").loc[list(S_PRE)]
        _stacked(ax, d, S_PRE, segs, ann, "s분 시점 미배차·미취소 콜 수")
        name = ["(가) 주간 10~14시", "(나) 야간 20~01시"][i]
        ax.set_title(f"{name}  (즉시호출 {n:,}건)" if ann else name, loc="left", pad=6 if ann else 4,
                     fontsize=8.5 if ann else None)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel(f"다음 {M}분 동안 일어나는 일")
    xl = "아직 배차되지 않은 채 기다린 시간 s(분)" + ("  (아래 숫자 = 그 시점에 아직 미배차인 콜 수)" if ann else "")
    if ann:
        fig.text(0.54, 0.165, xl, ha="center", fontsize=7.5, color=COLOR["ink2"])
    else:
        fig.supxlabel(xl, y=0.02, fontsize=None, color=COLOR["ink2"])
    h, l = axes[0].get_legend_handles_labels()
    if ann:
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.885), ncol=4, fontsize=7, handlelength=1.2,
                   columnspacing=1.4)
    else:
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.11, 1.0), ncol=4, handlelength=1.2, columnspacing=1.2)
    dd = t[t["집단"] == DAY].set_index("s(이미 기다린 분)")
    nn = t[t["집단"] == NIGHT].set_index("s(이미 기다린 분)")
    col = f"다음 {M}분 배차%"
    fig_title(fig, "아직 배차되지 않은 콜의 다음 30분 배차 확률은 처음 30분 동안 낮아진다(그 뒤 일부 회복)",
              f"다음 30분 배차 확률(0분→30분 기다린 뒤): 주간 {dd.loc[0, col]:.0f}%→{dd.loc[30, col]:.0f}%, "
              f"야간 {nn.loc[0, col]:.0f}%→{nn.loc[30, col]:.0f}%, 60분 시점 주간 {dd.loc[60, col]:.0f}%·야간 {nn.loc[60, col]:.0f}%. 결국 배차된 콜의 남은 배차 시간 90% 분위: "
              f"주간 {dd.loc[0, '남은 배차 시간 p90(분)']:.0f}분, 야간 {nn.loc[0, '남은 배차 시간 p90(분)']:.0f}분(접수 직후)",
              y_sub=0.915)
    note = ("주: 즉시호출(콜 기준). s분 시점에 아직 배차·취소되지 않은 콜이 그 뒤 30분 안에 어떻게 되는지(1분 이산 Aalen-Johansen 조건부 누적확률).\n"
            "    막대 안 숫자 = %. 재접수 취소 = 취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수. "
            "접수 시각 기준: 주간 = 10:00~14:59, 야간 = 20:00~01:59.")
    save_fig(fig, "fig_04_conditional_residual.png", note=note)
    plt.close(fig)


def fig_overall(t):
    """부록 fig_04b: 승차·취소 전 콜(이미 배차되어 차량을 기다리는 콜 포함)의 다음 30분 결과."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    ann = not clean()
    cnt = load_counts()
    segs = [(f"다음 {M}분 승차%", "승차", COLOR["board"], "white"),
            (f"다음 {M}분 최종 포기%", "최종 포기", COLOR["cancel"], "white"),
            (f"다음 {M}분 재접수 취소%", "재접수 취소", "#f5b494", None),
            (f"{M}분 뒤에도 대기%", "계속 대기", "#e1e0d9", None)]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width(), 3.9 if ann else 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    if ann:
        fig.subplots_adjust(left=0.095, right=0.985, top=0.73, bottom=0.29)
    else:
        fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.16)
    for i, (ax, g, n) in enumerate(zip(axes, [DAY, NIGHT], [cnt["imm_day_n"], cnt["imm_night_n"]])):
        d = t[(t["집단"] == g) & (t["기준"] == "기본")].set_index("s(이미 기다린 분)").loc[list(S_LIST)]
        _stacked(ax, d, S_LIST, segs, ann, "s분 시점 승차·취소 전 콜 수")
        name = ["(가) 주간 10~14시", "(나) 야간 20~01시"][i]
        ax.set_title(f"{name}  (즉시호출 {n:,}건)" if ann else name, loc="left", pad=6 if ann else 4,
                     fontsize=8.5 if ann else None)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel(f"다음 {M}분 동안 일어나는 일")
    xl = "승차·취소 없이 기다린 시간 s(분)" + ("  (아래 숫자 = 그 시점에 아직 승차·취소 전인 콜 수)" if ann else "")
    if ann:
        fig.text(0.54, 0.165, xl, ha="center", fontsize=7.5, color=COLOR["ink2"])
    else:
        fig.supxlabel(xl, y=0.02, fontsize=None, color=COLOR["ink2"])
    h, l = axes[0].get_legend_handles_labels()
    if ann:
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.885), ncol=4, fontsize=7, handlelength=1.2,
                   columnspacing=1.4)
    else:
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.11, 1.0), ncol=4, handlelength=1.2, columnspacing=1.2)
    dd = t[(t["집단"] == DAY) & (t["기준"] == "기본")].set_index("s(이미 기다린 분)")
    nn = t[(t["집단"] == NIGHT) & (t["기준"] == "기본")].set_index("s(이미 기다린 분)")
    fig_title(fig, "승차·취소 전 콜 기준: 오래 기다린 콜일수록 이미 배차된 콜이 많아 곧 탈 확률이 오른다",
              f"30분 시점 승차·취소 전 콜 중 이미 배차된 비율: 주간 {dd.loc[30, '그중 이미 배차된 비율%']:.0f}%, "
              f"야간 {nn.loc[30, '그중 이미 배차된 비율%']:.0f}%. 배차 전 단계만 보면 fig_04", y_sub=0.915)
    note = ("주: 즉시호출(콜 기준). s분 시점에 아직 승차·취소 전인 콜(이미 배차되어 차량을 기다리는 콜 포함)이 그 뒤 30분 안에 어떻게 되는지\n"
            "    (1분 이산 Aalen-Johansen 조건부 누적확률). 막대 안 숫자 = %. 기타 실패(<0.1%)는 생략. "
            "접수 시각 기준: 주간 = 10:00~14:59, 야간 = 20:00~01:59.")
    save_fig(fig, "fig_04b_conditional_overall.png", note=note)
    plt.close(fig)


def main():
    t0 = time.time()
    df = pd.read_parquet(CALLS, columns=["call_type", "T1", "E1", "E1_ab", "T_all", "E_all", "E_all_ab", "t_request"])
    imm = df[df.call_type == "immediate"]
    c = load_counts()
    hr = imm.t_request.dt.hour
    assert len(imm) == c["imm_n"] and hr.isin(NIGHT_HOURS).sum() == c["imm_night_n"] and hr.isin(DAY_HOURS).sum() == c["imm_day_n"]
    t_all = residual_table(imm)
    t_pre = predispatch_table(imm)
    draw_figures(fig_predispatch, t_pre)
    draw_figures(fig_overall, t_all)
    print(f"[05_conditional_residual] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
