"""05. 조건부 잔여대기: 이미 s분 기다린 사람에게 다음 30분은 어떻게 되나 (CLAUDE.md 5절).

대상: 즉시호출, Overall(접수 -> 승차), 분모 = 콜 기준. 사건 E_all_ab(1 승차, 2 최종 포기, 4 재접수 취소, 3 기타 실패).
s분 시점에 아직 대기 중(구간 s 시작까지 사건 없음)인 콜에 대해, 1분 이산 Aalen-Johansen 곡선으로
  P(다음 m분 안에 원인 j) = (F_j(s+m) - F_j(s)) / S(s),   m = 30
  승차 50% 도달 남은 시간 = 조건부 승차 누적확률이 50%에 처음 닿는 시간(닿지 않으면 빈칸)
  대기 종료 50% 남은 시간 = 어떤 사건이든 끝날 확률이 50%에 닿는 시간
비교: naive KM(취소 = 중도절단)의 조건부 승차 확률. 민감도: 취소를 한 사건으로(E_all).
출력: outputs/tables/conditional_residual.csv, outputs/figures/fig_04_conditional_residual.png
"""
import time

import numpy as np
import pandas as pd

from survival import conditional, conditional_time_to, discrete_cif
from utils import (CALLS, COLOR, DAY_HOURS, FIG_WIDTH_IN, HOUR_GROUPS, NIGHT_HOURS, apply_style, load_counts,
                   save_fig, save_table)

S_LIST = (0, 15, 30, 45, 60, 90)
M = 30
CAUSES = [1, 2, 4, 3]


def time_to_resolution(curve, s, level=0.5):
    c = curve.set_index("t")["S"]
    r = c.loc[s:] / c.loc[s]
    hit = r[r <= 1 - level]
    return float(hit.index[0] - s) if len(hit) else np.nan


def residual_table(imm):
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
                       "s(이미 기다린 분)": s, "s분 시점 대기 중 콜 수": p["위험집합"],
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
    show = t[(t["기준"] == "기본") & t["집단"].isin(["전체", "주간(10~14시)", "야간(20~01시)"])]
    print(show.drop(columns=["기준", "집단 콜 수", "즉시호출 대비%"]).to_string(index=False))
    return t


def fig_residual(t):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    cnt = load_counts()
    segs = [(f"다음 {M}분 승차%", "승차", COLOR["board"], "white"),
            (f"다음 {M}분 최종 포기%", "최종 포기", COLOR["cancel"], "white"),
            (f"다음 {M}분 재접수 취소%", "재접수 취소", "#f5b494", COLOR["ink"]),
            (f"{M}분 뒤에도 대기%", f"{M}분 뒤에도 대기", "#e1e0d9", COLOR["ink"])]
    panels = [("주간(10~14시)", "주간 10~14시", cnt["imm_day_n"]), ("야간(20~01시)", "야간 20~01시", cnt["imm_night_n"])]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.9), sharey=True, gridspec_kw={"wspace": 0.06})
    fig.subplots_adjust(left=0.095, right=0.985, top=0.73, bottom=0.29)
    x = np.arange(len(S_LIST))
    for ax, (g, title, n) in zip(axes, panels):
        d = t[(t["집단"] == g) & (t["기준"] == "기본")].set_index("s(이미 기다린 분)").loc[list(S_LIST)]
        bottom = np.zeros(len(x))
        GAP = 0.6                                    # 조각 사이 표면색 간격(%p)
        for col, lab, color, tcol in segs:
            v = d[col].values
            ax.bar(x, np.maximum(v - GAP, 0), bottom=bottom + GAP / 2, width=0.62, color=color, label=lab, zorder=2)
            for xi, (b, vv) in enumerate(zip(bottom, v)):
                if col != f"{M}분 뒤에도 대기%" and vv >= 6:
                    ax.text(xi, b + vv / 2, f"{vv:.0f}", ha="center", va="center", fontsize=6.8, color=tcol)
            bottom += v
        nr = d["s분 시점 대기 중 콜 수"].values
        fmt = lambda v: f"{v / 1e4:.1f}만" if v >= 1e4 else f"{v:,}"
        ax.set_xticks(x, [f"{s}분\n{fmt(v)}" for s, v in zip(S_LIST, nr)], fontsize=6.8)
        ax.set_ylim(0, 100)
        ax.tick_params(length=0)
        ax.set_title(f"{title}  (즉시호출 {n:,}건)", loc="left", fontsize=8.5, pad=6)
        ax.spines["bottom"].set_visible(False)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel(f"다음 {M}분 동안 일어나는 일")
    fig.text(0.54, 0.165, "이미 기다린 시간 s  (아래 숫자 = 그 시점에 아직 대기 중인 콜 수)", ha="center", fontsize=7.5, color=COLOR["ink2"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.07, 0.885), ncol=4, fontsize=7, handlelength=1.2,
               columnspacing=1.4)
    day = t[(t["집단"] == "주간(10~14시)") & (t["기준"] == "기본")].set_index("s(이미 기다린 분)")
    night = t[(t["집단"] == "야간(20~01시)") & (t["기준"] == "기본")].set_index("s(이미 기다린 분)")
    fig.text(0.01, 0.975, f"이미 30분 기다린 사람이 다음 30분 안에 탈 확률: 주간 {day.loc[30, f'다음 {M}분 승차%']:.0f}%, "
                          f"야간 {night.loc[30, f'다음 {M}분 승차%']:.0f}%",
             fontsize=10, fontweight="bold", color=COLOR["ink"], va="top")
    note = ("주: 즉시호출(콜 기준). s분 시점에 아직 배차·승차·취소 없이 기다리던 콜이 그 뒤 30분 안에 어떻게 끝나는지(1분 이산 Aalen-Johansen 조건부 누적확률).\n"
            "    막대 안 숫자 = %. 재접수 취소 = 취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수. "
            "기타 실패(<0.1%)는 생략.\n    접수 시각 기준: 주간 = 10:00~14:59, 야간 = 20:00~01:59.")
    save_fig(fig, "fig_04_conditional_residual.png", note=note)
    plt.close(fig)


def main():
    t0 = time.time()
    df = pd.read_parquet(CALLS, columns=["call_type", "T_all", "E_all", "E_all_ab", "t_request"])
    imm = df[df.call_type == "immediate"]
    c = load_counts()
    hr = imm.t_request.dt.hour
    assert len(imm) == c["imm_n"] and hr.isin(NIGHT_HOURS).sum() == c["imm_night_n"] and hr.isin(DAY_HOURS).sum() == c["imm_day_n"]
    t = residual_table(imm)
    fig_residual(t)
    print(f"[05_conditional_residual] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
