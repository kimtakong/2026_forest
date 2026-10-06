"""04. 선행연구 방식(naive KM) vs 경쟁위험(Aalen-Johansen 누적발생함수) (CLAUDE.md 5절).

대상: 즉시호출, Overall(접수 -> 승차). 시간 = T_all(분), 1분 이산 구간 [k, k+1).
  naive KM : 승차만 사건, 취소·기타 실패는 중도절단(손종훈 2022 방식). 승차 누적확률 = 1 - S(t).
  경쟁위험  : 승차 / 최종 포기 / 재접수 취소 / 기타 실패를 서로 경쟁하는 사건으로 둔 Aalen-Johansen CIF.
              (사건 코드 E_all_ab: 1 승차, 2 최종 포기, 3 기타 실패, 4 재접수 취소, 0 중도절단)
이산시간 추정식은 survival.py.
분모 = 콜 기준(본문). 에피소드 기준(재접수 취소 콜 제외)은 민감도.
민감도: (a) 전체 취소를 한 사건으로(E_all)  (b) 기타 실패 콜을 빼고 계산
출력: outputs/tables/naive_vs_cif_at_t.csv (30·60·90분), cif_curves.csv (0~180분 곡선)
      outputs/figures/fig_03_naive_vs_cif.png (전체 / 야간 20~01시)
"""
import time

import numpy as np
import pandas as pd

from survival import discrete_cif
from utils import CALLS, COLOR, FIG_WIDTH_IN, NIGHT_HOURS, apply_style, load_counts, save_fig, save_table

T_REPORT = (30, 60, 90)
CAUSES = {1: "승차", 2: "최종 포기", 4: "재접수 취소", 3: "기타 실패"}


def run_group(d, label):
    """기본(콜 기준, E_all_ab)과 민감도 세 가지를 계산한다."""
    res = {}
    res["기본"] = discrete_cif(d.T_all.values, d.E_all_ab.values, [1, 2, 4, 3])
    res["민감도: 취소를 한 사건으로"] = discrete_cif(d.T_all.values, d.E_all.values, [1, 2, 3])
    keep = d.E_all_ab != 3
    res["민감도: 기타 실패 제외"] = discrete_cif(d.T_all.values[keep], d.E_all_ab.values[keep], [1, 2, 4])
    keep = d.E_all_ab != 4   # 재접수 취소 콜을 빼면 '이용 건(에피소드)' 기준. 03단계 '최종 포기 기준'과 같은 분모
    res["민감도: 재접수 취소 콜 제외(에피소드)"] = discrete_cif(d.T_all.values[keep], d.E_all_ab.values[keep], [1, 2, 3])
    for k, v in res.items():
        v.insert(0, "기준", k)
        v.insert(0, "집단", label)
    return res


def at_t_table(results, n_groups):
    n_all = n_groups["전체 시간대"]
    rows = []
    for (grp, basis), c in results.items():
        for t in T_REPORT:
            r = c.loc[c.t == t].iloc[0]
            row = {"집단": grp, "집단 콜 수": n_groups[grp], "즉시호출 대비%": n_groups[grp] / n_all * 100,
                   "기준": basis, "t(분)": t, "위험집합": int(r["위험집합"]),
                   "naive 승차 누적확률%": r["naive_승차"] * 100, "AJ 승차 CIF%": r["F_1"] * 100}
            row["과대평가(naive-AJ)%p"] = row["naive 승차 누적확률%"] - row["AJ 승차 CIF%"]
            if basis == "민감도: 취소를 한 사건으로":
                row["AJ 취소 CIF%"] = r["F_2"] * 100
            else:
                row["AJ 최종 포기 CIF%"] = r["F_2"] * 100
                if "F_4" in c:
                    row["AJ 재접수 취소 CIF%"] = r["F_4"] * 100
            if "F_3" in c:
                row["AJ 기타 실패 CIF%"] = r["F_3"] * 100
            row["아직 대기 중%"] = r["S"] * 100
            rows.append(row)
    t = pd.DataFrame(rows).round(2)
    save_table(t, "naive_vs_cif_at_t.csv")
    print(t[t["기준"] == "기본"].to_string(index=False))
    print(t[(t["기준"] != "기본") & (t["t(분)"] == 60)].to_string(index=False))
    return t


def fig_naive_vs_cif(results, tab, n_groups):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.70, bottom=0.27)
    blue, orange, orange_lt, gray = COLOR["board"], COLOR["cancel"], "#f5b494", COLOR["ink2"]
    T_SHOW = 120
    for ax, grp in zip(axes, ["전체 시간대", "야간(20~01시)"]):
        c = results[(grp, "기본")]
        c = c[c.t <= T_SHOW]
        ax.fill_between(c.t, c.F_1 * 100, c.naive_승차 * 100, color=gray, alpha=0.12, lw=0)
        ax.plot(c.t, c.naive_승차 * 100, color=gray, lw=1.8, label="선행연구 방식: 취소를 중도절단(naive KM)")
        ax.plot(c.t, c.F_1 * 100, color=blue, lw=2, label="경쟁위험: 승차")
        ax.plot(c.t, c.F_2 * 100, color=orange, lw=2, label="경쟁위험: 최종 포기")
        ax.plot(c.t, c.F_4 * 100, color=orange_lt, lw=1.6, label="경쟁위험: 재접수 취소")
        r = tab[(tab["집단"] == grp) & (tab["기준"] == "기본") & (tab["t(분)"] == 60)].iloc[0]
        ax.axvline(60, color=COLOR["axis"], lw=0.8, zorder=0)
        ax.plot([60], [r["naive 승차 누적확률%"]], "o", ms=4.5, color=gray, mec=COLOR["surface"], mew=1)
        ax.plot([60], [r["AJ 승차 CIF%"]], "o", ms=4.5, color=blue, mec=COLOR["surface"], mew=1)
        ax.plot([60], [r["AJ 최종 포기 CIF%"]], "o", ms=4.5, color=orange, mec=COLOR["surface"], mew=1)
        ax.text(63, 40,
                f"60분: {r['naive 승차 누적확률%']:.1f}% vs {r['AJ 승차 CIF%']:.1f}%\n"
                f"→ {r['과대평가(naive-AJ)%p']:.1f}%p 과대평가", fontsize=7, color=COLOR["ink"], va="center")
        ax.text(62, r["AJ 최종 포기 CIF%"] + 4, f"최종 포기 {r['AJ 최종 포기 CIF%']:.1f}%",
                fontsize=7, color=COLOR["ink"], va="bottom")
        ax.set_xlim(0, T_SHOW)
        ax.set_xticks(range(0, T_SHOW + 1, 30))
        ax.set_ylim(0, 100)
        ax.set_xlabel("접수 후 경과 시간(분)")
        ax.grid(axis="y"); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.set_title(f"{grp}  (n = {n_groups[grp]:,})", loc="left", fontsize=8.5, pad=6)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel("누적확률")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.085, 0.905), ncol=2, fontsize=7, handlelength=1.8,
               columnspacing=1.5)
    cnt = load_counts()
    fig.text(0.01, 0.975, "취소를 '관찰 끊김'으로 처리하면 60분 안에 탈 확률을 부풀린다",
             fontsize=10, fontweight="bold", color=COLOR["ink"], va="top")
    note = ("주: 즉시호출(접수→승차). naive KM = 취소를 중도절단한 1-S(t)(선행연구 방식). 경쟁위험 = 1분 이산 Aalen-Johansen 누적발생함수.\n"
            f"    최종 포기 {cnt['imm_final']:,}건, 재접수 취소 {cnt['imm_recall']:,}건(취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수).\n"
            f"    기타 실패 {cnt['imm_other_fail']:,}건은 그림에서 생략(표에 수록). 회색 음영 = 선행연구 방식이 부풀린 부분.")
    save_fig(fig, "fig_03_naive_vs_cif.png", note=note)
    plt.close(fig)


def main():
    t0 = time.time()
    df = pd.read_parquet(CALLS, columns=["call_type", "T_all", "E_all", "E_all_ab", "t_request"])
    imm = df[df.call_type == "immediate"]
    assert len(imm) == load_counts()["imm_n"]
    groups = {"전체 시간대": imm, "야간(20~01시)": imm[imm.t_request.dt.hour.isin(NIGHT_HOURS)]}
    results = {}
    for g, d in groups.items():
        for basis, c in run_group(d, g).items():
            results[(g, basis)] = c
    curves = pd.concat(results.values(), ignore_index=True)
    save_table(curves.round(6), "cif_curves.csv")
    n_groups = {g: len(d) for g, d in groups.items()}
    tab = at_t_table(results, n_groups)
    fig_naive_vs_cif(results, tab, n_groups)
    print(f"[04_cif_vs_naive] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
