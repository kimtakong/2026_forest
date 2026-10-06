"""03. 공식 시간대별 대기시간 vs 탑승내역으로 다시 계산한 값 (CLAUDE.md 5절).

공식값 산식(00단계에서 역추정, utils.official_formula): 원본 전체(제외 전)의 승차한 콜(즉시+사전)에서
  (승차 - 예정), 각 시각을 분 단위로 절사, 음수는 0, 예정 시간대별 평균을 내림. 8,718칸 중 93.8%가 분 단위까지 일치.
비교 지표(즉시호출, 접수 시간대 기준)
  (a) 승차한 콜의 접수->승차 평균      (b) 같은 값의 중앙값
  (c) 60분 안에 승차한 비율
      - 승차자만: 승차한 콜 중 60분 안에 탄 비율 (공식 지표가 보여 주는 세계)
      - 전체(최종 포기 기준): 재접수 취소를 뺀 즉시호출 중 60분 안에 탄 비율 [기본]
      - 전체(모든 취소 포함): 즉시호출 전체 중 60분 안에 탄 비율 [민감도]
  60분 전 중도절단은 0.01% 미만이라 단순 비율 = Aalen-Johansen 누적발생함수 추정치다.
출력: outputs/tables/official_compare.csv (일자 x 시간대), official_by_hour.csv, official_compare_summary.csv
      outputs/figures/fig_02_official_vs_actual.png
"""
import time

import numpy as np
import pandas as pd

from utils import (CALLS, COLOR, FIG_WIDTH_IN, add_status, apply_style, load_counts, load_official_wait, load_raw,
                   official_formula, save_fig, save_table)

H = 60   # 비교 시점(분)


def official_reproduction():
    """공식 산식 재현값(일자 x 예정 시간대). 공단과 같은 모집단(원본 전체)으로 계산한다."""
    rep = official_formula(add_status(load_raw()))
    return np.floor(rep).rename("reproduced"), rep.rename("reproduced_raw")


def cell_metrics(imm):
    """즉시호출을 접수 일자 x 시간대 칸으로 묶은 지표."""
    k = imm.t_request.dt.floor("h")
    board = imm.E_all == 1
    b60 = board & (imm.T_all <= H)
    final_base = imm.E_all_ab != 4                  # 재접수 취소를 뺀 모집단
    g = pd.DataFrame({"k": k, "board": board, "b60": b60, "final_base": final_base,
                      "cancel": imm.E_all == 2, "final": imm.E_all_ab == 2,
                      "wait_b": imm.T_all.where(board)})
    out = g.groupby("k").agg(
        즉시호출=("board", "size"), 승차=("board", "sum"), 취소=("cancel", "sum"), 최종포기=("final", "sum"),
        승차_60분내=("b60", "sum"), 최종포기기준_모집단=("final_base", "sum"),
        승차자_평균대기=("wait_b", "mean"), 승차자_중앙대기=("wait_b", "median"))
    out["P60_승차자만%"] = out["승차_60분내"] / out["승차"] * 100
    out["P60_전체_최종포기기준%"] = out["승차_60분내"] / out["최종포기기준_모집단"] * 100
    out["P60_전체_모든취소%"] = out["승차_60분내"] / out["즉시호출"] * 100
    return out


def build_tables(df):
    off = load_official_wait()
    rep, rep_raw = official_reproduction()
    imm = df[df.call_type == "immediate"]
    cm = cell_metrics(imm)
    t = pd.concat([off, rep, rep_raw, cm], axis=1)
    t = t[t["official"].notna()]
    t.index.name = "일자시간"
    t["재현일치"] = t["official"] == t["reproduced"]
    t.insert(0, "hour", t.index.hour)
    save_table(t.round(3).reset_index(), "official_compare.csv")

    # 상관·오차 요약: 공식값이 각 지표와 얼마나 같이 움직이는가
    rows = []
    for col in ["reproduced", "승차자_평균대기", "승차자_중앙대기", "P60_승차자만%", "P60_전체_최종포기기준%", "P60_전체_모든취소%"]:
        j = t[["official", col]].dropna()
        rows.append({"지표": col, "칸 수": len(j), "공식값과 상관": round(j.corr().iloc[0, 1], 4),
                     "평균": round(j[col].mean(), 2)})
    s = pd.DataFrame(rows)
    s.loc[len(s)] = {"지표": "재현 일치율%(분 단위)", "칸 수": int(t["reproduced"].notna().sum()),
                     "공식값과 상관": np.nan, "평균": round(t["재현일치"].mean() * 100, 2)}
    save_table(s, "official_compare_summary.csv")
    print(s.to_string(index=False))

    # 시간대별(그림용). 공식값·재현값은 일자 칸의 단순 평균, 확률은 콜 가중.
    h = t.groupby("hour").agg(공식값=("official", "mean"), 재현값=("reproduced", "mean"),
                              즉시호출=("즉시호출", "sum"), 승차=("승차", "sum"), 승차_60분내=("승차_60분내", "sum"),
                              최종포기기준_모집단=("최종포기기준_모집단", "sum"), 취소=("취소", "sum"), 최종포기=("최종포기", "sum"))
    ih = imm.groupby(imm.t_request.dt.hour)
    h["승차자_평균대기"] = ih.apply(lambda x: x.T_all[x.E_all == 1].mean())
    h["승차자_중앙대기"] = ih.apply(lambda x: x.T_all[x.E_all == 1].median())
    h["P60_승차자만%"] = h["승차_60분내"] / h["승차"] * 100
    h["P60_전체_최종포기기준%"] = h["승차_60분내"] / h["최종포기기준_모집단"] * 100
    h["P60_전체_모든취소%"] = h["승차_60분내"] / h["즉시호출"] * 100
    h["격차_최종포기기준%p"] = h["P60_승차자만%"] - h["P60_전체_최종포기기준%"]
    save_table(h.round(2).reset_index(), "official_by_hour.csv")
    print(h[["공식값", "재현값", "승차자_평균대기", "P60_승차자만%", "P60_전체_최종포기기준%", "P60_전체_모든취소%"]].round(1).to_string())

    # 전체 요약 숫자(즉시호출)
    n = len(imm)
    b60 = int(((imm.E_all == 1) & (imm.T_all <= H)).sum())
    base = int((imm.E_all_ab != 4).sum())
    overall = {"승차자만": b60 / (imm.E_all == 1).sum() * 100, "최종포기기준": b60 / base * 100, "모든취소": b60 / n * 100}
    c = load_counts()
    assert base == c["imm_episodes"], "기준 건수표와 다르다"
    print("  전체 60분 내 승차 비율%:", {k: round(v, 1) for k, v in overall.items()})
    return t, h, overall


def fig_official_vs_actual_v1(t, h, overall):
    """초기 2패널 버전(보관용)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    c = load_counts()
    match = t["재현일치"].mean() * 100
    hrs = h.index.values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.4), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.28)
    for ax in (a1, a2):
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("접수 시간대(시)")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    # (가) 공식 평균 대기시간과 재현값(공식 기준: 예정 시간대)
    a1.set_xlabel("예정 시간대(시)")
    ink2 = COLOR["ink2"]
    a1.plot(hrs, h["공식값"], color=ink2, lw=2, solid_capstyle="round", zorder=2)
    a1.scatter(hrs, h["재현값"], s=16, color=COLOR["board"], edgecolor=COLOR["surface"], linewidth=1, zorder=3)
    a1.set_ylim(0, max(h["공식값"].max(), h["재현값"].max()) * 1.15)
    a1.set_ylabel("평균 대기시간(분)")
    a1.set_title("(가) 공식 평균 대기시간(승차한 사람만)", loc="left", fontsize=8.5, pad=8)
    i = int(np.argmax(h["공식값"].values))
    a1.annotate("공식값(선)", xy=(hrs[i], h["공식값"].iloc[i]), xytext=(hrs[i] - 9, h["공식값"].iloc[i] + 3),
                fontsize=7, color=COLOR["ink"], arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))
    a1.annotate("탑승내역 재현값(점)", xy=(hrs[12], h["재현값"].iloc[12]), xytext=(9.5, h["재현값"].iloc[12] - 14),
                fontsize=7, color=COLOR["ink"], arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))

    # (나) 60분 안에 승차한 비율
    pct = FuncFormatter(lambda v, _: f"{v:.0f}%")
    # 세 선이 오른쪽 끝에서 모이므로 끝 레이블 대신 범례(아래 빈 곳)를 쓴다
    lines = [("P60_승차자만%", ink2, 2.0, "승차자만(공식 지표의 시야)"),
             ("P60_전체_최종포기기준%", COLOR["cancel"], 2.0, "즉시호출 전체(최종 포기 포함)"),
             ("P60_전체_모든취소%", "#f5b494", 1.4, "모든 취소 포함(민감도)")]
    for col, color, lw, lab in lines:
        a2.plot(hrs, h[col], color=color, lw=lw, solid_capstyle="round", label=lab)
    a2.legend(loc="lower left", fontsize=6.8, handlelength=1.6, borderaxespad=0.3)
    a2.set_ylim(0, 105)
    a2.yaxis.set_major_formatter(pct)
    a2.set_ylabel(f"접수 후 {H}분 안에 승차한 비율")
    a2.set_title(f"(나) {H}분 안에 승차한 비율", loc="left", fontsize=8.5, pad=8)
    j = int(np.argmax(h["격차_최종포기기준%p"].values))
    a2.annotate(f"{hrs[j]}시: {h['P60_승차자만%'].iloc[j]:.0f}% → {h['P60_전체_최종포기기준%'].iloc[j]:.0f}%",
                xy=(hrs[j], h["P60_전체_최종포기기준%"].iloc[j]), xytext=(hrs[j] - 6, 30), fontsize=7,
                color=COLOR["ink"], arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))

    fig.text(0.01, 0.97, f"공식 통계는 승차한 사람만 센다: 즉시호출 취소 {c['imm_cancel'] / 1e4:.1f}만 건"
                         f"(최종 포기 {c['imm_final'] / 1e4:.1f}만 건)이 빠져 있다",
             fontsize=10, fontweight="bold", color=COLOR["ink"], va="top")
    fig.text(0.01, 0.895, f"즉시호출 전체로 보면 {H}분 안에 탄 비율은 {overall['최종포기기준']:.1f}%"
                          f"(승차자만 보면 {overall['승차자만']:.1f}%)",
             fontsize=7.8, color=COLOR["ink2"], va="top")
    note = ("주: 공식값 = 서울시설공단 '시간대별 대기시간평균'. 탑승내역으로 공식 산식(승차한 콜의 승차-예정 평균, 분 단위 절사, 음수 0,\n"
            f"    예정 시간대별)을 재현하면 일자×시간대 {int(t['reproduced'].notna().sum()):,}칸 중 {match:.1f}%가 분 단위까지 일치. "
            "(가)는 일자 칸의 단순 평균.\n"
            "    (나)는 즉시호출. 최종 포기 = 취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수가 없는 취소.")
    save_fig(fig, "fig_02_official_vs_actual_v1.png", note=note)
    plt.close(fig)


def fig_official_vs_actual(t, h, overall):
    """보고서용: 60분 안에 탄 비율 격차(큰 패널) + 공식 평균 대기시간(위쪽 얇은 띠 패널, x축 공유).
    두 지표는 단위가 달라 한 축에 겹치지 않고(이중 축 금지), 같은 시간축의 두 패널로 나눈다."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    apply_style()
    c = load_counts()
    hrs = h.index.values
    a_col, b_col = "P60_승차자만%", "P60_전체_최종포기기준%"
    gap = h[a_col] - h[b_col]
    night = [20, 21, 22, 23, 0, 1]
    day = [11, 12, 13]
    g_night = (gap.loc[night].min(), gap.loc[night].max())
    g_day = gap.loc[day].mean()

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 4.1), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 3.3], "hspace": 0.12})
    fig.subplots_adjust(left=0.085, right=0.83, top=0.83, bottom=0.235)
    shade = "#f0efec"
    for ax in (a0, a1):
        ax.axvspan(19.5, 23.5, color=shade, lw=0, zorder=0)
        ax.axvspan(-0.5, 1.5, color=shade, lw=0, zorder=0)
        ax.set_xlim(-0.5, 23.5)
        ax.tick_params(length=0)

    # 위: 공식 평균 대기시간(분)
    a0.bar(hrs, h["공식값"], width=0.62, color=COLOR["axis"], zorder=2)
    a0.set_ylim(0, 70)
    a0.set_yticks([0, 30, 60])
    a0.grid(axis="y"); a0.set_axisbelow(True)
    a0.text(-0.3, 66, "공식 평균 대기시간(분, 승차한 사람만)", fontsize=7.3, color=COLOR["ink2"], va="top")
    for hh in (int(h["공식값"].idxmin()), int(h["공식값"].idxmax())):
        a0.text(hh, h.loc[hh, "공식값"] + 2, f"{h.loc[hh, '공식값']:.0f}", ha="center", va="bottom", fontsize=6.8,
                color=COLOR["ink2"])

    # 아래: 60분 안에 탄 비율과 그 격차
    a1.fill_between(hrs, h[b_col], h[a_col], color=COLOR["cancel"], alpha=0.14, lw=0, zorder=1)
    a1.plot(hrs, h[a_col], color=COLOR["ink2"], lw=2, solid_capstyle="round", zorder=3)
    a1.plot(hrs, h[b_col], color=COLOR["cancel"], lw=2, solid_capstyle="round", zorder=3)
    for col, lab, color in [(a_col, "승차자만", COLOR["ink2"]), (b_col, "즉시호출 전체", COLOR["cancel"])]:
        y = h[col].iloc[-1]
        a1.plot([hrs[-1]], [y], "o", ms=4.5, color=color, mec=COLOR["surface"], mew=1, zorder=4)
        a1.text(23.9, y, f"{lab}  {y:.0f}%", fontsize=7, color=COLOR["ink"], va="center", ha="left",
                linespacing=1.25, clip_on=False)
    a1.set_ylim(0, 105)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a1.set_ylabel(f"접수 후 {H}분 안에 승차한 비율")
    a1.grid(axis="y"); a1.set_axisbelow(True)
    a1.set_xticks(range(0, 24, 3))
    a1.set_xlabel("접수 시간대(시)")
    a1.text(21.5, 5, f"공식 통계가\n가리는 구간\n격차 {g_night[0]:.0f}~{g_night[1]:.0f}%p", ha="center", va="bottom",
            fontsize=7, color=COLOR["ink"], linespacing=1.3)
    a1.text(11.4, 79, f"11~13시 격차 {g_day:.0f}%p", ha="center", va="top", fontsize=7, color=COLOR["ink2"])
    j = 21
    a1.annotate(f"{j}시: {h.loc[j, a_col]:.0f}% → {h.loc[j, b_col]:.0f}%", xy=(j, h.loc[j, b_col]),
                xytext=(14.2, 30), fontsize=7, color=COLOR["ink"],
                arrowprops=dict(arrowstyle="-", color=COLOR["muted"], lw=0.6))

    fig.text(0.01, 0.975, "공식 평균이 가장 많이 가리는 사람은 밤에 부르는 이용자다",
             fontsize=10, fontweight="bold", color=COLOR["ink"], va="top")
    fig.text(0.01, 0.91, f"{H}분 안에 탄 비율: 승차자만 보면 {overall['승차자만']:.1f}%, 취소한 사람까지 넣으면 "
                         f"{overall['최종포기기준']:.1f}%. 음영 = 공식 지표에 보이지 않는 격차",
             fontsize=7.8, color=COLOR["ink2"], va="top")
    match = t["재현일치"].mean() * 100
    note = (f"주: 공식값 = 서울시설공단 '시간대별 대기시간평균'. 탑승내역으로 공식 산식(승차한 콜의 승차-예정 평균)을 재현하면 일자×시간대 "
            f"{int(t['reproduced'].notna().sum()):,}칸 중\n    {match:.1f}%가 분 단위까지 일치. 아래 패널은 즉시호출. "
            f"최종 포기 기준 = 재접수 취소(취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수)\n"
            f"    {c['imm_recall']:,}건을 분모에서 뺀 값. 모든 취소를 포함하면 {overall['모든취소']:.1f}%. 위 막대는 일자 칸의 단순 평균.")
    save_fig(fig, "fig_02_official_vs_actual.png", note=note)
    plt.close(fig)


def main():
    t0 = time.time()
    df = pd.read_parquet(CALLS)
    t, h, overall = build_tables(df)
    fig_official_vs_actual_v1(t, h, overall)
    fig_official_vs_actual(t, h, overall)
    print(f"[03_official_vs_actual] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
