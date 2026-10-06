"""07. 출발동(지도 단위) x 시간대 그룹별 60분 내 승차·최종 포기 지도 (CLAUDE.md 5절).

대상: 즉시호출(콜 기준). 지도 단위 423개(00단계 시점별 경계 대응) x 주간(10~14시)/저녁(15~19시)/야간(20~01시).
칸마다 1분 이산 Aalen-Johansen(survival.py)으로
  (a) 60분 내 최종 포기 누적확률   (b) 60분 내 승차 누적확률   (참고) 60분 내 재접수 취소 누적확률
  (비교) 선행연구 방식 naive KM(취소 = 중도절단)의 60분 내 승차 누적확률과 중앙값
작은 칸: 같은 구·같은 시간대 값 쪽으로 수축. 수축값 = w x 원값 + (1 - w) x 구 값, w = n / (n + 50). 20건 미만은 회색.
굵은 테두리(fig_06): 콜 50건 이상이면서 원값 기준 60분 내 승차 < 50%인 동. 제목 숫자와 같은 기준.
출력: outputs/tables/dong_hour_abandon.csv, dong_top10_abandon.csv, dong_bottom10_board.csv
      outputs/figures/fig_06_equity_map.png(야간: 선행연구 방식 vs 경쟁위험), fig_A3_abandon_day_night.png
"""
import time

import numpy as np
import pandas as pd

from survival import discrete_cif
from utils import (CALLS, COLOR, DATA_PROC, FS, HOUR_GROUPS, MAP_UNITS, apply_style, clean, draw_figures, fig_title,
                   fig_width, load_counts, save_fig, save_table)

H = 60
M_SHRINK = 50
N_GRAY = 20
N_FLAG = 50
GROUPS = ["주간(10~14시)", "저녁(15~19시)", "야간(20~01시)"]


def cell_values(g):
    c = discrete_cif(g.T_all.values, g.E_all_ab.values, [1, 2, 4, 3]).set_index("t")
    nb = c["naive_승차"]
    return {"n": len(g), "포기60": c.loc[H, "F_2"] * 100, "승차60": c.loc[H, "F_1"] * 100,
            "재접수60": c.loc[H, "F_4"] * 100, "naive승차60": nb.loc[H] * 100,
            "naive중앙값": float(nb.index[(nb >= 0.5).to_numpy()][0]) if (nb >= 0.5).any() else np.nan}


def compute(imm, units):
    gu_of = units.set_index("unit_id")["unit_gu"]
    hr = imm.t_request.dt.hour
    rows = []
    for grp in GROUPS:
        d = imm[hr.isin(HOUR_GROUPS[grp])]
        gu_val = {gu: cell_values(x) for gu, x in d.groupby("o_gu", observed=True)}
        for u, x in d.groupby("unit_id", observed=True):
            r = {"unit_id": u, "시간대": grp, **cell_values(x)}
            gu = gu_of[u]
            w = r["n"] / (r["n"] + M_SHRINK)
            r.update({"구": gu, "수축가중치": w})
            for k in ["포기60", "승차60", "재접수60", "naive승차60"]:
                r[f"구_{k}"] = gu_val[gu][k]
                r[f"{k}_수축"] = w * r[k] + (1 - w) * gu_val[gu][k]
            rows.append(r)
    t = pd.DataFrame(rows).merge(units[["unit_id", "unit_name"]], on="unit_id", how="left")
    t["회색(20건 미만)"] = t["n"] < N_GRAY
    t["표시(50건 이상·승차60<50%)"] = (t["n"] >= N_FLAG) & (t["승차60"] < 50)
    t["표시_naive(50건 이상·naive승차60<50%)"] = (t["n"] >= N_FLAG) & (t["naive승차60"] < 50)
    cols = ["unit_id", "unit_name", "구", "시간대", "n", "수축가중치", "포기60", "포기60_수축", "승차60", "승차60_수축",
            "재접수60", "재접수60_수축", "naive승차60", "naive승차60_수축", "naive중앙값", "구_포기60", "구_승차60",
            "회색(20건 미만)", "표시(50건 이상·승차60<50%)", "표시_naive(50건 이상·naive승차60<50%)"]
    t = t[cols].rename(columns={"포기60": "60분 내 최종 포기%(원값)", "포기60_수축": "60분 내 최종 포기%(수축)",
                                "승차60": "60분 내 승차%(원값)", "승차60_수축": "60분 내 승차%(수축)",
                                "재접수60": "60분 내 재접수 취소%(원값)", "재접수60_수축": "60분 내 재접수 취소%(수축)",
                                "naive승차60": "naive KM 60분 내 승차%(원값)", "naive승차60_수축": "naive KM 60분 내 승차%(수축)",
                                "naive중앙값": "naive KM 승차 중앙값(분)", "구_포기60": "구 60분 내 최종 포기%",
                                "구_승차60": "구 60분 내 승차%"})
    save_table(t.round(3), "dong_hour_abandon.csv")
    return t


def rank_tables(t):
    big = t[t["n"] >= N_FLAG]
    keep = ["시간대", "unit_name", "구", "n", "60분 내 최종 포기%(수축)", "60분 내 최종 포기%(원값)", "60분 내 승차%(수축)",
            "60분 내 승차%(원값)", "naive KM 60분 내 승차%(원값)"]
    top = big.sort_values("60분 내 최종 포기%(수축)", ascending=False).groupby("시간대", sort=False).head(10)
    bot = big.sort_values("60분 내 승차%(수축)").groupby("시간대", sort=False).head(10)
    order = {g: i for i, g in enumerate(GROUPS)}
    top = top[keep].sort_values(["시간대", "60분 내 최종 포기%(수축)"], key=lambda s: s.map(order) if s.name == "시간대" else -s)
    bot = bot[keep].sort_values(["시간대", "60분 내 승차%(수축)"], key=lambda s: s.map(order) if s.name == "시간대" else s)
    save_table(top.round(2), "dong_top10_abandon.csv")
    save_table(bot.round(2), "dong_bottom10_board.csv")
    for name, x in [("포기율 상위 10(야간)", top), ("60분 승차 하위 10(야간)", bot)]:
        print(f"\n  {name}\n", x[x["시간대"] == "야간(20~01시)"].drop(columns="시간대").round(1).to_string(index=False))


def summary_counts(t):
    n = t[t["시간대"] == "야간(20~01시)"]
    s = {"all_aj": int((n["60분 내 승차%(원값)"] < 50).sum()), "all_naive": int((n["naive KM 60분 내 승차%(원값)"] < 50).sum()),
         "big_aj": int(n["표시(50건 이상·승차60<50%)"].sum()), "big_naive": int(n["표시_naive(50건 이상·naive승차60<50%)"].sum()),
         "n_units": len(n), "n_big": int((n["n"] >= N_FLAG).sum()), "n_gray": int(n["회색(20건 미만)"].sum()),
         "naive_med_missing": int(n["naive KM 승차 중앙값(분)"].isna().sum()),
         "flag_abandon_mean": n.loc[n["표시(50건 이상·승차60<50%)"], "60분 내 최종 포기%(원값)"].mean()}
    print("  야간 요약:", s)
    return s


# ---------------------------------------------------------------- 그림
BLUE_BINS = [0, 40, 50, 60, 70, 80, 90, 100]
BLUE = ["#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]     # 기준 팔레트 파랑 단계
ORANGE_BINS = [0, 3, 6, 9, 12, 15, 20, 100]
ORANGE = ["#fde7dc", "#f9c8b0", "#f5a684", "#ef8559", "#eb6834", "#c4501f", "#8a3412"]
GRAY = "#d9d8d2"


def _binned(v, bins, colors):
    idx = np.clip(np.digitize(v, bins[1:-1], right=False), 0, len(colors) - 1)
    return [colors[i] for i in idx]


def _draw(ax, units, gu, vals, gray, bins, colors, flag=None):
    import matplotlib.patheffects  # noqa: F401
    g = units.merge(vals, on="unit_id", how="left")
    fc = np.where(g["gray"].fillna(True), GRAY, pd.Series(_binned(g["v"].fillna(0).values, bins, colors), index=g.index))
    g.plot(ax=ax, color=fc, edgecolor=COLOR["surface"], linewidth=0.25)
    gu.boundary.plot(ax=ax, color=COLOR["surface"], linewidth=0.9)
    if flag is not None:
        f = g[g["flag"].fillna(False)]
        if len(f):
            f.boundary.plot(ax=ax, color=COLOR["ink"], linewidth=0.9)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _legend(fig, x0, y0, bins, colors, label, w=0.05, h=0.022, fmt="{:.0f}", fs=6.3):
    from matplotlib.patches import Rectangle
    for i, c in enumerate(colors):
        fig.patches.append(Rectangle((x0 + i * w, y0), w - 0.003, h, color=c, transform=fig.transFigure, figure=fig))
        lab = f"<{bins[1]}" if i == 0 else (f"≥{bins[i]}" if i == len(colors) - 1 else fmt.format(bins[i]))
        fig.text(x0 + i * w + (w - 0.003) / 2, y0 - 0.008, lab, ha="center", va="top", fontsize=fs, color=COLOR["ink2"])
    fig.text(x0, y0 + h + 0.008, label, fontsize=fs + 0.5, color=COLOR["ink2"], va="bottom")


def _key(fig, x, y, text, fs, **patch):
    """그림 좌표에 작은 견본 사각형 + 설명(지도 범례용)."""
    from matplotlib.patches import Rectangle
    fig.patches.append(Rectangle((x, y), 0.03, 0.022, transform=fig.transFigure, figure=fig, **patch))
    fig.text(x + 0.037, y + 0.011, text, fontsize=fs, color=COLOR["ink2"], va="center")


def fig_equity(t, units, gu, s):
    import matplotlib.pyplot as plt
    apply_style()
    ann = not clean()
    cnt = load_counts()
    n = t[t["시간대"] == "야간(20~01시)"]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width(), 4.2 if ann else 3.3), gridspec_kw={"wspace": 0.02})
    if ann:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.27)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.22)
    panels = [("naive KM 60분 내 승차%(수축)", "표시_naive(50건 이상·naive승차60<50%)",
               f"선행연구 방식(취소 = 중도절단): 50% 미만 {s['big_naive']}곳", "(가) 선행연구 방식(취소 = 중도절단)"),
              ("60분 내 승차%(수축)", "표시(50건 이상·승차60<50%)",
               f"경쟁위험(취소 = 경쟁사건): 50% 미만 {s['big_aj']}곳", "(나) 경쟁위험(취소 = 경쟁사건)")]
    for ax, (col, flag, title_a, title_c) in zip(axes, panels):
        vals = pd.DataFrame({"unit_id": n["unit_id"], "v": n[col], "gray": n["회색(20건 미만)"], "flag": n[flag]})
        _draw(ax, units, gu, vals, None, BLUE_BINS, BLUE, flag=True)
        ax.set_title(title_a if ann else title_c, loc="left", fontsize=8.3 if ann else None, pad=2)
    if ann:
        _legend(fig, 0.06, 0.225, BLUE_BINS, BLUE, "야간 60분 안에 승차한 비율(%)")
        _key(fig, 0.52, 0.225, "= 콜 50건 이상이면서 50% 미만", 6.8, facecolor="white", edgecolor=COLOR["ink"], linewidth=0.9)
        _key(fig, 0.52, 0.19, f"= 야간 콜 {N_GRAY}건 미만({s['n_gray']}곳)", 6.8, color=GRAY)
    else:
        fs = FS("small")
        _legend(fig, 0.02, 0.085, BLUE_BINS, BLUE, "야간 60분 안에 승차한 비율(%)", w=0.06, h=0.03, fs=fs)
        _key(fig, 0.5, 0.11, "콜 50건 이상이면서 50% 미만", fs, facecolor="white", edgecolor=COLOR["ink"], linewidth=0.9)
        _key(fig, 0.5, 0.06, f"야간 콜 {N_GRAY}건 미만", fs, color=GRAY)
    fig_title(fig, f"선행연구 방식은 문제를 가린다: 밤에 60분 안에 절반도 못 타는 동 {s['big_aj']}곳이 "
                   f"{s['big_naive']}곳으로 보인다",
              f"야간(20~01시) 즉시호출 {cnt['imm_night_n']:,}건, 지도 단위 {s['n_units']}곳. "
              "두 지도는 같은 데이터·같은 색 척도이고, 취소를 처리하는 방식만 다르다", y_sub=0.915)
    note = (f"주: {s['big_aj']}곳·{s['big_naive']}곳은 야간 콜 {N_FLAG}건 이상인 {s['n_big']}곳 중 원값 기준(콜 수와 무관하게 세면 "
            f"{s['all_aj']}곳 vs {s['all_naive']}곳).\n"
            f"    굵은 테두리 {s['big_aj']}곳의 야간 60분 내 최종 포기 확률은 평균 {s['flag_abandon_mean']:.1f}%. "
            f"색은 같은 구 야간 값 쪽으로 수축한 값(w = n/(n+{M_SHRINK})). 회색 = 야간 콜 {N_GRAY}건 미만({s['n_gray']}곳).\n"
            f"    선행연구 방식 KM으로는 승차 중앙값이 없는 '빈칸'이 {s['naive_med_missing']}곳뿐이어서, 빈칸 대신 같은 지표(60분 내 승차)로 비교함.\n"
            "    지도 단위 = 출발동을 시점별 행정동 경계로 2025-12-31 경계에 대응(통계청 SGIS 경계, vuski/admdongkor 가공, CC BY 4.0).")
    save_fig(fig, "fig_06_equity_map.png", note=note)
    plt.close(fig)


def fig_day_night(t, units, gu):
    import matplotlib.pyplot as plt
    apply_style()
    ann = not clean()
    fig, axes = plt.subplots(1, 2, figsize=(fig_width(), 3.7 if ann else 3.2), gridspec_kw={"wspace": 0.02})
    if ann:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.22)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.2)
    meds = {}
    for i, (ax, grp) in enumerate(zip(axes, ["주간(10~14시)", "야간(20~01시)"])):
        d = t[t["시간대"] == grp]
        vals = pd.DataFrame({"unit_id": d["unit_id"], "v": d["60분 내 최종 포기%(수축)"], "gray": d["회색(20건 미만)"]})
        _draw(ax, units, gu, vals, None, ORANGE_BINS, ORANGE)
        meds[grp] = d[d["n"] >= N_FLAG]["60분 내 최종 포기%(원값)"].median()
        if ann:
            ax.set_title(f"{grp}: 동 중앙값 {meds[grp]:.1f}%", loc="left", fontsize=8.3, pad=4)
        else:
            ax.set_title(["(가) 주간 10~14시", "(나) 야간 20~01시"][i], loc="left", pad=2)
    if ann:
        _legend(fig, 0.06, 0.17, ORANGE_BINS, ORANGE, "60분 안에 최종 포기한 비율(%)")
    else:
        fs = FS("small")
        _legend(fig, 0.02, 0.08, ORANGE_BINS, ORANGE, "60분 안에 최종 포기한 비율(%)", w=0.06, h=0.03, fs=fs)
        _key(fig, 0.5, 0.08, f"콜 {N_GRAY}건 미만", fs, color=GRAY)
    fig_title(fig, "같은 동이라도 밤에 부르면 포기가 훨씬 많다",
              f"콜 {N_FLAG}건 이상인 동의 60분 내 최종 포기 중앙값: 주간 {meds['주간(10~14시)']:.1f}%, 야간 {meds['야간(20~01시)']:.1f}%")
    note = (f"주: 즉시호출, 경쟁위험(1분 이산 Aalen-Johansen) 60분 내 최종 포기 누적확률.\n"
            f"    색은 같은 구·같은 시간대 값 쪽으로 수축(w = n/(n+{M_SHRINK})), 회색 = 콜 {N_GRAY}건 미만. 중앙값은 콜 {N_FLAG}건 이상인 동의 원값 기준. "
            "최종 포기 = 취소 후 30분 안에 같은 출발동·목적동·장애유형 재접수가 없는 취소.")
    save_fig(fig, "fig_A3_abandon_day_night.png", note=note)
    plt.close(fig)


def main():
    import geopandas as gpd
    t0 = time.time()
    df = pd.read_parquet(CALLS, columns=["call_type", "T_all", "E_all_ab", "t_request", "unit_id", "o_gu"])
    imm = df[df.call_type == "immediate"]
    assert len(imm) == load_counts()["imm_n"] and imm.unit_id.notna().all()
    units = gpd.read_parquet(MAP_UNITS)
    gu = gpd.read_parquet(DATA_PROC / "gu_boundary.parquet")
    t = compute(imm, units)
    rank_tables(t)
    s = summary_counts(t)
    draw_figures(fig_equity, t, units, gu, s)
    draw_figures(fig_day_night, t, units, gu)
    print(f"[07_equity_map] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
