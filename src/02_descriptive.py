"""02. 기술통계 (CLAUDE.md 5절).

입력 : data_processed/calls.parquet
출력 : outputs/tables/desc_*.csv
       outputs/figures/fig_01_status_flow.png  즉시호출 흐름도(접수 -> 배차/배차 전 취소 -> 승차/배차 후 취소)
사건은 01단계의 분석 사건 열(E1, E2, E_all, E_all_ab)로 센다. 건수는 canonical_counts.csv와 같아야 한다.
비율 정의
  취소율        = (배차 전 + 배차 후 취소) / 전체 콜
  최종 포기율    = 30분 안에 같은 조건 재접수가 없는 취소 / 전체 콜
  배차 전 취소율 = 배차 전 취소 / 전체 콜
  배차 후 취소율 = 배차 후 취소 / 배차된 콜
시간은 관측 상한을 적용하지 않은 원래 간격(분). 시작 = immediate 접수, scheduled 예정.
"""
import time

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from utils import CALLS, COLOR, FIG_WIDTH_IN, apply_style, load_counts, minutes, save_fig, save_table

DOW = ["월", "화", "수", "목", "금", "토", "일"]


def add_intervals(df):
    df["w_dispatch"] = minutes(df.t_start, df.t_dispatch)        # 시작 -> 배차
    df["w_board"] = minutes(df.t_start, df.t_board)              # 시작 -> 승차
    df["w_cancel"] = minutes(df.t_start, df.t_cancel)            # 시작 -> 취소
    df["w_d2b"] = minutes(df.t_dispatch, df.t_board)             # 배차 -> 승차
    df["w_d2c"] = minutes(df.t_dispatch, df.t_cancel)            # 배차 -> 배차 후 취소
    df["is_disp"] = df.E1 == 1
    df["is_pre"] = df.E1 == 2
    df["is_board"] = (df.E2 == 1).fillna(False).astype(bool)
    df["is_post"] = (df.E2 == 2).fillna(False).astype(bool)
    df["is_cancel"] = df.E_all == 2
    df["is_final"] = df.E_all_ab == 2
    return df


def status_combo(df):
    ox = lambda s: np.where(s, "O", "X")
    t = (df.assign(배차=ox(df.t_dispatch.notna()), 승차=ox(df.t_board.notna()), 취소=ox(df.t_cancel.notna()))
         .groupby(["call_type", "배차", "승차", "취소", "status"], observed=True).size().rename("건수").reset_index())
    t["비율%"] = (t["건수"] / t.groupby("call_type")["건수"].transform("sum") * 100).round(3)
    label = {"pre_cancel": "배차 전 취소", "post_cancel": "배차 후 취소", "boarded": "승차",
             "disp_noboard": "기타 실패(배차 후 미승차·미취소)", "unknown": "상태 불명(중도절단)"}
    t["해석"] = t["status"].astype(str).map(label)
    t.loc[(t.status == "boarded") & (t.취소 == "O"), "해석"] = "승차(승차 후 취소 기록, 플래그)"
    return save_table(t.sort_values(["call_type", "건수"], ascending=[True, False]), "desc_status_combo.csv")


def time_quantiles(df):
    specs = [("시작→배차", "w_dispatch", df.is_disp),
             ("시작→배차 전 취소", "w_cancel", df.is_pre),
             ("배차→승차", "w_d2b", df.is_board),
             ("배차→배차 후 취소", "w_d2c", df.is_post),
             ("시작→승차", "w_board", df.is_board),
             ("시작→취소(배차 전+후)", "w_cancel", df.is_cancel),
             ("시작→최종 포기(재접수 30분 기준)", "w_cancel", df.is_final)]
    rows = []
    for ct, g in df.groupby("call_type", observed=True):
        for name, col, mask in specs:
            v = g.loc[mask[g.index], col]
            rows.append({"call_type": ct, "구간": name, "건수": len(v), "평균": v.mean(), "중앙값": v.median(),
                         "p75": v.quantile(.75), "p90": v.quantile(.9), "p99": v.quantile(.99)})
    t = pd.DataFrame(rows).round(1)
    save_table(t, "desc_time_quantiles.csv")
    print(t[t.call_type == "immediate"].to_string(index=False))
    return t


def group_summary(d, by):
    """그룹별 콜 수, 취소율, 대기시간(즉시호출 전제)."""
    g = d.groupby(by, observed=True)
    disp = d.is_disp
    out = pd.DataFrame({
        "콜수": g.size(),
        "취소율%": g["is_cancel"].mean() * 100,
        "최종포기율%": g["is_final"].mean() * 100,
        "배차전취소율%": g["is_pre"].mean() * 100,
        "배차후취소율%(배차 대비)": d[disp].groupby(by, observed=True)["is_post"].mean() * 100,
        "접수→배차 중앙값": d[disp].groupby(by, observed=True)["w_dispatch"].median(),
        "접수→배차 p90": d[disp].groupby(by, observed=True)["w_dispatch"].quantile(.9),
        "접수→승차 중앙값": d[d.is_board].groupby(by, observed=True)["w_board"].median(),
        "배차→승차 중앙값": d[d.is_board].groupby(by, observed=True)["w_d2b"].median(),
        "임차택시비중%(배차 대비)": d[disp].groupby(by, observed=True).apply(lambda x: (x.vehicle == "임차택시").mean() * 100),
    })
    return out.round(2).reset_index()


def by_groups(imm):
    save_table(group_summary(imm, "hour"), "desc_by_hour.csv")
    t = group_summary(imm, "dow")
    t.insert(1, "요일", t["dow"].map(dict(enumerate(DOW))))
    save_table(t, "desc_by_dow.csv")
    t = group_summary(imm.assign(일구분=np.where(imm.is_offday, "주말·공휴일", "평일")), "일구분")
    save_table(t, "desc_by_offday.csv")
    save_table(group_summary(imm, "month"), "desc_by_month.csv")
    t = group_summary(imm, "o_gu").sort_values("취소율%", ascending=False)
    save_table(t, "desc_by_gu.csv")
    print("\n  구별 취소율 상·하위 3\n", pd.concat([t.head(3), t.tail(3)])[["o_gu", "콜수", "취소율%", "접수→배차 중앙값"]].to_string(index=False))
    save_table(group_summary(imm, "disability_grp").sort_values("콜수", ascending=False), "desc_by_disability.csv")
    save_table(group_summary(imm, "purpose").sort_values("콜수", ascending=False), "desc_by_purpose.csv")


def by_vehicle(imm):
    """배정 차종은 배차된 콜에만 의미가 있다(00단계 1번). 배차 콜만 비교한다."""
    d = imm[imm.is_disp]
    rows = []
    for (v, h), g in d.groupby(["vehicle", "hour"], observed=True):
        rows.append({"vehicle": v, "hour": h, "배차콜수": len(g),
                     "배차후취소율%": g.is_post.mean() * 100,
                     "배차→승차 중앙값": g.loc[g.is_board, "w_d2b"].median()})
    hv = pd.DataFrame(rows).round(2)
    save_table(hv, "desc_by_vehicle_hour.csv")
    t = d.groupby("vehicle", observed=True).agg(
        배차콜수=("status", "size"), 배차후취소=("is_post", "sum"),
        배차_승차_중앙값=("w_d2b", "median"), 접수_배차_중앙값=("w_dispatch", "median"))
    t["배차후취소율%"] = (t["배차후취소"] / t["배차콜수"] * 100).round(2)
    # 같은 시간대끼리 비교(임차택시가 낮 시간대에 몰려 있어 단순 비교는 시간대 효과가 섞인다)
    day = d[d.hour.between(9, 17)]
    t["배차후취소율%(09~17시만)"] = (day.groupby("vehicle", observed=True)["is_post"].mean() * 100).round(2)
    save_table(t.round(1).reset_index(), "desc_by_vehicle.csv")
    print("\n", t.to_string())


def scheduled_appendix(df):
    s = df[df.call_type == "scheduled"].copy()
    s["유형"] = np.select([s.resv_purpose, s.flag_auto_125], ["예약목적", "자동 사전접수(125분)"], "기타 사전접수(간격 5분 초과)")
    t = group_summary(s, "유형")
    t = t.rename(columns={"접수→배차 중앙값": "예정→배차 중앙값", "접수→배차 p90": "예정→배차 p90",
                          "접수→승차 중앙값": "예정→승차 중앙값"})
    save_table(t, "desc_scheduled.csv")


def flow_counts(imm):
    """흐름도 숫자는 기준 건수표(canonical_counts.csv)에서 가져오고, 여기서 다시 센 값과 맞는지 확인한다."""
    c = load_counts()
    assert c["imm_n"] == len(imm) and c["imm_post_cancel"] == int(imm.is_post.sum()), "기준 건수표와 calls가 다르다"
    med = lambda col, m: imm.loc[m, col].median()
    rows = [("접수", c["imm_n"], np.nan),
            ("배차", c["imm_dispatch"], med("w_dispatch", imm.is_disp)),
            ("배차 전 취소", c["imm_pre_cancel"], med("w_cancel", imm.is_pre)),
            ("배차 전 최종 포기", c["imm_pre_final"], med("w_cancel", imm.E1_ab == 2)),
            ("배차 전 재접수 취소", c["imm_pre_recall"], med("w_cancel", imm.E1_ab == 4)),
            ("상태 불명", c["imm_s1_censor"], np.nan),
            ("승차", c["imm_board"], med("w_d2b", imm.is_board)),
            ("배차 후 취소", c["imm_post_cancel"], med("w_d2c", imm.is_post)),
            ("배차 후 최종 포기", c["imm_post_final"], med("w_d2c", imm.E2_ab == 2)),
            ("배차 후 재접수 취소", c["imm_post_recall"], med("w_d2c", imm.E2_ab == 4)),
            ("기타 실패", c["imm_other_fail"], np.nan),
            ("360분 초과 중도절단", c["imm_s2_censor"], np.nan),
            ("취소 전체", c["imm_cancel"], np.nan),
            ("최종 포기 전체", c["imm_final"], np.nan)]
    t = pd.DataFrame(rows, columns=["단계", "건수", "직전 단계부터 중앙값(분)"])
    t["접수 대비%"] = (t["건수"] / c["imm_n"] * 100).round(2)
    save_table(t.round(1), "desc_status_flow.csv")
    return t.set_index("단계")


def _band(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.18):
    """x0의 [y0a,y0b] 구간에서 x1의 [y1a,y1b] 구간으로 이어지는 흐름 띠."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig_status_flow(f):
    """즉시호출이 어디로 가는가 - 3열 흐름도. 높이 = 건수 비율. 취소 노드는 최종 포기(진한)와 재접수 취소(옅은)로 나눈다."""
    import matplotlib.pyplot as plt
    apply_style()
    n = f.loc["접수", "건수"]
    p = lambda k: f.loc[k, "건수"] / n
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.16)
    ax.set_xlim(0, 1); ax.set_ylim(-0.27, 1.02); ax.axis("off")
    W, GAP = 0.03, 0.02                   # 막대 폭, 노드 사이 여백(표면색 간격)
    X = [0.02, 0.29, 0.56]                # 오른쪽 열 레이블은 막대 바깥(오른쪽)에 둔다
    blue, orange = COLOR["board"], COLOR["cancel"]
    orange_lt = "#f5b494"                 # 같은 주황 계열의 옅은 단계 = 재접수 취소

    top = 1.0
    pd_, pc = p("배차"), p("배차 전 취소")
    pb, pa = p("승차"), p("배차 후 취소")
    y_pre = (top - pd_ - GAP - pc, top - pd_ - GAP)
    y_post = (top - pb - GAP - pa, top - pb - GAP)
    _band(ax, X[0] + W, X[1], top, top - pd_, top, top - pd_, blue)
    _band(ax, X[0] + W, X[1], top - pd_, top - pd_ - pc, y_pre[1], y_pre[0], orange)
    _band(ax, X[1] + W, X[2], top, top - pb, top, top - pb, blue)
    _band(ax, X[1] + W, X[2], top - pb, top - pb - pa, y_post[1], y_post[0], orange)
    rect = lambda x, y0, y1, c: ax.add_patch(Rectangle((x, y0), W, y1 - y0, facecolor=c, edgecolor="none"))
    rect(X[0], 0.0, top, blue)
    rect(X[1], top - pd_, top, blue)
    rect(X[2], top - pb, top, blue)
    for x, (y0, y1), fin, rec in [(X[1], y_pre, "배차 전 최종 포기", "배차 전 재접수 취소"),
                                  (X[2], y_post, "배차 후 최종 포기", "배차 후 재접수 취소")]:
        yf = y1 - (y1 - y0) * f.loc[fin, "건수"] / (f.loc[fin, "건수"] + f.loc[rec, "건수"])
        rect(x, yf, y1, orange)                     # 위: 최종 포기
        rect(x, y0, yf - 0.004, orange_lt)          # 아래: 재접수 취소(표면색 간격)

    m = lambda k: f.loc[k, "직전 단계부터 중앙값(분)"]
    cnt = lambda k: f"{f.loc[k, '건수']:,}건 ({f.loc[k, '건수'] / n * 100:.1f}%)"
    LH = 0.072                            # 한 줄 높이(데이터 좌표)

    def block(x, y_top, lines, bold_first=False):
        """여러 줄 레이블. 줄마다 (견본색 또는 None, 문장). 견본은 막대와 같은 색의 작은 사각형."""
        for i, (sw, s_) in enumerate(lines):
            y = y_top - i * LH
            tx = x
            if sw is not None:
                ax.add_patch(Rectangle((x, y - 0.019), 0.012, 0.038, facecolor=sw, edgecolor="none"))
                tx = x + 0.018
            ax.text(tx, y, s_, ha="left", va="center", fontsize=8 if (bold_first and i == 0) else 7.5,
                    color=COLOR["ink"], fontweight="bold" if (bold_first and i == 0) else "normal")

    block(X[0] + W + 0.008, 0.5 + LH / 2, [(None, "즉시호출 접수"), (None, f"{n:,}건")], bold_first=True)
    block(X[1] + W + 0.008, top - pd_ / 2 + LH / 2, [(None, f"배차  {cnt('배차')}"),
                                                     (None, f"접수 후 중앙값 {m('배차'):.1f}분")])
    block(X[2] + W + 0.012, top - pb / 2 + LH / 2, [(None, f"승차  {cnt('승차')}"),
                                                    (None, f"배차 후 중앙값 {m('승차'):.1f}분")])
    block(X[1], y_pre[0] - 0.06, [
        (None, f"배차 전 취소  {cnt('배차 전 취소')}"),
        (orange, f"최종 포기 {f.loc['배차 전 최종 포기', '건수']:,}건  접수 후 중앙값 {m('배차 전 최종 포기'):.1f}분"),
        (orange_lt, f"재접수 {f.loc['배차 전 재접수 취소', '건수']:,}건  접수 후 중앙값 {m('배차 전 재접수 취소'):.1f}분")])
    block(X[2] + W + 0.012, y_post[1] - 0.01, [
        (None, f"배차 후 취소  {cnt('배차 후 취소')}"),
        (orange, f"최종 포기 {f.loc['배차 후 최종 포기', '건수']:,}건  배차 후 중앙값 {m('배차 후 최종 포기'):.1f}분"),
        (orange_lt, f"재접수 {f.loc['배차 후 재접수 취소', '건수']:,}건  배차 후 중앙값 {m('배차 후 재접수 취소'):.1f}분")])

    canc, fin = f.loc["취소 전체", "건수"], f.loc["최종 포기 전체", "건수"]
    fig.text(0.01, 0.97, f"즉시호출 {n / 1e4:.1f}만 건 중 {canc / 1e4:.1f}만 건이 취소됐고, "
                         f"{fin / 1e4:.1f}만 건({fin / n * 100:.1f}%)은 다시 부르지 않았다",
             fontsize=10, fontweight="bold", color=COLOR["ink"], va="top")
    fig.text(0.01, 0.905, "장애인콜택시 즉시호출의 처리 흐름(2025년, 서울 출발). 띠의 높이 = 건수 비율. "
                          "취소 막대의 진한 색 = 최종 포기, 옅은 색 = 재접수",
             fontsize=7.5, color=COLOR["ink2"], va="top")
    note = ("주: 재접수 = 취소 후 30분 안에 같은 출발동·목적동·장애유형으로 새 접수가 들어온 경우(이용자 ID가 없어 쓴 대리 기준.\n"
            "    같은 규칙을 승차 콜에 적용하면 우연 일치는 약 2%). "
            f"기타 실패 {f.loc['기타 실패', '건수']:,}건, 상태 불명 {f.loc['상태 불명', '건수']:,}건, "
            f"360분 초과 중도절단 {f.loc['360분 초과 중도절단', '건수']:,}건은 너무 작아 그리지 않음.\n"
            "    즉시호출 = 이용목적이 예약이 아니고, 예정시각이 접수시각의 1분 전~5분 후인 콜.")
    save_fig(fig, "fig_01_status_flow.png", note=note)
    plt.close(fig)


def main():
    t0 = time.time()
    df = add_intervals(pd.read_parquet(CALLS))
    imm = df[df.call_type == "immediate"].copy()
    status_combo(df)
    time_quantiles(df)
    by_groups(imm)
    by_vehicle(imm)
    scheduled_appendix(df)
    f = flow_counts(imm)
    print("\n", f.to_string())
    fig_status_flow(f)
    print(f"[02_descriptive] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
