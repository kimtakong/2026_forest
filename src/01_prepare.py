"""01. 분석용 콜 테이블 만들기 (CLAUDE.md 4절).

입력 : 탑승내역 원본(utils.load_raw), data_processed/dong_mapping.csv(00단계)
출력 : data_processed/calls.parquet          콜 1건 = 1행, Stage 1·2·Overall 시간(분)과 사건
       outputs/tables/exclusion_log.csv     단계별 제외 건수
       outputs/tables/prepare_summary.csv   호출유형 x 상태, 단계별 사건 건수
       outputs/tables/cancel_recall_check.csv  재접수 판별(10/30/60분)과 취소 시각 분포
       outputs/tables/canonical_counts.csv  기준 건수표. 모든 그림·표의 숫자는 여기서 인용한다

사건 코드
  Stage 1 (시작 -> 배차)      E1: 0 중도절단, 1 배차, 2 배차 전 취소
  Stage 2 (배차 -> 승차)      E2: 0 중도절단, 1 승차, 2 배차 후 취소, 3 기타 실패(배차 후 미승차·미취소)
  Overall (시작 -> 승차)  E_all: 0 중도절단, 1 승차, 2 취소(배차 전+후), 3 기타 실패
  '최종 포기' 기준(E1_ab, E2_ab, E_all_ab): 취소 중 30분 안에 같은 조건 재접수가 있는 건을 사건 4로 분리(recall.py)
시작 시점: immediate = 접수일시, scheduled = 예정일시. 관측 상한: 시작 후 360분.
"""
import time

import numpy as np
import pandas as pd

import recall
from utils import (CALLS, CANONICAL, DONG_MAPPING, HOLIDAYS_2025, OBS_CAP_MIN, SEOUL_GU, add_call_type, add_status,
                   load_raw, minutes, save_table)


def apply_exclusions(df):
    """제외 규칙을 순서대로 적용하고 단계별 건수를 기록한다."""
    r2d = minutes(df.t_request, df.t_dispatch)
    r2c = minutes(df.t_request, df.t_cancel)
    d2b = minutes(df.t_dispatch, df.t_board)
    b2a = minutes(df.t_board, df.t_alight)
    rules = [
        ("E1", "출발구 또는 출발동 결측", df.o_gu.isna() | df.o_dong.isna()),
        ("E2", "출발구가 서울 25개 구가 아님", ~df.o_gu.isin(SEOUL_GU)),
        ("E3a", "음수 시간: 배차일시 < 접수일시", r2d < 0),
        ("E3b", "음수 시간: 취소일시 < 접수일시", r2c < 0),
        ("E3c", "음수 시간: 승차일시 < 배차일시", d2b < 0),
        ("E3d", "음수 시간: 하차일시 < 승차일시", b2a < 0),
        ("E4", "예정일시가 접수일시보다 1분 넘게 앞섬(날짜 입력 오류 추정)", df.gap_min < -1),
    ]
    keep = pd.Series(True, index=df.index)
    imm = df.call_type == "immediate"
    log = [{"단계": "E0", "규칙": "원본", "제외": 0, "남은 건수": len(df),
            "제외(immediate)": 0, "남은 건수(immediate)": int(imm.sum())}]
    for code, rule, mask in rules:
        drop = keep & mask.fillna(False)
        keep &= ~drop
        log.append({"단계": code, "규칙": rule, "제외": int(drop.sum()), "남은 건수": int(keep.sum()),
                    "제외(immediate)": int((drop & imm).sum()), "남은 건수(immediate)": int((keep & imm).sum())})
    log = pd.DataFrame(log)
    print(log.to_string(index=False))
    save_table(log, "exclusion_log.csv")
    return df[keep].copy()


def add_times_events(df):
    imm = df.call_type == "immediate"
    start = df.t_request.where(imm, df.t_sched)
    df["t_start"] = start
    st = df.status
    disp = df.t_dispatch.notna()
    cap = OBS_CAP_MIN

    # Stage 1: 시작 -> 배차 / 배차 전 취소
    t_disp = minutes(start, df.t_dispatch)
    T1 = np.select([disp, st == "pre_cancel"], [t_disp, minutes(start, df.t_cancel)], 0.0)
    E1 = np.select([disp, st == "pre_cancel"], [1, 2], 0)
    over = T1 > cap
    df["T1"], df["E1"] = np.where(over, cap, T1), np.where(over, 0, E1)

    # Stage 2: 배차 -> 승차 / 배차 후 취소 / 기타 실패 (배차가 관측 상한 안에 있는 콜만)
    in2 = disp & ~over
    d2_end = np.select([st == "boarded", st == "post_cancel", st == "disp_noboard"],
                       [minutes(df.t_dispatch, df.t_board), minutes(df.t_dispatch, df.t_cancel),
                        minutes(df.t_dispatch, df.t_alight).fillna(0.0)], np.nan)
    E2 = np.select([st == "boarded", st == "post_cancel", st == "disp_noboard"], [1, 2, 3], 0)
    over2 = (t_disp + d2_end) > cap
    df["T2"] = np.where(in2, np.where(over2, cap - t_disp, d2_end), np.nan)
    df["E2"] = pd.Series(np.where(over2, 0, E2), index=df.index).where(in2).astype("Int8")

    # Overall: 시작 -> 승차 / 취소 / 기타 실패
    T = np.select([st == "boarded", st.isin(["pre_cancel", "post_cancel"]), st == "disp_noboard"],
                  [minutes(start, df.t_board), minutes(start, df.t_cancel),
                   minutes(start, df.t_alight).fillna(t_disp)], 0.0)
    E = np.select([st == "boarded", st.isin(["pre_cancel", "post_cancel"]), st == "disp_noboard"], [1, 2, 3], 0)
    overT = T > cap
    df["T_all"], df["E_all"] = np.where(overT, cap, T), np.where(overT, 0, E)
    for c in ["E1", "E_all"]:
        df[c] = df[c].astype("int8")
    return df


def add_covariates(df):
    s = df.t_start
    df["date"] = s.dt.normalize()
    df["hour"] = s.dt.hour.astype("int8")
    df["dow"] = s.dt.dayofweek.astype("int8")          # 0=월
    df["month"] = s.dt.month.astype("int8")
    df["is_holiday"] = df["date"].isin(HOLIDAYS_2025)
    df["is_offday"] = df["is_holiday"] | (df["dow"] >= 5)
    top6 = df["disability"].value_counts().head(6).index
    df["disability_grp"] = df["disability"].astype(str).where(df["disability"].isin(top6), "기타").astype("category")
    df["flag_auto_125"] = df["gap_min"].between(124.5, 125.5)   # 시스템 자동 사전접수(00단계 3번)

    m = pd.read_csv(DONG_MAPPING, encoding="utf-8-sig", usecols=["o_gu", "o_dong", "unit_id", "unit_name"])
    df = df.merge(m, on=["o_gu", "o_dong"], how="left")
    df["o_gu"] = df["o_gu"].astype(str).astype("category")
    df["o_dong"] = df["o_dong"].astype(str).astype("category")
    for c in ["unit_id", "unit_name"]:
        df[c] = df[c].astype("category")
    n_na = df["unit_id"].isna().sum()
    print(f"  지도 단위 미배정: {n_na:,}건")
    return df


def summarize(df):
    rows = []
    for ct, g in df.groupby("call_type", observed=True):
        for col, lab in [("status", "상태")]:
            for k, v in g[col].value_counts().items():
                rows.append({"call_type": ct, "구분": lab, "값": k, "건수": int(v)})
        for col, lab, names in [("E1", "Stage1", {0: "중도절단", 1: "배차", 2: "배차 전 취소"}),
                                ("E2", "Stage2", {0: "중도절단", 1: "승차", 2: "배차 후 취소", 3: "기타 실패"}),
                                ("E_all", "Overall", {0: "중도절단", 1: "승차", 2: "취소", 3: "기타 실패"})]:
            for k, v in g[col].value_counts().sort_index().items():
                rows.append({"call_type": ct, "구분": lab, "값": names[int(k)], "건수": int(v)})
    t = pd.DataFrame(rows)
    t["비율%"] = (t["건수"] / t.groupby(["call_type", "구분"])["건수"].transform("sum") * 100).round(2)
    save_table(t, "prepare_summary.csv")
    print(t[t.call_type == "immediate"].to_string(index=False))

    imm = df[df.call_type == "immediate"]
    q = imm.groupby("E1")["T1"].quantile([.5, .9]).unstack().round(1)
    print("\n  immediate Stage1 시간(분) 분위수 by 사건\n", q.to_string())


def recall_check(df):
    """즉시호출 취소를 '재접수 취소'와 '최종 포기'로 나눈 결과와 취소 시각 분포."""
    imm = df[df.call_type == "immediate"]
    n_calls = len(imm)
    pre = imm[imm.E1 == 2]
    post = imm[imm.E2 == 2]
    groups = {"배차 전 취소": pre, "배차 후 취소": post, "전체 취소": pd.concat([pre, post])}
    status_of = df.set_index("row_id")["status"]
    rows = []
    add = lambda sec, typ, w, k, v: rows.append({"구분": sec, "취소유형": typ, "기준(분)": w, "지표": k, "값": v})

    for typ, g in groups.items():
        for w in recall.RECALL_WINDOWS:
            r = g[f"recall_{w}"]
            nxt = status_of.reindex(g.loc[r, "recall_next_row"].astype("int64")).values
            add("A 재접수 판별", typ, w, "취소 건수", len(g))
            add("A 재접수 판별", typ, w, "재접수 취소 건수", int(r.sum()))
            add("A 재접수 판별", typ, w, "재접수 취소 비율%", round(r.mean() * 100, 2))
            add("A 재접수 판별", typ, w, "최종 포기 건수", int((~r).sum()))
            add("A 재접수 판별", typ, w, "최종 포기 비율%(취소 대비)", round((~r).mean() * 100, 2))
            add("A 재접수 판별", typ, w, "최종 포기 비율%(즉시호출 전체 대비)", round((~r).sum() / n_calls * 100, 2))
            add("A 재접수 판별", typ, w, "재접수 콜이 결국 승차한 비율%", round((nxt == "boarded").mean() * 100, 2))

    base = recall.baseline_false_match(imm)
    for w, v in base.items():
        add("B 위양성 기준선", "승차 콜(같은 규칙을 승차 시각에 적용)", w, "우연 일치 비율%", round(v, 2))

    for typ, g in groups.items():
        r = g[f"recall_{recall.RECALL_DEFAULT}"]
        r2c = minutes(g.t_request, g.t_cancel)
        d2c = minutes(g.t_dispatch, g.t_cancel)
        for sub, m in [("전체", slice(None)), ("재접수 취소", r), ("최종 포기", ~r)]:
            x = r2c[m]
            add("C 취소 시각 분포", f"{typ} | {sub}", recall.RECALL_DEFAULT, "접수→취소 중앙값(분)", round(x.median(), 2))
            for c in (2, 5, 10):
                add("C 취소 시각 분포", f"{typ} | {sub}", recall.RECALL_DEFAULT, f"접수 후 {c}분 이내 취소 비중%", round((x <= c).mean() * 100, 2))
            if typ == "배차 후 취소":
                y = d2c[m]
                add("C 취소 시각 분포", f"{typ} | {sub}", recall.RECALL_DEFAULT, "배차→취소 중앙값(분)", round(y.median(), 2))
                add("C 취소 시각 분포", f"{typ} | {sub}", recall.RECALL_DEFAULT, "배차 후 2분 이내 취소 비중%", round((y <= 2).mean() * 100, 2))
                add("C 취소 시각 분포", f"{typ} | {sub}", recall.RECALL_DEFAULT,
                    "배차 후 9~12분 취소 비중%(전화 미연결 자동취소 규정 구간)", round(y.between(9, 12).mean() * 100, 2))

    for typ, g in groups.items():
        add("D 참고", typ, None, "취소 전에 같은 조건 새 접수(중복 접수) 비율%", round(g["dup_before_cancel"].mean() * 100, 2))
    gap = post.loc[post["recall_60"], "recall_gap_min"]
    add("D 참고", "배차 후 취소", 60, "재접수 중 취소 후 10~11분에 들어온 비중%(배차 후 취소 시 10분 접수 제한 규정)",
        round(gap.between(10, 11).mean() * 100, 2))

    n_recall = int(groups["전체 취소"][f"recall_{recall.RECALL_DEFAULT}"].sum())
    n_final = len(groups["전체 취소"]) - n_recall
    add("E 이용 건(에피소드) 기준", "전체 취소", recall.RECALL_DEFAULT, "에피소드 수(콜 - 재접수 취소)", n_calls - n_recall)
    add("E 이용 건(에피소드) 기준", "전체 취소", recall.RECALL_DEFAULT, "최종 포기 비율%(에피소드 대비)",
        round(n_final / (n_calls - n_recall) * 100, 2))
    t = pd.DataFrame(rows)
    save_table(t, "cancel_recall_check.csv")
    print(t[(t["기준(분)"] == 30) & t["구분"].str.startswith("A")].to_string(index=False))
    print(t[t["구분"].str[0].isin(["B", "D", "E"])].to_string(index=False))


def canonical_counts(raw, df):
    """기준 건수표. 세 층을 구분한다: 원본(제외 전 상태) / 분석 대상 상태 / 분석 사건(360분 상한 적용)."""
    rows = []
    add = lambda key, layer, label, n, note="": rows.append(
        {"key": key, "층": layer, "항목": label, "건수": int(n), "설명": note})
    canc = lambda d: d.status.isin(["pre_cancel", "post_cancel"])
    ri = raw[raw.call_type == "immediate"]
    add("raw_n", "원본", "탑승내역 전체", len(raw))
    add("raw_cancel", "원본", "취소 전체(즉시+사전)", canc(raw).sum(), "승차 후 취소 기록 34건은 승차로 셈. 공식 통계에서 빠진 규모")
    add("raw_imm_cancel", "원본", "즉시호출 취소", canc(ri).sum())
    add("raw_imm_pre_cancel", "원본", "즉시호출 배차 전 취소", (ri.status == "pre_cancel").sum())
    add("raw_imm_post_cancel", "원본", "즉시호출 배차 후 취소", (ri.status == "post_cancel").sum())
    add("raw_sched_cancel", "원본", "사전접수 취소", canc(raw[raw.call_type == "scheduled"]).sum(), "일정 변경일 수 있어 '포기'로 부르지 않음")

    imm = df[df.call_type == "immediate"]
    add("n_all", "분석 대상 상태", "제외 후 전체 콜", len(df))
    add("n_sched", "분석 대상 상태", "사전접수 콜", (df.call_type == "scheduled").sum())
    add("imm_n", "분석 대상 상태", "즉시호출 콜", len(imm), "주 분석 모집단")
    for st, lab in [("boarded", "승차"), ("pre_cancel", "배차 전 취소"), ("post_cancel", "배차 후 취소"),
                    ("disp_noboard", "배차 후 미승차·미취소"), ("unknown", "상태 불명")]:
        add(f"imm_status_{st}", "분석 대상 상태", f"즉시호출 {lab}(상태 기준)", (imm.status == st).sum())

    w = recall.RECALL_DEFAULT
    add("imm_dispatch", "분석 사건", "배차(E1=1)", (imm.E1 == 1).sum())
    add("imm_pre_cancel", "분석 사건", "배차 전 취소(E1=2)", (imm.E1 == 2).sum())
    add("imm_s1_censor", "분석 사건", "Stage 1 중도절단(상태 불명)", (imm.E1 == 0).sum())
    add("imm_board", "분석 사건", "승차(E2=1)", (imm.E2 == 1).sum())
    add("imm_post_cancel", "분석 사건", "배차 후 취소(E2=2)", (imm.E2 == 2).sum(),
        "상태 기준보다 1건 적음: 접수 후 360분 넘어 취소돼 중도절단")
    add("imm_other_fail", "분석 사건", "기타 실패(E2=3, 배차 후 미승차·미취소)", (imm.E2 == 3).sum(),
        "상태 기준 622건 중 23건은 하차일시가 접수 후 360분을 넘어 중도절단")
    add("imm_s2_censor", "분석 사건", "Stage 2 중도절단(360분 초과)", (imm.E2 == 0).sum())
    add("imm_cancel", "분석 사건", "취소 전체(E_all=2)", (imm.E_all == 2).sum())
    add("imm_censor", "분석 사건", "Overall 중도절단(E_all=0)", (imm.E_all == 0).sum(), "상태 불명 24 + 360분 초과 24")
    add("imm_pre_final", "분석 사건", f"배차 전 최종 포기(E1_ab=2, 재접수 {w}분 기준)", (imm.E1_ab == 2).sum())
    add("imm_pre_recall", "분석 사건", "배차 전 재접수 취소(E1_ab=4)", (imm.E1_ab == 4).sum())
    add("imm_post_final", "분석 사건", "배차 후 최종 포기(E2_ab=2)", (imm.E2_ab == 2).sum())
    add("imm_post_recall", "분석 사건", "배차 후 재접수 취소(E2_ab=4)", (imm.E2_ab == 4).sum())
    add("imm_final", "분석 사건", "최종 포기 전체(E_all_ab=2)", (imm.E_all_ab == 2).sum(), "보고서 '포기'의 기본 숫자")
    add("imm_recall", "분석 사건", "재접수 취소 전체(E_all_ab=4)", (imm.E_all_ab == 4).sum())
    add("imm_episodes", "분석 사건", "이용 건(에피소드) = 즉시호출 - 재접수 취소", len(imm) - (imm.E_all_ab == 4).sum())
    t = pd.DataFrame(rows)
    c = dict(zip(t.key, t["건수"]))
    assert c["imm_dispatch"] + c["imm_pre_cancel"] + c["imm_s1_censor"] == c["imm_n"]
    assert c["imm_board"] + c["imm_post_cancel"] + c["imm_other_fail"] + c["imm_s2_censor"] == c["imm_dispatch"]
    assert c["imm_pre_cancel"] + c["imm_post_cancel"] == c["imm_cancel"] == c["imm_final"] + c["imm_recall"]
    t.to_csv(CANONICAL, index=False, encoding="utf-8-sig")
    print(f"  -> {CANONICAL.name}")
    print(t[["key", "항목", "건수"]].to_string(index=False))


def main():
    t0 = time.time()
    raw = add_call_type(add_status(load_raw()))
    assert raw["flag_board_no_dispatch"].sum() == 0, "배차 없이 승차한 행이 있다"
    df = apply_exclusions(raw)
    df = add_times_events(df)
    df = add_covariates(df)
    df = recall.add_abandon_events(recall.add_recall(df))
    summarize(df)
    recall_check(df)
    canonical_counts(raw, df)
    cols = ["row_id", "call_type", "status", "t_request", "t_sched", "t_start", "t_dispatch", "t_board", "t_alight",
            "t_cancel", "T1", "E1", "T2", "E2", "T_all", "E_all", "E1_ab", "E2_ab", "E_all_ab", "recall_gap_min",
            "recall_next_row", "recall_10", "recall_30", "recall_60", "dup_before_cancel", "gap_min", "date", "hour",
            "dow", "month", "is_holiday", "is_offday", "o_gu", "o_dong", "unit_id", "unit_name", "d_gu", "d_dong",
            "purpose", "resv_purpose", "disability", "disability_grp", "vehicle", "fare", "distance_m",
            "flag_board_and_cancel", "flag_auto_125"]
    df[cols].to_parquet(CALLS, index=False)
    print(f"\n  -> {CALLS.name}: {len(df):,}행 (immediate {int((df.call_type == 'immediate').sum()):,})")
    print(f"[01_prepare] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
