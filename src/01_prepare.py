"""01. 분석용 콜 테이블 만들기 (CLAUDE.md 4절).

입력 : 탑승내역 원본(utils.load_raw), data_processed/dong_mapping.csv(00단계)
출력 : data_processed/calls.parquet          콜 1건 = 1행, Stage 1·2·Overall 시간(분)과 사건
       outputs/tables/exclusion_log.csv     단계별 제외 건수
       outputs/tables/prepare_summary.csv   호출유형 x 상태, 단계별 사건 건수

사건 코드
  Stage 1 (시작 -> 배차)      E1: 0 중도절단, 1 배차, 2 배차 전 취소
  Stage 2 (배차 -> 승차)      E2: 0 중도절단, 1 승차, 2 배차 후 취소, 3 기타 실패(배차 후 미승차·미취소)
  Overall (시작 -> 승차)  E_all: 0 중도절단, 1 승차, 2 취소(배차 전+후), 3 기타 실패
시작 시점: immediate = 접수일시, scheduled = 예정일시. 관측 상한: 시작 후 360분.
"""
import time

import numpy as np
import pandas as pd

from utils import (CALLS, DONG_MAPPING, HOLIDAYS_2025, OBS_CAP_MIN, SEOUL_GU, add_call_type, add_status, load_raw,
                   minutes, save_table)


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


def main():
    t0 = time.time()
    df = add_call_type(add_status(load_raw()))
    assert df["flag_board_no_dispatch"].sum() == 0, "배차 없이 승차한 행이 있다"
    df = apply_exclusions(df)
    df = add_times_events(df)
    df = add_covariates(df)
    summarize(df)
    cols = ["row_id", "call_type", "status", "t_request", "t_sched", "t_start", "t_dispatch", "t_board", "t_alight",
            "t_cancel", "T1", "E1", "T2", "E2", "T_all", "E_all", "gap_min", "date", "hour", "dow", "month", "is_holiday",
            "is_offday", "o_gu", "o_dong", "unit_id", "unit_name", "d_gu", "d_dong", "purpose", "resv_purpose",
            "disability", "disability_grp", "vehicle", "fare", "distance_m", "flag_board_and_cancel", "flag_auto_125"]
    df[cols].to_parquet(CALLS, index=False)
    print(f"\n  -> {CALLS.name}: {len(df):,}행 (immediate {int((df.call_type == 'immediate').sum()):,})")
    print(f"[01_prepare] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
