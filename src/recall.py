"""재접수 판별. 이용자 ID가 없으므로 대리 기준을 쓴다.

재접수 취소: 취소된 콜과 같은 (출발구, 출발동, 목적구, 목적동, 장애유형)으로
            취소 시각 이후 w분 안에 새 접수가 있는 경우. 없으면 '최종 포기'.
위양성 기준선: 승차한 콜에 같은 규칙을 승차 시각 기준으로 적용한 비율.
              승차한 사람은 같은 경로를 다시 부를 이유가 없으므로, 이 비율은
              '다른 사람이 우연히 같은 조건으로 부른' 배경 수준이다.
재접수 여부는 사건 이후 정보로 정한 라벨이다. 모형 피처로 쓰면 안 된다.
"""
import numpy as np
import pandas as pd

RECALL_KEY = ["o_gu", "o_dong", "d_gu", "d_dong", "disability"]
RECALL_WINDOWS = (10, 30, 60)
RECALL_DEFAULT = 30


def _key(df):
    k = df[RECALL_KEY].astype(str).agg("|".join, axis=1)
    return k.where(df[RECALL_KEY].notna().all(axis=1))


def next_request_gap(anchor_df, anchor_col, pool):
    """anchor 시각 이후(초과) 같은 키로 들어온 첫 접수까지의 분과 그 콜의 row_id."""
    a = anchor_df[[anchor_col, "row_id"]].assign(key=_key(anchor_df)).dropna(subset=["key", anchor_col])
    a = a.rename(columns={"row_id": "anchor_row"}).sort_values(anchor_col)
    p = pool[["t_request", "row_id"]].assign(key=_key(pool)).dropna(subset=["key"])
    p = p.rename(columns={"row_id": "next_row", "t_request": "t_next"}).sort_values("t_next")
    m = pd.merge_asof(a, p, left_on=anchor_col, right_on="t_next", by="key",
                      direction="forward", allow_exact_matches=False)
    m["gap_min"] = (m["t_next"] - m[anchor_col]).dt.total_seconds() / 60
    return m.set_index("anchor_row")[["gap_min", "next_row"]]


def add_recall(df):
    """calls 테이블에 재접수 관련 열을 붙인다(취소 콜만 값이 있음)."""
    canc = df[df["status"].isin(["pre_cancel", "post_cancel"])]
    nx = next_request_gap(canc, "t_cancel", df)
    df["recall_gap_min"] = df["row_id"].map(nx["gap_min"])           # 취소 -> 같은 조건 새 접수(분)
    df["recall_next_row"] = df["row_id"].map(nx["next_row"]).astype("Int64")
    for w in RECALL_WINDOWS:
        df[f"recall_{w}"] = df["recall_gap_min"].le(w) & df["status"].isin(["pre_cancel", "post_cancel"])
    # 참고: 취소하기 전에 이미 같은 조건으로 새 접수를 넣은 경우(중복 접수)
    dup = next_request_gap(canc, "t_request", df)
    dup_t = df["row_id"].map(dup["gap_min"])
    life = (df["t_cancel"] - df["t_request"]).dt.total_seconds() / 60
    df["dup_before_cancel"] = dup_t.le(life) & df["status"].isin(["pre_cancel", "post_cancel"])
    return df


def baseline_false_match(df, windows=RECALL_WINDOWS):
    """승차 콜에 같은 규칙을 승차 시각 기준으로 적용한 비율(%)."""
    b = df[df["status"] == "boarded"]
    nx = next_request_gap(b, "t_board", df)
    return {w: float(nx["gap_min"].le(w).mean() * 100) for w in windows}


def add_abandon_events(df, w=RECALL_DEFAULT):
    """'최종 포기' 기준 사건 코드(재접수 취소를 사건 4로 분리).
    E1_ab   : 0 절단, 1 배차, 2 배차 전 최종 포기, 4 배차 전 재접수 취소
    E2_ab   : 0 절단, 1 승차, 2 배차 후 최종 포기, 3 기타 실패, 4 배차 후 재접수 취소
    E_all_ab: 0 절단, 1 승차, 2 최종 포기(배차 전+후), 3 기타 실패, 4 재접수 취소
    시간(T1, T2, T_all)은 그대로 쓴다."""
    r = df[f"recall_{w}"]
    df["E1_ab"] = np.where((df["E1"] == 2) & r, 4, df["E1"]).astype("int8")
    e2 = df["E2"].astype("float")
    df["E2_ab"] = pd.Series(np.where((e2 == 2) & r, 4, e2), index=df.index).astype("Int8")
    df["E_all_ab"] = np.where((df["E_all"] == 2) & r, 4, df["E_all"]).astype("int8")
    return df
