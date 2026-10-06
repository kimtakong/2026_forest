"""06·08단계 공용: 시스템 부하 피처와 5분 person-period 데이터.

누수 방지 원칙: 피처는 각 5분 구간 시작 시각 t에 알 수 있는 정보만 쓴다.
  - 부하 피처는 t가 속한 '분'의 시작 시각(<= t) 스냅숏으로 계산한다.
  - 대기 콜 수: 스냅숏 시각 tau에 '대기 시작 < tau <= 대기 종료'인 콜 수
      대기 시작 = 접수(즉시호출) 또는 예정(사전접수), 대기 종료 = 배차 또는 배차 전 취소
  - 직전 30분 건수: [tau-30분, tau) 동안의 접수·배차·하차 건수
  - 재접수 여부(recall_*), 차량구분은 Stage 1 피처로 쓰지 않는다(사후 정보·배정 결과).
"""
import numpy as np
import pandas as pd

from utils import SEOUL_GU

EPOCH = pd.Timestamp("2024-12-31 00:00:00")
N_MIN = 368 * 1440                     # 2024-12-31 ~ 2026-01-02
BIN = 5                                # 구간 폭(분)
N_BINS = 37                            # 0~35 = 0~180분, 36 = 180분 초과
LOOKBACK = 30
GU_INDEX = {g: i for i, g in enumerate(SEOUL_GU)}
LOAD_COLS = ["wait_seoul", "wait_gu", "req30_seoul", "req30_gu", "disp30_seoul", "disp30_gu", "alight30_gu"]


def _minute_idx(ts):
    return ((ts - EPOCH) // pd.Timedelta(minutes=1)).to_numpy(dtype="float64")


class LoadIndex:
    """분 단위 부하 시계열(서울 전체 + 구별 25개)."""

    def __init__(self, calls):
        gu = calls["o_gu"].astype(str).map(GU_INDEX).to_numpy(dtype="float64")
        dgu = calls["d_gu"].astype(str).map(GU_INDEX).to_numpy(dtype="float64")
        # 대기 구간
        start = calls["t_start"]
        end = calls["t_dispatch"].fillna(calls["t_cancel"].where(calls["status"] == "pre_cancel"))
        ok = end.notna() & (end >= start)
        s_idx = _minute_idx(start[ok]) + 1
        e_idx = _minute_idx(end[ok]) + 1
        g = gu[ok.to_numpy()]
        self.wait = np.zeros((26, N_MIN + 2), dtype=np.int64)     # 0~24 구, 25 서울 전체
        for row, mask in [(25, np.ones(len(s_idx), bool)), *[(i, g == i) for i in range(25)]]:
            diff = np.zeros(N_MIN + 2, dtype=np.int64)
            np.add.at(diff, np.clip(s_idx[mask], 0, N_MIN + 1).astype(int), 1)
            np.add.at(diff, np.clip(e_idx[mask], 0, N_MIN + 1).astype(int), -1)
            self.wait[row] = diff.cumsum()
        # 분당 사건 수의 누적합: cum[m] = 분 m 이전(< m)까지의 합
        self.cum = {}
        for name, ts, gidx in [("req", calls["t_request"], gu), ("disp", calls["t_dispatch"], gu),
                               ("alight", calls["t_alight"], dgu)]:
            arr = np.zeros((26, N_MIN + 1), dtype=np.int64)
            v = ts.notna().to_numpy()
            m = _minute_idx(ts[v]).astype(int)
            gi = gidx[v]
            inside = (m >= 0) & (m < N_MIN)
            np.add.at(arr[25], m[inside], 1)
            ing = inside & ~np.isnan(gi)
            np.add.at(arr, (gi[ing].astype(int), m[ing]), 1)
            c = np.zeros((26, N_MIN + 2), dtype=np.int64)
            c[:, 1:] = arr.cumsum(axis=1)
            self.cum[name] = c

    def features(self, t, gu_idx):
        """시각 t(Timestamp 배열)와 구 번호에서의 부하 피처."""
        m = np.clip(_minute_idx(pd.Series(t)).astype(int), LOOKBACK, N_MIN)
        g = np.asarray(gu_idx, dtype=int)
        past = lambda name, row: self.cum[name][row, m] - self.cum[name][row, m - LOOKBACK]
        return pd.DataFrame({
            "wait_seoul": self.wait[25, m], "wait_gu": self.wait[g, m],
            "req30_seoul": past("req", 25), "req30_gu": past("req", g),
            "disp30_seoul": past("disp", 25), "disp30_gu": past("disp", g),
            "alight30_gu": past("alight", g)})


def dong_target_encoding(train_calls, m=100):
    """학습 구간 콜만으로 출발동 비율을 구하고, 같은 구의 비율 쪽으로 수축한다(m = 사전 가중)."""
    d = train_calls.assign(ab=(train_calls["E1_ab"] == 2).astype(float),
                           d30=((train_calls["E1"] == 1) & (train_calls["T1"] < 30)).astype(float))
    gu = d.groupby("o_gu", observed=True)[["ab", "d30"]].mean()
    dg = d.groupby(["o_gu", "o_dong"], observed=True).agg(n=("ab", "size"), ab=("ab", "mean"), d30=("d30", "mean"))
    dg = dg.reset_index()
    for c in ["ab", "d30"]:
        prior = dg["o_gu"].astype(str).map(gu[c].rename(index=str)).astype(float)
        dg[f"dong_{c}"] = (dg["n"] * dg[c] + m * prior) / (dg["n"] + m)
    gu = gu.rename(columns={"ab": "dong_ab", "d30": "dong_d30"})
    return dg[["o_gu", "o_dong", "dong_ab", "dong_d30"]], gu


def apply_te(df, te, gu_prior):
    out = df[["o_gu", "o_dong"]].astype(str).merge(te.astype({"o_gu": str, "o_dong": str}), on=["o_gu", "o_dong"],
                                                     how="left")
    for c in ["dong_ab", "dong_d30"]:
        prior = df["o_gu"].astype(str).map(gu_prior[c].rename(index=str)).astype(float).to_numpy()
        out[c] = np.where(out[c].isna(), prior, out[c])
    return out[["dong_ab", "dong_d30"]].set_axis(df.index)


STATIC_COLS = ["hour", "dow", "is_holiday", "month", "o_gu", "dong_ab", "dong_d30", "purpose", "disability_grp"]
LABEL_MAP = {1: 1, 2: 2, 4: 3}         # E1_ab -> 클래스(1 배차, 2 최종 포기, 3 재접수 취소). 0 = 사건 없음
CLASS_NAMES = ["none", "배차", "최종 포기", "재접수 취소"]


def person_period(calls, load):
    """콜 x 5분 구간 행. 마지막 구간에 사건 클래스, 나머지는 0."""
    K = np.minimum(np.floor(calls["T1"].to_numpy() / BIN).astype(int), N_BINS - 1)
    rep = np.repeat(np.arange(len(calls)), K + 1)
    k = np.arange(len(rep)) - np.repeat(np.cumsum(K + 1) - (K + 1), K + 1)
    base = calls.iloc[rep].reset_index(drop=True)
    pp = base[["row_id"] + STATIC_COLS].copy()
    pp["k"] = k.astype("int16")
    ev = calls["E1_ab"].map(LABEL_MAP).fillna(0).to_numpy().astype(int)
    last = k == K[rep]
    pp["y"] = np.where(last, ev[rep], 0).astype("int8")
    t = base["t_request"] + pd.to_timedelta(k * BIN, unit="min")
    lf = load.features(t, base["o_gu"].astype(str).map(GU_INDEX).to_numpy())
    for c in LOAD_COLS:
        pp[c] = lf[c].to_numpy()
    return pp


def frozen_frame(calls, load, s_bin, n_bins):
    """예측용: s_bin 시점까지 기다린 콜에 대해 구간 s_bin..s_bin+n_bins-1 행을 만든다.
    부하 피처는 예측 시점(s_bin 시작)의 값으로 고정한다(미래 부하는 알 수 없음)."""
    t0 = calls["t_request"] + pd.to_timedelta(s_bin * BIN, unit="min")
    lf = load.features(t0, calls["o_gu"].astype(str).map(GU_INDEX).to_numpy())
    rep = np.repeat(np.arange(len(calls)), n_bins)
    fr = calls.iloc[rep][STATIC_COLS].reset_index(drop=True)
    fr["k"] = np.minimum(np.tile(np.arange(s_bin, s_bin + n_bins), len(calls)), N_BINS - 1).astype("int16")
    for c in LOAD_COLS:
        fr[c] = lf[c].to_numpy()[rep]
    return fr


def hazards_to_cif(h, n_calls, n_bins):
    """h: (n_calls*n_bins, 4) 구간 해저드(열 0 = 사건 없음). Algorithm 2.
    반환: F (n_calls, n_bins+1, 3) 원인별 누적확률(시점 0..n_bins), S (n_calls, n_bins+1)."""
    h = h.reshape(n_calls, n_bins, 4)[:, :, 1:]
    F = np.zeros((n_calls, n_bins + 1, 3))
    S = np.ones((n_calls, n_bins + 1))
    for b in range(n_bins):
        F[:, b + 1] = F[:, b] + S[:, b, None] * h[:, b]
        S[:, b + 1] = S[:, b] * (1 - h[:, b].sum(axis=1))
    return F, S
