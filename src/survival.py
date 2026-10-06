"""비모수 생존·경쟁위험 추정(1분 이산시간). 04·05·07단계가 함께 쓴다.

n(k)     = 구간 k 시작 시점에 남아 있는 콜 수 (floor(T) >= k)
h_j(k)   = d_j(k) / n(k)                    원인 j의 구간 해저드
S(k+1)   = S(k) * (1 - sum_j h_j(k))        전체 생존(아직 대기 중)
F_j(k+1) = F_j(k) + S(k) * h_j(k)           원인 j의 누적발생함수(Aalen-Johansen)
naive    : S_b(k+1) = S_b(k) * (1 - h_1(k)), 원인 1만 사건으로 본 KM의 1 - S_b
같은 구간의 중도절단은 사건 뒤에 일어난 것으로 둔다(표준 관례).
lifelines의 AalenJohansenFitter는 동률 시각을 무작위로 흔들어 처리하므로 쓰지 않는다.
"""
import numpy as np
import pandas as pd

T_MAX = 180


def discrete_cif(T, E, causes, t_max=T_MAX):
    """반환: t=0..t_max(구간 시작 시점)의 위험집합, S, naive 승차(원인 1) 누적확률, 원인별 F_j."""
    T = np.asarray(T, dtype=float)
    E = np.asarray(E)
    k = np.minimum(np.floor(T).astype(int), t_max + 1)            # t_max 이후는 한 칸에 모은다
    n_bins = t_max + 2
    at_risk = np.bincount(k, minlength=n_bins)[::-1].cumsum()[::-1]
    d = {c: np.bincount(k[E == c], minlength=n_bins) for c in causes}
    S = np.ones(t_max + 1)
    F = {c: np.zeros(t_max + 1) for c in causes}
    Sb = np.ones(t_max + 1)
    for j in range(t_max):
        n = at_risk[j]
        h = {c: d[c][j] / n if n else 0.0 for c in causes}
        for c in causes:
            F[c][j + 1] = F[c][j] + S[j] * h[c]
        S[j + 1] = S[j] * (1 - sum(h.values()))
        Sb[j + 1] = Sb[j] * (1 - h.get(1, 0.0))
    out = pd.DataFrame({"t": np.arange(t_max + 1), "위험집합": at_risk[: t_max + 1], "S": S, "naive_승차": 1 - Sb})
    for c in causes:
        out[f"F_{c}"] = F[c]
    return out


def conditional(curve, s, m, causes):
    """이미 s분 기다린(구간 s 시작 시점까지 사건 없음) 콜이 다음 m분 안에 원인 j로 끝날 확률.
    P_j = (F_j(s+m) - F_j(s)) / S(s). 남은 확률 = S(s+m) / S(s)."""
    c = curve.set_index("t")
    out = {f"P_{j}": (c.loc[s + m, f"F_{j}"] - c.loc[s, f"F_{j}"]) / c.loc[s, "S"] for j in causes}
    out["P_대기중"] = c.loc[s + m, "S"] / c.loc[s, "S"]
    sb = 1 - c["naive_승차"]
    out["naive_P_1"] = 1 - sb.loc[s + m] / sb.loc[s]
    out["위험집합"] = int(c.loc[s, "위험집합"])
    return out


def conditional_time_to(curve, s, cause, level=0.5):
    """s분 기다린 콜의 원인 cause 조건부 누적확률이 level에 처음 닿는 남은 시간(분). 닿지 않으면 NaN."""
    c = curve.set_index("t")
    f = (c[f"F_{cause}"] - c.loc[s, f"F_{cause}"]) / c.loc[s, "S"]
    hit = f.loc[s:][f.loc[s:] >= level]
    return float(hit.index[0] - s) if len(hit) else np.nan
