"""08b. 08단계 확장: "제안 방식이 단순 규칙보다 나은가"를 같은 틀로 확인한다 (CLAUDE.md 9절 해석 원칙).

심사 질문: "모형 없이 접수 때 야간 호출·포기율 높은 구를 바로 안내하면 되지 않나?"
결과가 어느 쪽이든 그대로 보고한다. 08단계(A 현행, B 동적)의 계산·정의는 바꾸지 않고, 같은 테스트 호출·같은 지표로 평가한다.

① 단순 규칙과 비교(같은 평면: x = 하루 평균 표시 건수, y = 배차 전 최종 포기를 취소 전에 잡은 비율)
  R0 무작위      접수 시점에 확률 p로 무작위 표시(시드 42)
  R1 야간 일괄   접수 시각이 20~01시인 호출을 접수 시점에 전부 표시(점 하나)
  R2 룩업표      학습 기간(1~8월)의 (출발구 x 접수 시각 그룹)별 배차 전 최종 포기율, 같은 구 값 쪽으로 수축(w = n/(n+50)).
                 접수 시점에 셀 값이 τ 이상이면 표시. 그룹은 '4시간 블록 6개(20시 시작)'와 '24시간' 중 9월 검증 Brier가 낮은 쪽.
  R3 모형 1회    06단계 모형이 s=0에서 계산한 '다음 30분 내 최종 포기 확률'이 θ 이상이면 표시(재점검 없음)
  A 현행 60분 규칙(점), B 동적 규칙(5분마다 재점검, 08단계와 같은 곡선)
  θ·τ·p의 격자는 학습·검증 자료나 고정 값으로 정했고, 테스트 결과(재현율 등)를 보고 고르지 않았다.
  '같은 건수 비교'의 설정은 A의 하루 표시 건수(운영 부담)에 가장 가까운 점이다(건수만 쓰고 결과는 쓰지 않음, 08단계와 같은 방식).
② 수락률 시나리오(policy_acceptance.csv): 수락한 사람은 포기하지 않는다고 가정한 상한이며 인과가 아니다.
③ 안내 범위 검증(guidance_coverage.csv, 부록 fig_A4): 학습 기간 분위수 범위가 테스트 기간에 실제로 맞는 비율.
출력: outputs/tables/policy_baselines.csv, policy_lookup_choice.csv, policy_acceptance.csv, guidance_coverage.csv
      outputs/figures/fig_07_policy.png(기준선 추가판, 08이 만든 판을 덮어씀), fig_A4_guidance_coverage.png
"""
import importlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import features as F
from utils import (COLOR, DAY_HOURS, FS, MODELS, NIGHT_HOURS, OUT_TAB, SEED, apply_style, clean, draw_figures, fig_title,
                   fig_width, save_fig, save_table)

p08 = importlib.import_module("08_policy_simulation")     # 같은 테스트 호출·같은 지표 함수를 그대로 쓴다
hm = p08.hm

DC = "하루 평균 전환 대상(건)"
RC = "재현율%(배차 전 최종 포기를 취소 전에 잡음)"
LC = "미리 잡은 시간 중앙값(분)"
PC = "정밀도%(전환 대상 중 배차 전 최종 포기)"
UC = "불필요 전환%(전환 후 10분 안에 배차)"
SHRINK_M = 50
R0_GRID = np.round(np.arange(0.005, 0.5001, 0.005), 3)
R3_GRID = np.round(np.arange(0.01, 0.9001, 0.005), 3)
ACCEPT = (10, 20, 30, 50)
CAPS = (50, 100, 200)
NAME_A, NAME_R0, NAME_R1 = "A 현행 60분 규칙", "R0 무작위", "R1 야간 일괄"
NAME_R2, NAME_R3, NAME_B = "R2 룩업표(구×접수 시각)", "R3 모형, 접수 시점 1회", "B 동적 규칙(5분 재점검)"
NOTE_UB = "수락한 사람은 포기하지 않는다고 가정한 상한 계산이며 인과 효과가 아님"


# ------------------------------------------------------------------ R2 룩업표
def g6(h):
    """20시 시작 4시간 블록 6개: 20~23, 0~3, 4~7, 8~11, 12~15, 16~19시."""
    return ((np.asarray(h, dtype=int) - 20) % 24) // 4


def g24(h):
    return np.asarray(h, dtype=int)


def lookup_table(train, fn, m=SHRINK_M):
    d = train[train.E1_ab != 0]
    y = (d.E1_ab == 2).to_numpy(float)
    gu = d.o_gu.astype(str).to_numpy()
    gu_rate = pd.Series(y).groupby(gu).mean()
    cell = pd.DataFrame({"gu": gu, "g": fn(d.t_request.dt.hour), "y": y}).groupby(["gu", "g"])["y"].agg(["sum", "count"])
    prior = gu_rate.reindex(cell.index.get_level_values(0)).to_numpy()
    rate = (cell["sum"].to_numpy() + m * prior) / (cell["count"].to_numpy() + m)     # = w x 셀 값 + (1-w) x 구 값
    return pd.Series(rate, index=cell.index), gu_rate, cell["count"]


def lookup_score(d, table, gu_rate, fn):
    gu = d.o_gu.astype(str)
    idx = pd.MultiIndex.from_arrays([gu.to_numpy(), fn(d.t_request.dt.hour)])
    s = table.reindex(idx).to_numpy()
    return np.where(np.isnan(s), gu.map(gu_rate).to_numpy(float), s)


def choose_lookup(train, val):
    """9월 검증 Brier가 낮은 그룹 방식을 고른다."""
    yv = (val.E1_ab == 2).to_numpy(float)
    res = []
    for name, fn in [("6그룹(4시간 블록)", g6), ("24시간", g24)]:
        table, gu_rate, cnt = lookup_table(train, fn)
        s = lookup_score(val, table, gu_rate, fn)
        res.append({"그룹 방식": name, "셀 수": len(table), "학습 셀 콜 수 최소": int(cnt.min()),
                    "학습 셀 콜 수 중앙값": int(cnt.median()), "검증(9월) Brier": float(np.mean((s - yv) ** 2)),
                    "검증(9월) AUC": float(roc_auc_score(yv, s)), "fn": fn, "table": table, "gu_rate": gu_rate})
    best = min(res, key=lambda r: r["검증(9월) Brier"])
    out = pd.DataFrame([{k: v for k, v in r.items() if k not in ("fn", "table", "gu_rate")} for r in res])
    out["선택"] = out["그룹 방식"] == best["그룹 방식"]
    out["비고"] = f"학습 1~8월 {int((train.E1_ab != 0).sum()):,}콜로 표 작성, 수축 w = n/(n+{SHRINK_M}), 검증 9월 {len(val):,}콜"
    save_table(out.round(5), "policy_lookup_choice.csv")
    print(out.round(4).to_string(index=False))
    return best


# ------------------------------------------------------------------ 곡선·매칭
def sweep(rule, values, flag_fn, test, groups, n_days):
    rows = []
    for v in values:
        f = flag_fn(v)
        for g, m in groups.items():
            rows.append({"규칙": rule, "집단": g, "설정값": v, **p08.metrics(test, f, m, n_days)})
    return pd.DataFrame(rows)


def pick(cv, group, target):
    c = cv[cv["집단"] == group]
    return c.iloc[int((c[DC] - target).abs().to_numpy().argmin())]


def fmt_row(section, g, rule, setting, v, mt):
    return {"구분": section, "집단": g, "규칙": rule, "설정": setting, "설정값": v, DC: mt[DC], RC: mt[RC], LC: mt[LC],
            PC: mt[PC], UC: mt[UC], "전환 대상 콜 수": mt["전환 대상 콜 수"], "배차 전 최종 포기 콜 수": mt["배차 전 최종 포기 콜 수"]}


# ------------------------------------------------------------------ ② 하루 상한 N
def topn_static(test, score, N, rng):
    """점수가 접수 시점에 정해지는 규칙(R2): 하루 안에서 점수 상위 N건을 접수 시점에 표시."""
    d = pd.DataFrame({"day": test.t_request.dt.normalize().to_numpy(), "s": score, "r": rng.random(len(test))})
    d = d.sort_values(["day", "s", "r"], ascending=[True, False, True], kind="stable")
    d["rk"] = d.groupby("day").cumcount()
    flag = np.full(len(test), np.nan)
    flag[d.index[d.rk < N].to_numpy()] = 0.0
    return flag


def topn_dynamic(chk, test, N):
    """규칙 B: 하루 안에서 호출별 최대 위험(점검 중 최대 p)이 높은 N건을 고르고, 그 호출의 일일 컷오프를 처음 넘은 점검 시점에 표시."""
    peak = chk.groupby("i")["p_ab30"].max()
    d = pd.DataFrame({"i": peak.index.to_numpy(), "peak": peak.to_numpy()})
    d["day"] = test.t_request.dt.normalize().to_numpy()[d["i"].to_numpy()]
    d = d.sort_values(["day", "peak", "i"], ascending=[True, False, True], kind="stable")
    d["rk"] = d.groupby("day").cumcount()
    sel = d[d.rk < N]
    cut = sel.groupby("day")["peak"].transform("min")
    th = pd.Series(cut.to_numpy(), index=sel["i"].to_numpy())
    c = chk[chk["i"].isin(th.index)]
    c = c[c["p_ab30"].to_numpy() >= c["i"].map(th).to_numpy()]
    u, idx = np.unique(c["i"].to_numpy(), return_index=True)
    flag = np.full(len(test), np.nan)
    flag[u] = c["b"].to_numpy()[idx] * F.BIN
    return flag


def topn_rule_a(test, flag_a, mask, n_days):
    """규칙 A: 표시 대상은 모두 60분 시점이라 점수 차이가 없다. 하루 안에서 무작위로 N건을 고른 기댓값."""
    T, E = test.T1.to_numpy(), test.E1_ab.to_numpy()
    day = test.t_request.dt.normalize().to_numpy()
    fl = ~np.isnan(flag_a) & mask
    caught = fl & (E == 2) & (flag_a < T)
    d = pd.DataFrame({"day": day, "fl": fl, "caught": caught, "ab_fl": fl & (E == 2)})
    g = d.groupby("day").sum()
    return g, (E == 2)[mask].sum()


def a_cap(g, ab_total, N, n_days):
    keep = np.minimum(1.0, N / g["fl"].clip(lower=1))
    caught = (g["caught"] * keep).sum()
    flagged = np.minimum(g["fl"], N).sum()
    return {DC: flagged / n_days, RC: caught / ab_total * 100,
            PC: (g["ab_fl"] * keep).sum() / max(flagged, 1) * 100, "하루 잡은 최종 포기(건)": caught / n_days}


# ------------------------------------------------------------------ ③ 안내 범위
def rem_quantiles(d, s):
    w = d[(d.E1_ab == 1) & (np.floor(d.T1) >= s)]
    r = w.T1 - s
    return float(r.quantile(.75)), float(r.quantile(.90)), len(w)


def guidance_coverage(imm):
    train = imm[imm.split == "train"]
    test = imm[(imm.split == "test") & (imm.E1_ab != 0)]
    groups = {"주간(10~14시)": DAY_HOURS, "야간(20~01시)": NIGHT_HOURS}
    hr = lambda d: d.t_request.dt.hour
    rows = []
    for s in (0, 15, 30):
        pooled = rem_quantiles(train, s)
        for g, hours in groups.items():
            q = rem_quantiles(train[hr(train).isin(hours)], s)
            qy = rem_quantiles(imm[hr(imm).isin(hours)], s)             # 보고서 표(연간)와 같은 정의, 참고
            wait = test[hr(test).isin(hours) & (np.floor(test.T1) >= s)]
            rem = (wait[wait.E1_ab == 1].T1 - s)
            n_ab = int((wait.E1_ab == 2).sum())
            for k, (target, name) in enumerate([(75, "대부분(75% 분위)"), (90, "늦어도(90% 분위)")]):
                own, one = q[k], pooled[k]
                rows.append({
                    "집단": g, "이미 기다린 시간 s(분)": s, "안내 범위": name, "목표 적중률%": target,
                    "시간대별 범위(분, 학습 1~8월)": own, "하나의 범위(분, 학습 1~8월 전체 시간대)": one,
                    "(참고) 연간 시간대별 분위(보고서 표)": qy[k],
                    "테스트: s분 시점 미배차 콜 중 결국 배차된 콜 수": len(rem),
                    "적중률%: 시간대별 범위": (rem <= own).mean() * 100,
                    "적중률%: 하나의 범위": (rem <= one).mean() * 100,
                    "배차 전 최종 포기 콜 수(범위 밖으로 셈)": n_ab,
                    "보수적 적중률%: 시간대별 범위": (rem <= own).sum() / (len(rem) + n_ab) * 100,
                    "보수적 적중률%: 하나의 범위": (rem <= one).sum() / (len(rem) + n_ab) * 100})
    t = pd.DataFrame(rows).round(2)
    t["비고"] = ("적중률 = s분 시점에 아직 미배차인 테스트 호출 중 결국 배차된 호출만으로, 남은 배차 시간이 안내 범위 이하였던 비율. "
                "보수적 = 같은 시점에 미배차였다가 배차 전 최종 포기한 호출(재접수 취소는 제외)을 범위 밖으로 분모에 넣은 값")
    save_table(t, "guidance_coverage.csv")
    show = t[["집단", "이미 기다린 시간 s(분)", "안내 범위", "시간대별 범위(분, 학습 1~8월)", "하나의 범위(분, 학습 1~8월 전체 시간대)",
              "(참고) 연간 시간대별 분위(보고서 표)", "적중률%: 시간대별 범위", "적중률%: 하나의 범위",
              "보수적 적중률%: 시간대별 범위", "보수적 적중률%: 하나의 범위"]]
    print(show.to_string(index=False))
    return t


def fig_coverage(t):
    import matplotlib.pyplot as plt
    apply_style()
    ann = not clean()
    fig, axes = plt.subplots(1, 2, figsize=(fig_width(), 3.3 if ann else 2.9), sharey=True, gridspec_kw={"wspace": 0.16})
    if ann:
        fig.subplots_adjust(left=0.2, right=0.985, top=0.72, bottom=0.2)
    else:
        fig.subplots_adjust(left=0.27, right=0.985, top=0.77, bottom=0.2)
    labs, ys = [], []
    order = [(g, s) for g in ["주간(10~14시)", "야간(20~01시)"] for s in (0, 15, 30)]
    for k, name in enumerate(["대부분(75% 분위)", "늦어도(90% 분위)"]):
        ax = axes[k]
        d = t[t["안내 범위"] == name]
        target = d["목표 적중률%"].iloc[0]
        for j, (g, s) in enumerate(order):
            r = d[(d["집단"] == g) & (d["이미 기다린 시간 s(분)"] == s)].iloc[0]
            y = len(order) - 1 - j
            ax.plot([r["적중률%: 하나의 범위"], r["적중률%: 시간대별 범위"]], [y, y], color=COLOR["axis"], lw=1.2, zorder=1)
            ax.plot(r["적중률%: 하나의 범위"], y, "o", ms=6, color=COLOR["ink2"], mec=COLOR["surface"], mew=1, zorder=3,
                    label="하나의 범위(모두에게 같은 안내)" if j == 0 else None)
            ax.plot(r["적중률%: 시간대별 범위"], y, "o", ms=6, color=COLOR["board"], mec=COLOR["surface"], mew=1, zorder=3,
                    label="시간대별 범위(주간·야간 따로)" if j == 0 else None)
            if k == 0:
                labs.append((y, f"{g.split('(')[0]} · " + ("접수 직후" if s == 0 else f"{s}분째 미배차")))
        ax.axvline(target, color=COLOR["ink"], lw=0.9, zorder=2)
        ax.set_xlim(40, 100)
        ax.set_xlabel("범위 안에 실제로 배차된 비율(%)")
        ax.set_title(["(가) '대부분' 범위", "(나) '늦어도' 범위"][k], loc="left", fontsize=8.5 if ann else None, pad=4)
        ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)
    axes[0].set_yticks([y for y, _ in labs], [l for _, l in labs])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.2 if ann else 0.27, 0.9 if ann else 1.0), ncol=2,
               handlelength=1.0, columnspacing=1.4, fontsize=7 if ann else None)
    ex = t[(t["집단"] == "야간(20~01시)") & (t["이미 기다린 시간 s(분)"] == 0) & (t["안내 범위"] == "대부분(75% 분위)")].iloc[0]
    fig_title(fig, "야간에 주간과 같은 범위를 안내하면 범위가 자주 빗나간다",
              f"접수 직후 '대부분' 범위(목표 75%)의 실제 적중률: 야간 {ex['적중률%: 시간대별 범위']:.0f}%(시간대별 범위) vs "
              f"{ex['적중률%: 하나의 범위']:.0f}%(하나의 범위). 세로선 = 목표 적중률", y_sub=0.9)
    note = ("주: 학습 기간(1~8월)에서 s분 시점에 아직 미배차였다가 결국 배차된 호출의 남은 배차 시간 75%·90% 분위수를 안내 범위로 두고,\n"
            "    테스트 기간(10~12월)의 같은 조건 호출 중 범위 안에 배차된 비율을 계산. 결국 배차된 호출만 센 값이며, 배차 전에 최종 포기한 호출을 범위 밖으로 세면\n"
            "    더 낮아짐(guidance_coverage.csv의 보수적 열). 하나의 범위 = 같은 s에서 시간대 구분 없이 구한 분위수.")
    save_fig(fig, "fig_A4_guidance_coverage.png", note=note)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    imm, load = hm.prepare()
    bst = lgb.Booster(model_str=(MODELS / "lgbm_stage1.txt").read_text(encoding="utf-8"))
    feats = bst.feature_name()
    train = imm[imm.split == "train"]
    val = imm[(imm.split == "val") & (imm.E1_ab != 0)]
    test = imm[(imm.split == "test") & (imm.E1_ab != 0)].reset_index(drop=True)
    n, n_days = len(test), test.t_request.dt.normalize().nunique()
    night = test.t_request.dt.hour.isin(NIGHT_HOURS).to_numpy()
    groups = {"전체": np.ones(n, bool), "야간(20~01시)": night}
    G_ALL, G_NIGHT = "전체", "야간(20~01시)"
    rng = np.random.default_rng(SEED)

    # --- B(08단계와 같은 계산): 점검 시점별 예측
    chk = p08.score_checks(bst, feats, test, load)
    print(f"  점검 시점 {len(chk):,}개 예측 ({time.time() - t0:.0f}초)")
    flag_a = np.where(test.T1.to_numpy() >= p08.RULE_A_MIN, float(p08.RULE_A_MIN), np.nan)
    a_m = {g: p08.metrics(test, flag_a, m, n_days) for g, m in groups.items()}

    # --- 각 규칙의 접수 시점 점수
    best = choose_lookup(train, val)
    score_r2 = lookup_score(test, best["table"], best["gu_rate"], best["fn"])
    p0 = np.full(n, np.nan)
    c0 = chk[chk["b"] == 0]
    p0[c0["i"].to_numpy()] = c0["p_ab30"].to_numpy()
    u_rand = rng.random(n)
    flag_r1 = np.where(night, 0.0, np.nan)

    cv = {
        NAME_R0: sweep(NAME_R0, R0_GRID, lambda p: np.where(u_rand < p, 0.0, np.nan), test, groups, n_days),
        NAME_R2: sweep(NAME_R2, np.unique(np.round(score_r2, 6)), lambda t: np.where(score_r2 >= t - 1e-9, 0.0, np.nan),
                       test, groups, n_days),
        NAME_R3: sweep(NAME_R3, R3_GRID, lambda th: np.where(p0 >= th, 0.0, np.nan), test, groups, n_days),
        NAME_B: sweep(NAME_B, p08.THETAS, lambda th: p08.first_flag(chk, th, n), test, groups, n_days),
    }
    flag_fn = {NAME_R0: lambda p: np.where(u_rand < p, 0.0, np.nan),
               NAME_R2: lambda t: np.where(score_r2 >= t - 1e-9, 0.0, np.nan),
               NAME_R3: lambda th: np.where(p0 >= th, 0.0, np.nan),
               NAME_B: lambda th: p08.first_flag(chk, th, n)}
    r1_m = {g: p08.metrics(test, flag_r1, m, n_days) for g, m in groups.items()}
    print(f"  곡선 계산 완료 ({time.time() - t0:.0f}초)")

    # --- 같은 건수(A의 전체 하루 건수) 비교
    tgt = a_m[G_ALL][DC]
    rows, matched = [], {}
    S1 = "1 같은 하루 건수 비교(전체에서 A와 맞춤)"
    for g in groups:
        rows.append(fmt_row(S1, g, NAME_A, "60분 경과 & 미배차", np.nan, a_m[g]))
    for name, cvv in cv.items():
        sel = pick(cvv, G_ALL, tgt)
        matched[name] = sel["설정값"]
        for g in groups:
            r = cvv[(cvv["집단"] == g) & (cvv["설정값"] == sel["설정값"])].iloc[0]
            rows.append(fmt_row(S1, g, name, {NAME_R0: "표시 확률 p", NAME_R2: "셀 포기율 τ", NAME_R3: "θ", NAME_B: "θ"}[name],
                                sel["설정값"], r))
    for g in groups:
        rows.append(fmt_row(S1, g, NAME_R1, "접수 20~01시 전부(건수를 맞출 수 없어 실제 건수)", np.nan, r1_m[g]))

    # --- 야간만 운영: 야간 안에서 A의 야간 건수에 맞춤
    S2 = "2 야간만 운영(야간 안에서 A의 야간 건수와 맞춤)"
    tgt_n = a_m[G_NIGHT][DC]
    rows.append(fmt_row(S2, G_NIGHT, NAME_A, "60분 경과 & 미배차", np.nan, a_m[G_NIGHT]))
    night_sel = {}
    for name, cvv in cv.items():
        sel = pick(cvv, G_NIGHT, tgt_n)
        night_sel[name] = sel["설정값"]
        rows.append(fmt_row(S2, G_NIGHT, name, {NAME_R0: "표시 확률 p", NAME_R2: "셀 포기율 τ", NAME_R3: "θ", NAME_B: "θ"}[name],
                            sel["설정값"], sel))
    rows.append(fmt_row(S2, G_NIGHT, NAME_R1, "접수 20~01시 전부(건수를 맞출 수 없어 실제 건수)", np.nan, r1_m[G_NIGHT]))

    # --- 곡선 전체
    S3 = "3 곡선(설정 전체)"
    for name, cvv in cv.items():
        for _, r in cvv.iterrows():
            rows.append(fmt_row(S3, r["집단"], name, "곡선", r["설정값"], r))
    out = pd.DataFrame(rows)
    for c in (DC, RC, LC, PC, UC):
        out[c] = out[c].round(3)
    out["설정값"] = out["설정값"].round(5)
    save_table(out, "policy_baselines.csv")

    pd.set_option("display.width", 250)
    show = out[out["구분"].str[0].isin(["1", "2"])][["구분", "집단", "규칙", "설정값", DC, RC, LC, PC, UC]]
    print(show.round(2).to_string(index=False))

    # --- 기존 숫자 불변 확인: B(같은 건수)가 policy_compare.csv의 기존 행과 같은가
    pc = pd.read_csv(OUT_TAB / "policy_compare.csv", encoding="utf-8-sig")
    base = pc[(pc["구분"].str[0] == "1") & pc["규칙"].str.startswith("B")]
    th_old = float(base["θ"].iloc[0])
    assert abs(matched[NAME_B] - th_old) < 1e-9, f"θ 불일치: {matched[NAME_B]} vs {th_old}"
    for g in groups:
        bn = fmt_row("", g, "B", "", 0, cv[NAME_B][(cv[NAME_B]["집단"] == g) & (cv[NAME_B]["설정값"] == th_old)].iloc[0])
        old = base[base["집단"] == g].iloc[0]
        for c in (DC, RC, LC, PC, UC):
            assert abs(round(bn[c], 3) - old[c]) < 1e-6, f"기존 값 불일치 {g} {c}: {bn[c]} vs {old[c]}"
    print("  기존 08 숫자(B, θ = %.3f) 일치 확인" % th_old)

    # --- ② 수락률·하루 상한
    S_A = "1 수락률 시나리오(하루 건수를 A와 맞춤)"
    acc = []
    sets = {NAME_A: lambda g: a_m[g],
            NAME_B: lambda g: cv[NAME_B][(cv[NAME_B]["집단"] == g) & (cv[NAME_B]["설정값"] == matched[NAME_B])].iloc[0],
            NAME_R2: lambda g: cv[NAME_R2][(cv[NAME_R2]["집단"] == g) & (cv[NAME_R2]["설정값"] == matched[NAME_R2])].iloc[0]}
    for rule, getter in sets.items():
        for g in groups:
            m = getter(g)
            caught_day = m[RC] / 100 * m["배차 전 최종 포기 콜 수"] / n_days
            for a in ACCEPT:
                acc.append({"구분": S_A, "규칙": rule, "집단": g, "수락률%": a, DC: m[DC], "하루 잡은 최종 포기(건)": caught_day,
                            "하루 구제 상한(건)": caught_day * a / 100, "연간 구제 상한(건, 하루×365)": caught_day * a / 100 * 365,
                            "테스트 92일 합(건)": caught_day * a / 100 * n_days, "비고": NOTE_UB})
    S_N = "2 하루 전환 상한 N(하루 안에서 점수 상위 N건만 전환)"
    g_a = {g: topn_rule_a(test, flag_a, m, n_days) for g, m in groups.items()}
    for N in CAPS:
        for g, m in groups.items():
            ga, ab_total = g_a[g]
            r = a_cap(ga, ab_total, N, n_days)
            acc.append({"구분": S_N, "규칙": NAME_A + "(무작위 N건 기댓값)", "집단": g, "하루 전환 상한 N": N, DC: r[DC],
                        RC: r[RC], PC: r[PC], "하루 잡은 최종 포기(건)": r["하루 잡은 최종 포기(건)"],
                        "비고": "A는 표시 대상이 모두 60분 시점이라 점수 차이가 없어 하루 안 무작위 N건의 기댓값"})
        fb = topn_dynamic(chk, test, N)
        fr2 = topn_static(test, score_r2, N, np.random.default_rng(SEED))
        for rule, fl in [(NAME_B, fb), (NAME_R2, fr2)]:
            for g, m in groups.items():
                mt = p08.metrics(test, fl, m, n_days)
                acc.append({"구분": S_N, "규칙": rule, "집단": g, "하루 전환 상한 N": N, DC: mt[DC], RC: mt[RC], PC: mt[PC],
                            LC: mt[LC], "하루 잡은 최종 포기(건)": mt[RC] / 100 * mt["배차 전 최종 포기 콜 수"] / n_days,
                            "비고": "하루 안에서 점수 순위로 사후에 N건을 골랐으므로 운영에서 얻을 값의 상한에 가까움"})
    acc = pd.DataFrame(acc)
    for c in acc.select_dtypes("number").columns:
        acc[c] = acc[c].round(3)
    save_table(acc, "policy_acceptance.csv")
    m1 = (acc["구분"].str[0] == "1") & (acc["집단"] == G_ALL)
    print(acc[m1][["규칙", "수락률%", DC, "하루 잡은 최종 포기(건)", "하루 구제 상한(건)", "연간 구제 상한(건, 하루×365)"]].to_string(index=False))
    print(acc[acc["구분"].str[0] == "2"][["규칙", "집단", "하루 전환 상한 N", DC, RC, PC]].round(2).to_string(index=False))

    # --- ③ 안내 범위
    cov = guidance_coverage(imm)
    draw_figures(fig_coverage, cov)

    # --- fig_07 기준선 추가판(08이 만든 그림을 덮어씀)
    g_all = lambda name: cv[name][cv[name]["집단"] == G_ALL][[DC, RC]]
    pick_rc = lambda name: float(cv[name][(cv[name]["집단"] == G_ALL) & (cv[name]["설정값"] == matched[name])][RC].iloc[0])
    base_fig = {"R0": g_all(NAME_R0), "R2": g_all(NAME_R2), "R3": g_all(NAME_R3), "R1": (r1_m[G_ALL][DC], r1_m[G_ALL][RC]),
                "match": {"R2": pick_rc(NAME_R2), "R3": pick_rc(NAME_R3)}}
    curve = pd.read_csv(OUT_TAB / "policy_theta_curve.csv", encoding="utf-8-sig")
    draw_figures(p08.fig_policy, test, curve, a_m[G_ALL], a_m[G_NIGHT], matched[NAME_B], flag_a,
                 p08.first_flag(chk, matched[NAME_B], n), groups, base_fig)
    print(f"[08b_policy_baselines] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
