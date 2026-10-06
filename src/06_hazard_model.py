"""06. Stage 1(접수 -> 배차) 이산시간 원인별 해저드 모형 (CLAUDE.md 5-1절).

데이터  : 즉시호출, 5분 person-period(0~180분 36구간 + 180분 초과 1구간). features.py
라벨    : 0 사건 없음 / 1 배차 / 2 최종 포기(배차 전) / 3 재접수 취소(배차 전). Stage 1에는 기타 실패가 없다.
          재접수 여부는 라벨에만 쓰고 피처로 쓰지 않는다. 차량구분도 Stage 1 피처에서 뺀다(배정 결과라 누수).
피처    : 경과 구간 k, 접수 시각·요일·공휴일(·월), 출발구, 출발동 목표 인코딩(학습 구간만), 이용목적, 장애유형,
          부하(구간 시작 시점의 서울·같은 구 미배차 대기 콜 수, 직전 30분 접수·배차 건수, 같은 구 하차 건수)
분할    : 학습 1~8월, 검증 9월(조기 종료·모형 선택), 테스트 10~12월(전체)
월 피처 : 테스트 월은 학습에 없으므로, 월을 넣은 모형과 뺀 모형을 검증 손실로 골라 쓴다.
평가    : 테스트 콜마다 접수 시점(s=0) 피처로 120분까지의 원인별 CIF를 예측(미래 부하는 알 수 없어 접수 시점 값으로 고정).
          30·60분 원인별 AUC, Brier, 5~120분 integrated Brier score, 십분위 보정도.
기준선  : (a) 경과 구간 x 접수 시각만 쓴 경험적 해저드  (b) 승차 완료 건만 학습한 RandomForest 회귀(calltaxi-DA 방식)
SHAP    : LightGBM 내장 TreeSHAP(pred_contrib). 배차·최종 포기 클래스별 평균 |SHAP| 상위 10개.
출력    : models/lgbm_stage1.txt, outputs/tables/model_metrics.csv, calibration_deciles.csv, rf_underprediction.csv,
          shap_top10.csv, case_examples.csv, outputs/figures/fig_05_shap.png, fig_A2_calibration.png
--sample: 학습·검증·테스트 콜을 (월 x 시 x 출발구) 층화로 10%만 써서 빠르게 시연
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import features as F
from utils import (CALLS, COLOR, FIG_WIDTH_IN, MODELS, SEED, apply_style, load_counts, parse_sample_flag, save_fig,
                   save_table)

SAMPLE = parse_sample_flag()
H_EVAL = 24                     # 평가 구간 수(24 x 5분 = 120분)
T_EVAL = (30, 60)
CAUSE_NAMES = {1: "배차", 2: "최종 포기", 3: "재접수 취소"}
CAT_COLS = ["o_gu", "purpose", "disability_grp"]
BASE_FEATS = ["k", "hour", "dow", "is_holiday", "o_gu", "dong_ab", "dong_d30", "purpose", "disability_grp"] + F.LOAD_COLS
FEAT_LABEL = {"k": "경과 시간(5분 구간)", "hour": "접수 시각", "dow": "요일", "is_holiday": "공휴일", "month": "월",
              "o_gu": "출발구", "dong_ab": "출발동 포기율*", "dong_d30": "출발동 30분 배차율*",
              "purpose": "이용목적", "disability_grp": "장애유형", "wait_seoul": "서울 미배차 대기 콜 수",
              "wait_gu": "같은 구 미배차 대기 콜 수", "req30_seoul": "서울 직전30분 접수", "req30_gu": "같은 구 직전30분 접수",
              "disp30_seoul": "서울 직전30분 배차", "disp30_gu": "같은 구 직전30분 배차",
              "alight30_gu": "같은 구 직전30분 하차"}
TITLE_SHAP = "배차는 대기 콜 수(부하)가, 최종 포기는 부하에 더해 출발동·접수 시각·장애유형이 크게 좌우한다"
PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED,
              num_threads=16, verbose=-1)


def stratified_sample(d, frac):
    return d.groupby(["month", "hour", "o_gu"], observed=True, group_keys=False).sample(frac=frac, random_state=SEED)


def prepare():
    calls = pd.read_parquet(CALLS)
    load = F.LoadIndex(calls)
    imm = calls[calls.call_type == "immediate"].reset_index(drop=True)
    assert len(imm) == load_counts()["imm_n"]
    imm["is_holiday"] = imm["is_holiday"].astype("int8")
    split = np.select([imm.month <= 8, imm.month == 9], ["train", "val"], "test")
    imm["split"] = split
    te, gu_prior = F.dong_target_encoding(imm[imm.split == "train"])
    imm[["dong_ab", "dong_d30"]] = F.apply_te(imm, te, gu_prior)
    if SAMPLE:
        imm = stratified_sample(imm, 0.1).reset_index(drop=True)
    return imm, load


def train_models(pp_tr, pp_va):
    out = {}
    for name, feats in [("LGBM(월 포함)", BASE_FEATS + ["month"]), ("LGBM(월 제외)", BASE_FEATS)]:
        t0 = time.time()
        dtr = lgb.Dataset(pp_tr[feats], pp_tr["y"], categorical_feature=CAT_COLS, free_raw_data=False)
        dva = lgb.Dataset(pp_va[feats], pp_va["y"], categorical_feature=CAT_COLS, reference=dtr)
        bst = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        loss = bst.best_score["val"]["multi_logloss"]
        out[name] = (bst, feats, loss, time.time() - t0)
        print(f"  {name}: 검증 logloss {loss:.5f}, 반복 {bst.best_iteration}, {time.time() - t0:.0f}초")
    return out


def empirical_hazard(pp_tr):
    """기준선 (a): 경과 구간 x 접수 시각별 원인별 구간 해저드."""
    g = pp_tr.groupby(["k", "hour"])["y"]
    n = g.size()
    h = {j: (pp_tr.assign(e=pp_tr.y == j).groupby(["k", "hour"])["e"].sum() / n) for j in (1, 2, 3)}
    tab = pd.DataFrame(h).reindex(pd.MultiIndex.from_product([range(F.N_BINS), range(24)], names=["k", "hour"]))
    return tab.fillna(0.0)


def predict_cif_model(bst, feats, calls, load, s_bin=0, n_bins=H_EVAL):
    fr = F.frozen_frame(calls, load, s_bin, n_bins)
    h = bst.predict(fr[feats], num_iteration=bst.best_iteration)
    return F.hazards_to_cif(h, len(calls), n_bins)


def predict_cif_empirical(tab, calls, n_bins=H_EVAL):
    k = np.tile(np.arange(n_bins), len(calls))
    hr = np.repeat(calls["hour"].to_numpy(), n_bins)
    hz = tab.loc[list(zip(k, hr))].to_numpy()
    h = np.column_stack([1 - hz.sum(axis=1), hz])
    return F.hazards_to_cif(h, len(calls), n_bins)


def observed(calls, n_bins=H_EVAL):
    """O[i, b, j] = 시점 5b분 전에 원인 j가 일어났는지."""
    t = np.arange(n_bins + 1) * F.BIN
    cls = calls["E1_ab"].map(F.LABEL_MAP).fillna(0).to_numpy().astype(int)
    T = calls["T1"].to_numpy()
    O = np.zeros((len(calls), n_bins + 1, 3), dtype=bool)
    for j in (1, 2, 3):
        O[:, :, j - 1] = (cls[:, None] == j) & (T[:, None] < t[None, :])
    return O


def evaluate(preds, O):
    rows, calib = [], []
    for name, (Fp, _) in preds.items():
        for j in (1, 2, 3):
            for t in T_EVAL:
                b = t // F.BIN
                p, o = Fp[:, b, j - 1], O[:, b, j - 1]
                rows.append({"모형": name, "원인": CAUSE_NAMES[j], "지표": f"AUC@{t}분", "값": roc_auc_score(o, p)})
                rows.append({"모형": name, "원인": CAUSE_NAMES[j], "지표": f"Brier@{t}분", "값": np.mean((p - o) ** 2)})
                if j in (1, 2):
                    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
                    c = pd.DataFrame({"dec": dec, "p": p, "o": o}).groupby("dec").agg(예측=("p", "mean"), 실제=("o", "mean"),
                                                                                    n=("p", "size"))
                    for d, r in c.iterrows():
                        calib.append({"모형": name, "원인": CAUSE_NAMES[j], "t(분)": t, "십분위": int(d) + 1,
                                      "예측 CIF%": r["예측"] * 100, "실제%": r["실제"] * 100, "n": int(r["n"])})
            ibs = np.mean([np.mean((Fp[:, b, j - 1] - O[:, b, j - 1]) ** 2) for b in range(1, H_EVAL + 1)])
            rows.append({"모형": name, "원인": CAUSE_NAMES[j], "지표": "IBS(5~120분)", "값": ibs})
    m = pd.DataFrame(rows)
    m["값"] = m["값"].round(5)
    return m, pd.DataFrame(calib).round(3)


def rf_baseline(imm, load, F_model):
    """기준선 (b): 승차 완료 건만 학습한 RandomForest 회귀(접수 -> 승차 분). 접수 시점 피처."""
    t0 = time.time()
    X = imm[["hour", "dow", "is_holiday", "dong_ab", "dong_d30"]].copy()
    for c in CAT_COLS:
        X[c] = imm[c].cat.codes
    lf = load.features(imm["t_request"], imm["o_gu"].astype(str).map(F.GU_INDEX).to_numpy())
    for c in F.LOAD_COLS:
        X[c] = lf[c].to_numpy()
    tr = (imm.split == "train") & (imm.E_all == 1)
    idx = imm.index[tr]
    if len(idx) > 300_000:
        idx = pd.Index(np.random.default_rng(SEED).choice(idx, 300_000, replace=False))
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEED)
    rf.fit(X.loc[idx], imm.loc[idx, "T_all"])
    te = (imm.split == "test") & (imm.E1_ab != 0)          # evaluate()의 test와 같은 행·순서
    pred = pd.Series(rf.predict(X[te]), index=imm.index[te])
    d = imm[te].assign(rf=pred, p_ab30=F_model[:, 30 // F.BIN, 1], p_d60=F_model[:, 60 // F.BIN, 0])
    b = d[d.E_all == 1]
    groups = [("승차 전체", b, "T_all"), ("승차 중 실제 대기 60분 초과", b[b.T_all > 60], "T_all"),
              ("승차 중 실제 대기 상위 10%", b[b.T_all >= b.T_all.quantile(.9)], "T_all"),
              ("배차 전 최종 포기", d[d.E1_ab == 2], "T1"), ("배차 전 재접수 취소", d[d.E1_ab == 4], "T1")]
    rows = []
    for name, g, tcol in groups:
        r = {"집단": name, "콜 수": len(g), "실제 시간 기준": "접수→승차" if tcol == "T_all" else "접수→취소",
             "실제 중앙값(분)": g[tcol].median(), "실제 평균(분)": g[tcol].mean(),
             "RF 예측 대기 중앙값(분)": g.rf.median(), "RF 예측 대기 평균(분)": g.rf.mean(),
             "RF가 60분 안에 탈 것으로 예측한 비율%": (g.rf <= 60).mean() * 100,
             "해저드 모형 30분 내 최종 포기 예측 평균%": g.p_ab30.mean() * 100,
             "해저드 모형 60분 내 배차 예측 평균%": g.p_d60.mean() * 100}
        if tcol == "T_all":
            r["RF 과소예측 평균(실제-예측, 분)"] = (g[tcol] - g.rf).mean()
            r["RF MAE(분)"] = (g[tcol] - g.rf).abs().mean()
        rows.append(r)
    t = pd.DataFrame(rows).round(2)
    save_table(t, "rf_underprediction.csv")
    print(t.to_string(index=False))
    print(f"  RF 학습 {len(idx):,}건, {time.time() - t0:.0f}초")


def shap_top(bst, feats, pp_te):
    s = pp_te.sample(min(30_000, len(pp_te)), random_state=SEED)
    contrib = bst.predict(s[feats], num_iteration=bst.best_iteration, pred_contrib=True)
    nf = len(feats) + 1
    rows = []
    for j in (1, 2):
        block = contrib[:, j * nf:(j + 1) * nf - 1]
        imp = np.abs(block).mean(axis=0)
        for f, v in zip(feats, imp):
            rows.append({"클래스": CAUSE_NAMES[j], "피처": f, "피처(한글)": FEAT_LABEL[f], "평균|SHAP|": v})
    t = pd.DataFrame(rows)
    t["순위"] = t.groupby("클래스")["평균|SHAP|"].rank(ascending=False, method="first").astype(int)
    t = t.sort_values(["클래스", "순위"])
    save_table(t.round(5), "shap_top10.csv")
    return t


def fig_shap(t, n_rows):
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.7), gridspec_kw={"wspace": 0.95})
    fig.subplots_adjust(left=0.235, right=0.965, top=0.8, bottom=0.25)
    for ax, (cls, color) in zip(axes, [("배차", COLOR["board"]), ("최종 포기", COLOR["cancel"])]):
        d = t[(t["클래스"] == cls) & (t["순위"] <= 10)].sort_values("순위", ascending=False)
        ax.barh(d["피처(한글)"], d["평균|SHAP|"], height=0.6, color=color, zorder=2)
        for y, v in enumerate(d["평균|SHAP|"]):
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=6.5, color=COLOR["ink2"])
        ax.set_title(f"'{cls}' 해저드", loc="left", fontsize=8.5)
        ax.set_xlabel("평균 |SHAP| (로짓 단위)")
        ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.set_xlim(0, d["평균|SHAP|"].max() * 1.25)
    fig.text(0.01, 0.97, TITLE_SHAP,
             fontsize=9.5, fontweight="bold", color=COLOR["ink"], va="top")
    note = (f"주: Stage 1(접수→배차) 5분 이산시간 LightGBM 다중분류 해저드. 테스트 기간(10~12월) person-period {n_rows:,}행에서 "
            "LightGBM 내장\n    TreeSHAP으로 계산한 클래스별 평균 |SHAP| 상위 10개. * 학습 구간(1~8월) 콜로 계산한 출발동 비율(같은 구 쪽으로 수축).\n"
            "    재접수 여부·차량구분은 피처로 쓰지 않음. SHAP은 연관의 크기이며 인과 효과가 아님.")
    save_fig(fig, "fig_05_shap.png", note=note)
    plt.close(fig)


def fig_calibration(calib, best):
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_IN, 2.5), gridspec_kw={"wspace": 0.45})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.27)
    for ax, (cls, t) in zip(axes, [("배차", 30), ("배차", 60), ("최종 포기", 30), ("최종 포기", 60)]):
        for name, color in [(best, COLOR["board"] if cls == "배차" else COLOR["cancel"]),
                            ("기준선(a) 구간×시각", COLOR["muted"])]:
            d = calib[(calib["모형"] == name) & (calib["원인"] == cls) & (calib["t(분)"] == t)]
            ax.plot(d["예측 CIF%"], d["실제%"], "o-", ms=3.5, lw=1.2, color=color, mec=COLOR["surface"], mew=0.8,
                    label="해저드 모형" if name == best else "기준선(a)")
        hi = calib[(calib["원인"] == cls) & (calib["t(분)"] == t)][["예측 CIF%", "실제%"]].max().max() * 1.05
        ax.plot([0, hi], [0, hi], color=COLOR["axis"], lw=0.8, zorder=0)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"{cls} {t}분", loc="left", fontsize=8)
        ax.tick_params(length=0, labelsize=6.5)
        ax.set_xlabel("예측(%)", fontsize=7)
    axes[0].set_ylabel("실제(%)", fontsize=7)
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.text(0.01, 0.97, "십분위 보정도: 예측한 누적확률과 실제 비율(테스트 10~12월)", fontsize=9.5, fontweight="bold",
             color=COLOR["ink"], va="top")
    save_fig(fig, "fig_A2_calibration.png", note="주: 접수 시점 피처로 예측한 원인별 누적발생확률을 십분위로 나눠 실제 비율과 비교. 대각선 = 완전 보정.")
    plt.close(fig)


def case_examples(bst, feats, imm, load):
    """보고서용 사례(이용자 안내 화면 예시): 테스트 기간 평일의 세 조건.
    s분 시점에 실제로 아직 미배차로 기다리던 콜 각각을 그 시점의 자기 피처(부하는 접수+s분 시점 값)로 예측해 평균을 내고,
    같은 콜들의 실제 값(1분 이산 Aalen-Johansen 조건부 확률)과 나란히 둔다.
    주의: 조건의 '보통' 피처 하나로 예측하면 오래 남은 콜의 특성(높은 부하, 관측되지 않은 휠체어 여부 등)이 빠져 크게 낙관적이 된다."""
    from survival import conditional, discrete_cif
    te = imm[(imm.split == "test") & (imm.dow < 5) & (imm.is_holiday == 0)]
    specs = [("A 평일 13시 강남구", 13, "강남구"), ("B 평일 16시 서초구", 16, "서초구"), ("C 평일 21시 노원구", 21, "노원구")]
    rows = []
    for label, hr, gu in specs:
        g = te[(te.hour == hr) & (te.o_gu == gu)].reset_index(drop=True)
        curve = discrete_cif(g.T1.values, g.E1_ab.values, [1, 2, 4])
        for s_min in (0, 15, 30):
            kb = s_min // F.BIN
            surv = g[np.floor(g.T1) >= s_min].reset_index(drop=True)          # s분 시점에 아직 대기 중
            n_bins = F.N_BINS - kb
            Fp, S = predict_cif_model(bst, feats, surv, load, s_bin=kb, n_bins=n_bins)
            mF, mS = Fp.mean(axis=0), S.mean(axis=0)
            d50 = np.argmax(mF[:, 0] >= 0.5) * F.BIN if (mF[:, 0] >= 0.5).any() else np.nan
            lf = load.features(surv["t_request"] + pd.Timedelta(minutes=s_min), np.full(len(surv), F.GU_INDEX[gu]))
            o = conditional(curve, s_min, 30, [1, 2, 4])
            rows.append({"사례": label, "테스트 기간 같은 조건 콜 수": len(g), "이미 기다린 시간(분)": s_min,
                         "그 시점 아직 대기 중인 콜": len(surv),
                         "그 시점 서울 미배차 대기 콜(중앙값)": lf.wait_seoul.median(),
                         "그 시점 같은 구 미배차 대기 콜(중앙값)": lf.wait_gu.median(),
                         "모형 다음 30분 배차%": mF[6, 0] * 100, "모형 다음 30분 최종 포기%": mF[6, 1] * 100,
                         "모형 다음 30분 재접수 취소%": mF[6, 2] * 100, "모형 30분 뒤에도 미배차%": mS[6] * 100,
                         "모형 배차 확률 50% 도달 남은 시간(분)": d50,
                         "실제 다음 30분 배차%": o["P_1"] * 100, "실제 다음 30분 최종 포기%": o["P_2"] * 100,
                         "실제 다음 30분 재접수 취소%": o["P_4"] * 100, "실제 30분 뒤에도 미배차%": o["P_대기중"] * 100})
    t = pd.DataFrame(rows).round(1)
    save_table(t, "case_examples.csv")
    print(t.drop(columns=["테스트 기간 같은 조건 콜 수"]).to_string(index=False))


def main():
    t_all = time.time()
    imm, load = prepare()
    print(f"  콜: 학습 {int((imm.split == 'train').sum()):,} / 검증 {int((imm.split == 'val').sum()):,} / "
          f"테스트 {int((imm.split == 'test').sum()):,}{' (10% 층화 표본)' if SAMPLE else ''}")
    t0 = time.time()
    pp = {s: F.person_period(imm[imm.split == s], load) for s in ("train", "val", "test")}
    print(f"  person-period: " + ", ".join(f"{s} {len(v):,}행" for s, v in pp.items()) + f" ({time.time() - t0:.0f}초)")

    models = train_models(pp["train"], pp["val"])
    best = min(models, key=lambda k: models[k][2])
    bst, feats = models[best][0], models[best][1]
    print(f"  선택 모형: {best}")
    # LightGBM C API는 한글이 든 Windows 경로에 쓰지 못하므로 문자열로 받아 저장한다
    (MODELS / "lgbm_stage1.txt").write_text(bst.model_to_string(num_iteration=bst.best_iteration), encoding="utf-8")

    test = imm[(imm.split == "test") & (imm.E1_ab != 0)].reset_index(drop=True)   # 상태 불명 제외
    t0 = time.time()
    preds = {name: predict_cif_model(m[0], m[1], test, load) for name, m in models.items()}
    preds["기준선(a) 구간×시각"] = predict_cif_empirical(empirical_hazard(pp["train"]), test)
    print(f"  테스트 CIF 예측 {len(test):,}콜 ({time.time() - t0:.0f}초)")
    metrics, calib = evaluate(preds, observed(test))
    for name, m in models.items():
        metrics.loc[len(metrics)] = {"모형": name, "원인": "전체", "지표": "검증 logloss(9월)", "값": round(m[2], 5)}
        metrics.loc[len(metrics)] = {"모형": name, "원인": "전체", "지표": "학습 시간(초)", "값": round(m[3], 1)}
    metrics.loc[len(metrics)] = {"모형": best, "원인": "전체", "지표": "선택 모형(검증 logloss 최소)", "값": 1}
    save_table(metrics, "model_metrics.csv")
    save_table(calib, "calibration_deciles.csv")
    print(metrics.pivot_table(index=["원인", "지표"], columns="모형", values="값").round(4).to_string())

    rf_baseline(imm, load, preds[best][0])
    st = shap_top(bst, feats, pp["test"])
    print(st[st["순위"] <= 10][["클래스", "순위", "피처(한글)", "평균|SHAP|"]].to_string(index=False))
    fig_shap(st, min(30_000, len(pp["test"])))
    fig_calibration(calib, best)
    case_examples(bst, feats, imm, load)
    print(f"[06_hazard_model] 완료 {time.time() - t_all:.0f}초{' (--sample)' if SAMPLE else ''}")


if __name__ == "__main__":
    main()
