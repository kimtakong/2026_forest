"""00. 데이터 확인 (CLAUDE.md 10절). 결과 해석은 docs/data_checks.md.

1. 차량구분 = 요청 차종인가 배정 차종인가
2. 예정일시 < 접수일시 행의 차이 분포
3. 사전접수 간격 분포와 125분 자동 접수 블록
4. 공식 시간대별 대기시간의 산식 역추정 -> outputs/tables/official_definition_check.csv
5. 출발동 이름 -> 행정동 경계 대응표 -> data_processed/dong_mapping.csv, map_units.parquet
원본 전체(제외 규칙 적용 전)를 대상으로 한다.
"""
import time

import numpy as np
import pandas as pd

import geo
from download_boundary import data_dong_counts
from utils import DATA_PROC, DONG_MAPPING, MAP_UNITS, add_call_type, add_status, load_official_wait, load_raw, minutes, save_table


def pct(s):
    return (s * 100).round(2)


def check_vehicle(df):
    print("\n[1] 차량구분")
    t = pd.crosstab(df["status"], df["vehicle"])
    t["임차택시_비중%"] = pct(t["임차택시"] / t.sum(axis=1))
    save_table(t.reset_index(), "check_vehicle_by_status.csv")
    print(t.to_string())

    disp = df["t_dispatch"].notna()
    h = df["t_request"].dt.hour.rename("접수시")
    s = pd.DataFrame({
        "배차전취소_임차택시비중%": pct((df.loc[df.status == "pre_cancel", "vehicle"] == "임차택시").groupby(h).mean()),
        "배차콜_임차택시비중%": pct((df.loc[disp, "vehicle"] == "임차택시").groupby(h).mean())})
    save_table(s.reset_index(), "check_vehicle_share_by_hour.csv")

    d = df[disp]
    r = d.groupby("vehicle", observed=True).agg(배차건수=("status", "size"),
                                                배차후취소=("status", lambda x: (x == "post_cancel").sum()))
    r["배차후취소율%"] = pct(r["배차후취소"] / r["배차건수"])
    save_table(r.reset_index(), "check_vehicle_post_cancel_rate.csv")
    print(r.to_string())


def check_gaps(df):
    print("\n[2] 예정 < 접수")
    g = df["gap_min"]
    neg = df[g < 0]
    b = pd.cut(neg["gap_min"], [-np.inf, -60, -5, -1, 0], right=False).value_counts().sort_index()
    t = pd.DataFrame({"구간(분)": b.index.astype(str), "건수": b.values})
    t.loc[len(t)] = ["합계", len(neg)]
    t.loc[len(t)] = ["중앙값(초)", round(neg["gap_min"].median() * 60, 1)]
    t.loc[len(t)] = ["-1분 미만 중 취소 건수", int(neg.loc[neg.gap_min < -1, "t_cancel"].notna().sum())]
    save_table(t, "check_gap_negative.csv")
    print(t.to_string(index=False))

    print("\n[3] 사전접수 간격")
    bins = [-np.inf, -1, 0, 1, 5, 10, 15, 30, 60, 90, 110, 130, 180, 360, 1440, np.inf]
    t = pd.crosstab(pd.cut(g, bins), df["resv_purpose"]).rename(columns={False: "예약목적_아님", True: "예약목적"})
    save_table(t.reset_index().rename(columns={"gap_min": "간격(분)"}).astype({"간격(분)": str}), "check_gap_bins.csv")

    blk = df[g.between(110, 130)]
    auto = df[g.between(124.5, 125.5)]
    rows = [("110~130분 건수", len(blk)), ("정확히 125분(±0.5)", len(auto)),
            ("125분 블록 접수초=0 비율%", pct((auto.t_request.dt.second == 0).mean())),
            ("immediate 접수초=0 비율%", pct((df.loc[df.call_type == "immediate", "t_request"].dt.second == 0).mean())),
            ("125분 블록 평일 비율%", pct((auto.t_request.dt.dayofweek < 5).mean())),
            ("125분 블록 예약목적 건수", int(auto.resv_purpose.sum())),
            ("125분 블록 접수->배차 중앙값(분)", round(minutes(auto.t_request, auto.t_dispatch).median(), 1)),
            ("125분 블록 예정->배차 중앙값(분)", round(minutes(auto.t_sched, auto.t_dispatch).median(), 1)),
            ("125분 블록 배차전취소율%", pct((auto.status == "pre_cancel").mean())),
            ("125분 블록 배차후취소율%", pct((auto.status == "post_cancel").mean()))]
    rows += [(f"125분 블록 이용목적: {k}", v) for k, v in auto["purpose"].value_counts().head(6).items()]
    for lo, hi in [(-1, 5), (-1, 10), (-1, 15), (-np.inf, 5)]:
        rows.append((f"immediate 건수 (간격 {lo}~{hi}분)", int((~df.resv_purpose & g.between(lo, hi)).sum())))
    t = pd.DataFrame(rows, columns=["항목", "값"])
    save_table(t, "check_advance_block.csv")
    print(t.to_string(index=False))


def check_official(df):
    """공식 '대기시간평균'이 어떤 산식인지 후보별로 일자x시간대 칸을 맞춰 본다."""
    print("\n[4] 공식 대기시간 산식")
    off = load_official_wait()
    imm = df["call_type"] == "immediate"
    fl = lambda c: df[c].dt.floor("h")
    r2b = minutes(df.t_request, df.t_board)
    s2b = minutes(df.t_sched, df.t_board)
    s2b_trunc = minutes(df.t_sched.dt.floor("min"), df.t_board.dt.floor("min"))
    r2end = r2b.fillna(minutes(df.t_request, df.t_cancel))
    cands = [
        ("즉시호출·승차 | 접수→승차 평균 | 접수 시간대", r2b[imm], fl("t_request")[imm], "mean"),
        ("즉시호출·승차 | 접수→승차 중앙값 | 접수 시간대", r2b[imm], fl("t_request")[imm], "median"),
        ("즉시호출·승차+취소 | 접수→(승차 또는 취소) 평균 | 접수 시간대", r2end[imm], fl("t_request")[imm], "mean"),
        ("전체·승차 | 접수→승차 평균 | 접수 시간대", r2b, fl("t_request"), "mean"),
        ("즉시호출·배차 | 접수→배차 평균 | 접수 시간대", minutes(df.t_request, df.t_dispatch)[imm], fl("t_request")[imm], "mean"),
        ("전체·승차 | 예정→승차 평균 | 승차 시간대", s2b, fl("t_board"), "mean"),
        ("전체·승차 | 예정→승차 평균 | 예정 시간대", s2b, fl("t_sched"), "mean"),
        ("전체·승차 | 예정→승차, 음수→0, 평균 | 예정 시간대", s2b.clip(lower=0), fl("t_sched"), "mean"),
        ("전체·승차 | 예정→승차, 각 시각 분단위 절사, 음수→0, 평균 | 예정 시간대 [채택]", s2b_trunc.clip(lower=0), fl("t_sched"), "mean"),
    ]
    rows = []
    for name, v, k, how in cands:
        g = v.groupby(k).agg(how).dropna()
        j = pd.concat([off, g.rename("calc")], axis=1, join="inner")
        d = j["calc"] - j["official"]
        rows.append({"후보 산식": name, "비교 칸 수": len(j), "공식 칸 수": len(off),
                     "상관계수": round(j.corr().iloc[0, 1], 4), "평균차(계산-공식, 분)": round(d.mean(), 3),
                     "MAE(분)": round(d.abs().mean(), 3),
                     "내림 후 정확일치%": pct((np.floor(j["calc"]) == j["official"]).mean()),
                     "반올림 후 정확일치%": pct((np.round(j["calc"]) == j["official"]).mean())})
    t = pd.DataFrame(rows)
    save_table(t, "official_definition_check.csv")
    print(t.drop(columns=["공식 칸 수"]).to_string(index=False))
    n_cancel = int(df["status"].isin(["pre_cancel", "post_cancel"]).sum())
    print(f"  공식 산식에 들어가지 않는 취소 콜: {n_cancel:,}건")


def check_dong(df):
    print("\n[5] 출발동 -> 행정동 경계")
    dongs = data_dong_counts(df)
    idx = geo.load_index()
    eras = geo.pick_eras(dongs, idx)
    save_table(eras, "check_dong_eras.csv")
    print(eras[eras.era_version != geo.REF_VERSION].to_string(index=False))
    m, units, orphans = geo.build_mapping(dongs, eras, idx)
    m = dongs[["o_gu", "o_dong", "n_rows"]].merge(m.drop(columns="n_rows"), on=["o_gu", "o_dong"], how="left")
    m.to_csv(DONG_MAPPING, index=False, encoding="utf-8-sig")
    units.to_parquet(MAP_UNITS)
    geo.gu_boundaries().to_parquet(DATA_PROC / "gu_boundary.parquet")
    print(f"  -> {DONG_MAPPING.name}, {MAP_UNITS.name}, gu_boundary.parquet")
    save_table(orphans, "check_dong_orphans.csv")

    w = m["n_rows"]
    cur = idx[(idx.version_key == geo.REF_VERSION)]
    exact = [d in set(cur[cur.sggnm == g]["name"]) for g, d in zip(m.o_gu, m.o_dong)]
    named = m["name_only_cur_name"].notna()
    mapped = m["unit_id"].notna()
    rows = [("서울 출발 (구,동) 조합", len(m), int(w.sum())),
            ("기준 경계 이름 정확일치", int(np.sum(exact)), int(w[exact].sum())),
            ("표기 정규화 후 이름일치", int(named.sum()), int(w[named].sum())),
            ("  └ 그중 면적 대응과 다른 동에 붙음(오대응)", int((named & ~m.name_only_correct).sum()), int(w[named & ~m.name_only_correct].sum())),
            ("면적 대응으로 지도 단위 배정", int(mapped.sum()), int(w[mapped].sum())),
            ("  └ 단위가 옛 동 면적의 90% 이상 덮음", int((m.unit_coverage >= .9).sum()), int(w[m.unit_coverage >= .9].sum())),
            ("지도 단위 수", units.shape[0], np.nan),
            ("배정 못 한 기준 경계 동(빈칸)", int(orphans.attached_to.isna().sum()), np.nan)]
    t = pd.DataFrame(rows, columns=["항목", "조합 수", "행 수"])
    t["행 비율%"] = (t["행 수"] / w.sum() * 100).round(2)
    save_table(t, "check_dong_match.csv")
    print(t.to_string(index=False))
    print(m.loc[m.relation != "same", ["o_gu", "o_dong", "n_rows", "relation", "unit_name", "primary_share",
                                       "unit_coverage", "name_only_cur_name"]].round(3).to_string(index=False))


def main():
    t0 = time.time()
    df = add_call_type(add_status(load_raw()))
    check_vehicle(df)
    check_gaps(df)
    check_official(df)
    check_dong(df)
    print(f"\n[00_data_checks] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
