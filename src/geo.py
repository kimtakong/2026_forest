"""행정동 경계(vuski/admdongkor) 다운로드와 '데이터 동 이름 -> 지도 단위' 대응표 생성.

탑승내역의 출발동은 구마다 서로 다른 시점의 행정동 목록을 쓴다(예: 동대문구는 2008년 목록).
이름만으로 2025년 경계에 붙이면 같은 이름의 다른 구역에 잘못 붙는다(예: 2008년 전농2동 -> 현재 전농1동).
그래서 구마다 데이터 동 목록과 가장 잘 맞는 시점의 경계를 찾고, 그 경계를 기준 시점 경계와
면적으로 겹쳐 대응시킨다.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from utils import BOUNDARY_DIR, SEOUL_GU

ADMDONGKOR_REPO = "https://github.com/vuski/admdongkor"
ADMDONGKOR_COMMIT = "dd1881663fcabc69b81393604e91ebf3a4202e9a"   # 2026-09-03 master, 재현을 위해 고정
RAW_BASE = f"https://raw.githubusercontent.com/vuski/admdongkor/{ADMDONGKOR_COMMIT}"
REF_VERSION = "20251231"          # 지도 기준 시점 = 데이터 기준일
INDEX_FILE = "_index_v3.parquet"
MIN_ORPHAN_COVER = 0.30           # 데이터 동이 하나도 대응되지 않은 현재 동은, 가장 많이 덮는 데이터 동이 이 비율 이상 덮을 때 그 단위에 합친다


# ---------------------------------------------------------------- 다운로드
def download(rel_path: str, dest_dir: Path = BOUNDARY_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(rel_path).name
    if not dest.exists():
        url = f"{RAW_BASE}/{rel_path}"
        print(f"  download {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return dest


def emd_path(version: str) -> Path:
    return BOUNDARY_DIR / f"emd_{version}.parquet"


def load_index() -> pd.DataFrame:
    idx = pd.read_parquet(download(f"dist/data/{INDEX_FILE}"))
    idx = idx[(idx["level"] == "emd") & (idx["sidonm"] == "서울특별시") & (idx["version_key"] <= REF_VERSION)]
    return idx[["version_key", "code", "name", "sggnm"]].astype({"name": str, "sggnm": str, "code": str})


# ---------------------------------------------------------------- 이름 정규화
def name_variants(name: str) -> set[str]:
    """'수유제2동' ~ '수유2동', '종로1.2.3.4가동' ~ '종로1·2·3·4가동'.
    '홍제1동'처럼 '제'가 원래 이름의 일부인 경우를 지키려고, 제거 전·후를 모두 후보로 둔다."""
    s = re.sub(r"[·.\s]", "", str(name))
    return {s, re.sub(r"제(?=\d)", "", s)}


def match_name(name: str, candidates: list[str]) -> str | None:
    v = name_variants(name)
    hits = [c for c in candidates if v & name_variants(c)]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- 시점 선택
def pick_eras(data_dongs: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """구마다 데이터 동 목록과 Jaccard가 가장 높은 경계 시점(동률이면 최신)을 고른다."""
    rows = []
    for gu, sub in data_dongs.groupby("o_gu"):
        names = sub["o_dong"].tolist()
        best = None
        for ver, vs in idx[idx["sggnm"] == gu].groupby("version_key"):
            cand = vs["name"].tolist()
            matched = sum(match_name(n, cand) is not None for n in names)
            jac = matched / (len(names) + len(cand) - matched)
            key = (jac, ver)
            if best is None or key > best[0]:
                best = (key, ver, matched, len(cand))
        rows.append({"o_gu": gu, "era_version": best[1], "n_data_dong": len(names),
                     "n_matched": best[2], "n_era_dong": best[3], "jaccard": round(best[0][0], 4)})
    return pd.DataFrame(rows)


def needed_versions(eras: pd.DataFrame, data_dongs: pd.DataFrame, idx: pd.DataFrame) -> list[str]:
    vers = set(eras["era_version"]) | {REF_VERSION}
    for _, r in _fallback_versions(eras, data_dongs, idx).iterrows():
        vers.add(r["version"])
    return sorted(vers)


def _fallback_versions(eras, data_dongs, idx) -> pd.DataFrame:
    """선택 시점에 없는 이름은 그 이름이 존재한 마지막 시점을 쓴다."""
    out = []
    era = eras.set_index("o_gu")["era_version"]
    for gu, dong in data_dongs[["o_gu", "o_dong"]].itertuples(index=False):
        cand = idx[(idx["sggnm"] == gu) & (idx["version_key"] == era[gu])]["name"].tolist()
        if match_name(dong, cand) is None:
            g = idx[idx["sggnm"] == gu]
            ok = g[[bool(name_variants(dong) & name_variants(n)) for n in g["name"]]]
            if len(ok):
                out.append({"o_gu": gu, "o_dong": dong, "version": ok["version_key"].max()})
    return pd.DataFrame(out, columns=["o_gu", "o_dong", "version"])


# ---------------------------------------------------------------- 대응표
def _read_emd(version: str):
    import geopandas as gpd
    g = gpd.read_parquet(emd_path(version))
    g = g[g["sidonm"] == "서울특별시"].rename_geometry("geometry")
    return g[["emdcd", "emdnm", "sggnm", "geometry"]].astype({"emdcd": str, "emdnm": str, "sggnm": str})


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_mapping(data_dongs: pd.DataFrame, eras: pd.DataFrame, idx: pd.DataFrame):
    """데이터 (구, 동) -> 지도 단위(map unit) 대응표와 단위 경계를 만든다.

    1) 데이터 동마다 '시점 경계' 폴리곤을 찾는다(구별 era, 없으면 그 이름의 마지막 시점).
    2) 기준 시점(2025-12-31) 경계와 면적으로 겹쳐, 가장 많이 겹치는 현재 동을 1차 대응으로 둔다.
    3) 어떤 데이터 동에도 1차 대응되지 않은 현재 동(신설·분할로 생긴 동)은, 가장 많이 덮는
       데이터 동의 단위에 합친다(예: 상일제2동 <- 옛 강일동).
    4) 연결된 데이터 동·현재 동 묶음 하나가 지도 단위 하나다. 단위 경계 = 현재 동 경계의 합.
    """
    import geopandas as gpd

    cur = _read_emd(REF_VERSION)
    era = eras.set_index("o_gu")["era_version"]
    fb = _fallback_versions(eras, data_dongs, idx).set_index(["o_gu", "o_dong"])["version"]

    # 1) 데이터 동별 시점 폴리곤
    polys, cache = [], {}
    for gu, dong, n in data_dongs[["o_gu", "o_dong", "n_rows"]].itertuples(index=False):
        ver = fb.get((gu, dong), era[gu])
        if ver not in cache:
            cache[ver] = _read_emd(ver)
        g = cache[ver]
        g = g[g["sggnm"] == gu]
        hit = match_name(dong, g["emdnm"].tolist())
        if hit is None:
            continue
        row = g[g["emdnm"] == hit].iloc[0]
        polys.append({"o_gu": gu, "o_dong": dong, "n_rows": n, "era_version": ver,
                      "era_name": hit, "era_code": row["emdcd"], "geometry": row["geometry"]})
    old = gpd.GeoDataFrame(polys, geometry="geometry", crs=cur.crs)
    old["dkey"] = old["o_gu"] + "|" + old["o_dong"]

    # 2) 면적 겹침
    ov = gpd.overlay(old[["dkey", "geometry"]], cur[["emdcd", "emdnm", "sggnm", "geometry"]],
                     how="intersection", keep_geom_type=True)
    ov["a"] = ov.area
    ov = ov.groupby(["dkey", "emdcd", "emdnm", "sggnm"], as_index=False)["a"].sum()
    ov["share_old"] = ov["a"] / ov["dkey"].map(old.set_index("dkey").area)
    ov["share_cur"] = ov["a"] / ov["emdcd"].map(cur.set_index("emdcd").area)
    prim = ov.loc[ov.groupby("dkey")["share_old"].idxmax()].set_index("dkey")

    uf = _UnionFind()
    for dkey, r in prim.iterrows():
        uf.union("D:" + dkey, "C:" + r["emdcd"])
    # 3) 고아 현재 동
    orphans = sorted(set(cur["emdcd"]) - set(prim["emdcd"]))
    orphan_log = []
    for c in orphans:
        cand = ov[ov["emdcd"] == c].sort_values("share_cur", ascending=False)
        if len(cand) and cand.iloc[0]["share_cur"] >= MIN_ORPHAN_COVER:
            uf.union("C:" + c, "D:" + cand.iloc[0]["dkey"])
            orphan_log.append((c, cand.iloc[0]["emdnm"], cand.iloc[0]["dkey"], cand.iloc[0]["share_cur"]))
        else:
            orphan_log.append((c, cur.set_index("emdcd").loc[c, "emdnm"], None, cand.iloc[0]["share_cur"] if len(cand) else 0.0))

    # 4) 단위
    comp = {}
    for node in list(uf.p):
        comp.setdefault(uf.find(node), []).append(node)
    curname = cur.set_index("emdcd")["emdnm"]
    unit_rows, units = [], []
    for nodes in comp.values():
        cs = sorted(n[2:] for n in nodes if n.startswith("C:"))
        ds = sorted(n[2:] for n in nodes if n.startswith("D:"))
        if not ds:
            continue
        uid = "U" + cs[0]
        uname = "+".join(curname[c] for c in cs)
        geom = cur[cur["emdcd"].isin(cs)].union_all()
        units.append({"unit_id": uid, "unit_name": uname, "unit_gu": ds[0].split("|")[0],
                      "cur_codes": ",".join(cs), "n_data_dong": len(ds), "n_cur_dong": len(cs), "geometry": geom})
        for d in ds:
            unit_rows.append({"dkey": d, "unit_id": uid, "unit_name": uname,
                              "unit_n_data_dong": len(ds), "unit_n_cur_dong": len(cs), "cur_codes": cs})
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=cur.crs)
    ur = pd.DataFrame(unit_rows).set_index("dkey")

    m = old.drop(columns="geometry").set_index("dkey").join(ur)
    m = m.join(prim[["emdcd", "emdnm", "share_old"]].rename(
        columns={"emdcd": "primary_cur_code", "emdnm": "primary_cur_name", "share_old": "primary_share"}))
    # 단위가 옛 폴리곤을 얼마나 덮는가
    m["unit_coverage"] = [ov[(ov["dkey"] == k) & ov["emdcd"].isin(cs)]["share_old"].sum()
                          for k, cs in zip(m.index, m["cur_codes"])]
    # 이름만으로 2025 경계에 붙였다면 어디로 갔는가(검증용)
    m["name_only_cur_name"] = [match_name(d, cur[cur["sggnm"] == g]["emdnm"].tolist())
                               for g, d in zip(m["o_gu"], m["o_dong"])]
    m["name_only_correct"] = m["name_only_cur_name"].eq(m["primary_cur_name"])
    m["relation"] = np.select(
        [(m.unit_n_data_dong == 1) & (m.unit_n_cur_dong == 1) & m.name_only_correct,
         (m.unit_n_data_dong == 1) & (m.unit_n_cur_dong == 1),
         (m.unit_n_data_dong > 1) & (m.unit_n_cur_dong == 1),
         (m.unit_n_data_dong == 1) & (m.unit_n_cur_dong > 1)],
        ["same", "renamed", "merged_into_current", "current_split_dissolved"], "many_to_many")
    m = m.drop(columns="cur_codes").reset_index(drop=True)
    cols = ["o_gu", "o_dong", "n_rows", "relation", "unit_id", "unit_name", "era_version", "era_name", "era_code",
            "primary_cur_code", "primary_cur_name", "primary_share", "unit_coverage", "unit_n_data_dong",
            "unit_n_cur_dong", "name_only_cur_name", "name_only_correct"]
    orphan_df = pd.DataFrame(orphan_log, columns=["cur_code", "cur_name", "attached_to", "cover_share"])
    return m[cols], units, orphan_df


def gu_boundaries():
    """기준 시점 행정동을 구 단위로 합친 경계."""
    cur = _read_emd(REF_VERSION)
    g = cur[cur["sggnm"].isin(SEOUL_GU)].dissolve(by="sggnm").reset_index()
    return g[["sggnm", "geometry"]].rename(columns={"sggnm": "gu"})
