"""행정동 경계 내려받기 (vuski/admdongkor, 커밋 고정).

출처: 통계청 SGIS 행정동 경계(공공누리 제1유형) 기반, admdongkor 가공물(CC BY 4.0).
필요한 시점은 탑승내역의 출발동 목록으로 정한다(geo.pick_eras). 이미 받은 파일은 건너뛴다.
저장 위치: datas/admdongkor/ (git 제외)
"""
import time

import geo
from utils import SEOUL_GU, load_raw


def data_dong_counts(df):
    d = df[df["o_gu"].isin(SEOUL_GU) & df["o_dong"].notna()]
    c = d.groupby(["o_gu", "o_dong"], observed=True).size().rename("n_rows").reset_index()
    return c.astype({"o_gu": str, "o_dong": str})


def main():
    t0 = time.time()
    geo.download("LICENSE-DATA")
    idx = geo.load_index()
    dongs = data_dong_counts(load_raw())
    eras = geo.pick_eras(dongs, idx)
    vers = geo.needed_versions(eras, dongs, idx)
    print(f"필요한 경계 시점: {vers}")
    for v in vers:
        geo.download(f"parquet/emd_{v}.parquet")
    print(f"[download_boundary] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
