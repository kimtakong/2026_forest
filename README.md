# 장애인콜택시 "보이지 않는 포기" 분석

서울시설공단 장애인콜택시 2025년 탑승내역(1,729,476건)으로 호출 대기를 이산시간 생존분석한다.
취소를 중도절단이 아니라 배차·승차와 경쟁하는 사건(competing risk)으로 다룬다.

「AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전」(한겨레 × 숲과나눔) 출품 코드.
분석 설계는 [CLAUDE.md](CLAUDE.md), 데이터 확인 결과는 [docs/data_checks.md](docs/data_checks.md)에 있다.

## 실행

```bash
pip install -r requirements.txt     # Python 3.13.9에서 확인
# datas/ 에 아래 원본 CSV 4개를 넣는다
python run_all.py                    # 전체 재현
python run_all.py --only 01          # 한 단계만
python run_all.py --sample           # 오래 걸리는 단계를 표본으로 시연
```

| 단계 | 스크립트 | 실행 시간* |
|---|---|---:|
| dl | `src/download_boundary.py` 행정동 경계 내려받기(첫 실행만) | 약 20초 |
| 00 | `src/00_data_checks.py` 데이터 확인, 동 대응표 | 약 6초 |
| 01 | `src/01_prepare.py` 분석용 콜 테이블, 제외 로그, 재접수 판별, 기준 건수표 | 약 30초 |
| 02 | `src/02_descriptive.py` 기술통계, 흐름도(fig_01), 부록 fig_A1 | 약 10초 |
| 03 | `src/03_official_vs_actual.py` 공식값 vs 실제(fig_02) | 약 5초 |
| 04 | `src/04_cif_vs_naive.py` naive KM vs 경쟁위험 CIF(fig_03) | 약 3초 |
| 05 | `src/05_conditional_residual.py` 조건부 잔여대기(fig_04) | 약 3초 |
| 합계 | | 약 1분 20초 (첫 실행, CSV 파싱 포함) |

\* Windows 11 PC 기준 측정값. 이후 단계가 추가되면 갱신한다.

## 데이터

### 원본 (공공데이터포털, 제공: 서울시설공단) — 저장소에 포함하지 않음

아래 URL에서 CSV를 내려받아 파일명 그대로 `datas/`에 넣는다. 공공데이터포털 정보는 2026-10-06에 확인했다.

| 데이터명 | 제공기관 | 데이터 기간 | 갱신주기 | 등록·수정일 | 이용허락범위 | URL |
|---|---|---|---|---|---|---|
| 서울시설공단_장애인콜택시 탑승내역_20251231 | 서울시설공단(장애인콜택시운영처) | 2025.01~2025.12 | 연간(차기 2027-07-01) | 2026-06-01 | 이용허락범위 제한 없음 | https://www.data.go.kr/data/15115859/fileData.do |
| 서울시설공단_장애인콜택시 시간대별 대기시간_20251231 | 서울시설공단 | 2025.01~2025.12 | 연간 | 등록 2026-07-27, 수정 2026-07-28 | 이용허락범위 제한 없음 | https://www.data.go.kr/data/15147613/fileData.do |
| 서울시설공단_장애인콜택시 이용목적_20240502 | 서울시설공단 | 코드표 | 수시(1회성) | 등록 2024-05-02, 수정 2025-05-11 | 이용허락범위 제한 없음 | https://www.data.go.kr/data/15127909/fileData.do |
| 서울시설공단_장애인콜택시 장애종류_20250528 | 서울시설공단 | 코드표 | 수시(1회성) | 등록 2024-05-07, 수정 2025-05-28 | 이용허락범위 제한 없음 | https://www.data.go.kr/data/15127975/fileData.do |

| 파일명 | 인코딩 |
|---|---|
| `서울시설공단_장애인콜택시 탑승내역_20251231.csv` | UTF-8 BOM |
| `서울시설공단_장애인콜택시 시간대별 대기시간_20251231.csv` | UTF-8 BOM |
| `서울시설공단_장애인콜택시 이용목적_20240502.csv` | CP949 |
| `서울시설공단_장애인콜택시 장애종류_20250528.csv` | CP949 |

### 행정동 경계 — 스크립트로 자동 다운로드

- 출처: [vuski/admdongkor](https://github.com/vuski/admdongkor) — 대한민국 행정동 경계 시계열(1975~현재)
  - 커밋 `dd1881663fcabc69b81393604e91ebf3a4202e9a`(2026-09-03)로 고정했다(`src/geo.py`).
- 원자료: 통계청 통계지리정보서비스(SGIS) 행정동 경계, **공공누리 제1유형(출처표시)**
- 라이선스: admdongkor 가공물은 **CC BY 4.0**이다. SGIS 출처표시 의무가 함께 승계된다. 원문은 `datas/admdongkor/LICENSE-DATA`에 함께 받는다.
- 사용 파일:
  - `dist/data/_index_v3.parquet`(시점별 행정동 이름 색인)
  - `parquet/emd_YYYYMMDD.parquet`. 기준 시점 2025-12-31과, 데이터의 동 이름이 가리키는 과거 시점(2008~2022 일부)을 받는다.
- 왜 과거 시점이 필요한가: 탑승내역의 출발동은 구마다 서로 다른 시점의 행정동 목록을 쓴다(예: 동대문구는 2008년 목록). 그래서 이름이 아니라 면적 겹침으로 2025년 경계에 대응시킨다. 자세한 근거는 `docs/data_checks.md` 5절에 있다.

## 폴더

```
datas/            원본 CSV, 행정동 경계 (git 제외)
data_processed/   가공본: raw_trips.parquet, calls.parquet, dong_mapping.csv, map_units.parquet (git 제외)
src/              utils.py(경로·상수), geo.py(경계·동 대응), download_boundary.py, 00~09 단계 스크립트
outputs/tables/   결과 표,  outputs/figures/  결과 그림
docs/             data_checks.md(데이터 확인), ai_log.md(AI 사용 기록)
```

## AI 사용

Claude Code(Anthropic)로 코드 작성과 데이터 확인을 했다. 작업 범위와 주요 프롬프트는 [docs/ai_log.md](docs/ai_log.md)에 기록한다.
