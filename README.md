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
| 01 | `src/01_prepare.py` 분석용 콜 테이블, 제외 로그 | 약 5초 |
| 합계 | | 약 40초 (첫 실행, CSV 파싱 포함) |

\* Windows 11 PC 기준 측정값. 이후 단계가 추가되면 갱신한다.

## 데이터

### 원본 (공공데이터포털, 제공: 서울시설공단) — 저장소에 포함하지 않음

공공데이터포털(https://www.data.go.kr)에서 아래 데이터명으로 검색해 CSV를 내려받아 `datas/`에 넣는다.

| 데이터명 | 파일명 | 기준일 | 인코딩 |
|---|---|---|---|
| 서울시설공단_장애인콜택시 탑승내역 | `서울시설공단_장애인콜택시 탑승내역_20251231.csv` | 2025-12-31 | UTF-8 BOM |
| 서울시설공단_장애인콜택시 시간대별 대기시간 | `서울시설공단_장애인콜택시 시간대별 대기시간_20251231.csv` | 2025-12-31 | UTF-8 BOM |
| 서울시설공단_장애인콜택시 이용목적 | `서울시설공단_장애인콜택시 이용목적_20240502.csv` | 2024-05-02 | CP949 |
| 서울시설공단_장애인콜택시 장애종류 | `서울시설공단_장애인콜택시 장애종류_20250528.csv` | 2025-05-28 | CP949 |

데이터셋별 상세 페이지 URL: _(확인 필요 — 제출 전 기입)_

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
