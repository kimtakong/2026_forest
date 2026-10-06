"""공용 상수·경로·로더. 모든 스크립트는 여기 정의된 경로만 쓴다."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "datas"
DATA_PROC = ROOT / "data_processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
BOUNDARY_DIR = DATA_RAW / "admdongkor"
for _p in (DATA_PROC, OUT_FIG, OUT_TAB, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

RAW_TRIPS = DATA_RAW / "서울시설공단_장애인콜택시 탑승내역_20251231.csv"
RAW_OFFICIAL_WAIT = DATA_RAW / "서울시설공단_장애인콜택시 시간대별 대기시간_20251231.csv"
RAW_PURPOSE = DATA_RAW / "서울시설공단_장애인콜택시 이용목적_20240502.csv"        # CP949
RAW_DISABILITY = DATA_RAW / "서울시설공단_장애인콜택시 장애종류_20250528.csv"     # CP949
RAW_CACHE = DATA_PROC / "raw_trips.parquet"   # 원본을 타입만 바꿔 저장한 캐시
CALLS = DATA_PROC / "calls.parquet"
DONG_MAPPING = DATA_PROC / "dong_mapping.csv"
MAP_UNITS = DATA_PROC / "map_units.parquet"

# ---------------------------------------------------------------- 상수
SEED = 42
OBS_CAP_MIN = 360                 # 관측 상한: 대기 시작 후 360분
IMMEDIATE_GAP_MIN = (-1.0, 5.0)   # immediate: -1분 <= 예정-접수 <= 5분 (예약목적 제외)
RESV_PURPOSE_PATTERN = r"^(예약|심야예약)"
SOURCE_NOTE = "자료: 서울시설공단 장애인콜택시 탑승내역(2025)"

SEOUL_GU = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
            "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
            "관악구", "서초구", "강남구", "송파구", "강동구"]

# 2025 한국 공휴일(대체·임시공휴일 포함)
HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",  # 신정, 임시공휴일, 설날 연휴
    "2025-03-01", "2025-03-03",                                              # 삼일절, 대체공휴일
    "2025-05-05", "2025-05-06",                                              # 어린이날·부처님오신날, 대체공휴일
    "2025-06-03", "2025-06-06",                                              # 대통령선거일, 현충일
    "2025-08-15",                                                            # 광복절
    "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",   # 개천절, 추석 연휴, 대체공휴일
    "2025-10-09", "2025-12-25",                                              # 한글날, 성탄절
])

# 원본 컬럼명 -> 분석용 이름
COLS = {
    "접수일시": "t_request", "예정일시": "t_sched", "배차일시": "t_dispatch", "승차일시": "t_board",
    "하차일시": "t_alight", "취소일시": "t_cancel", "출발구": "o_gu", "출발동": "o_dong",
    "목적구": "d_gu", "목적동": "d_dong", "이용목적": "purpose", "요금": "fare", "승차거리": "distance_m",
    "차량구분": "vehicle", "장애유형": "disability",
}
TIME_COLS = ["t_request", "t_sched", "t_dispatch", "t_board", "t_alight", "t_cancel"]
CAT_COLS = ["o_gu", "o_dong", "d_gu", "d_dong", "purpose", "vehicle", "disability"]


# ---------------------------------------------------------------- 로더
def load_raw(refresh: bool = False) -> pd.DataFrame:
    """탑승내역 원본을 읽어 타입을 바꾼다. 원본은 수정하지 않고 parquet 캐시만 만든다."""
    if RAW_CACHE.exists() and not refresh and RAW_CACHE.stat().st_mtime > RAW_TRIPS.stat().st_mtime:
        return pd.read_parquet(RAW_CACHE)
    t0 = time.time()
    df = pd.read_csv(RAW_TRIPS, encoding="utf-8-sig", dtype=str).rename(columns=COLS)
    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    for c in ["fare", "distance_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))  # 원본 행 순서(0부터)
    df.to_parquet(RAW_CACHE, index=False)
    print(f"[load_raw] CSV {len(df):,}행 파싱 {time.time() - t0:.0f}초 -> {RAW_CACHE.name}")
    return df


def load_official_wait() -> pd.Series:
    """공식 시간대별 대기시간평균(분). index = 일자+시간대 시작 시각."""
    o = pd.read_csv(RAW_OFFICIAL_WAIT, encoding="utf-8-sig")
    o.index = pd.to_datetime(o["일자"] + " " + o["시간대"])
    return o["대기시간평균"].rename("official")


# ---------------------------------------------------------------- 파생
def minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    """b - a 를 분(float)으로."""
    return (b - a).dt.total_seconds() / 60


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """배차·승차·취소 기록 조합으로 상태를 정한다 (CLAUDE.md 3-2 표)."""
    disp, board, canc = df["t_dispatch"].notna(), df["t_board"].notna(), df["t_cancel"].notna()
    df["status"] = pd.Categorical(np.select(
        [board, disp & canc, ~disp & canc, disp & ~canc],
        ["boarded", "post_cancel", "pre_cancel", "disp_noboard"], "unknown"),
        categories=["boarded", "pre_cancel", "post_cancel", "disp_noboard", "unknown"])
    df["flag_board_and_cancel"] = board & canc       # 승차 후 취소 기록(34건) -> 승차로 처리
    df["flag_board_no_dispatch"] = board & ~disp     # 배차 없이 승차(0건이어야 함)
    return df


def add_call_type(df: pd.DataFrame) -> pd.DataFrame:
    """CLAUDE.md 4-1 호출 유형."""
    df["gap_min"] = minutes(df["t_request"], df["t_sched"])          # 예정 - 접수
    df["resv_purpose"] = df["purpose"].astype(str).str.match(RESV_PURPOSE_PATTERN)
    lo, hi = IMMEDIATE_GAP_MIN
    imm = ~df["resv_purpose"] & df["gap_min"].between(lo, hi)
    df["call_type"] = pd.Categorical(np.where(imm, "immediate", "scheduled"), categories=["immediate", "scheduled"])
    return df


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    p = OUT_TAB / name
    df.to_csv(p, index=index, encoding="utf-8-sig")
    print(f"  -> {p.relative_to(ROOT)}")
    return p


# ---------------------------------------------------------------- 그림 공통
# 기준 팔레트(라이트, 인쇄용). 범주색은 순서대로만 쓴다: 1 파랑, 2 주황, 3 청록 ...
COLOR = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "board": "#2a78d6",      # 승차(배차) 경로 = 슬롯 1
    "cancel": "#eb6834",     # 취소 = 슬롯 2
    "neutral": "#898781",    # 기타·불명
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
}
FIG_WIDTH_IN = 16 / 2.54   # 보고서 본문 폭 16cm


def apply_style() -> str:
    """한글 폰트 + 차분한 축·격자. 모든 그림 스크립트가 처음에 부른다."""
    import matplotlib
    font = setup_korean_font()
    matplotlib.rcParams.update({
        "figure.facecolor": COLOR["surface"], "axes.facecolor": COLOR["surface"], "savefig.facecolor": COLOR["surface"],
        "axes.edgecolor": COLOR["axis"], "axes.linewidth": 0.8, "axes.labelcolor": COLOR["ink2"],
        "axes.titlecolor": COLOR["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": COLOR["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": COLOR["muted"], "ytick.color": COLOR["muted"], "xtick.labelcolor": COLOR["ink2"],
        "ytick.labelcolor": COLOR["ink2"], "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6,
    })
    return font


def save_fig(fig, name: str, source: str = SOURCE_NOTE, note: str | None = None) -> Path:
    """출처(와 주석)를 왼쪽 아래에 넣고 300dpi PNG로 저장."""
    text = source if note is None else f"{note}\n{source}"
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=6.5, color=COLOR["muted"], linespacing=1.4)
    p = OUT_FIG / name
    fig.savefig(p, dpi=300)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def setup_korean_font() -> str:
    """matplotlib 한글 폰트(Windows: Malgun Gothic, 없으면 NanumGothic)."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"] if f in names), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = font
    matplotlib.rcParams["axes.unicode_minus"] = False
    return font


def parse_sample_flag() -> bool:
    return "--sample" in sys.argv
