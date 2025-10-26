# -*- coding: utf-8 -*-
"""
condition.py — 20MA 상승변곡(필수) + 3MA '상승전환 시점' 즉시 포착 + 쌍바닥(외바닥 밑 제외)
- 경로: C:\work\mygit
- 입력: all_stock_data.json
- 출력: selected.txt
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import subprocess
import os

# =========================
# 경로/입출력
# =========================
REPO_DIR = Path(r"C:\work\mygit").resolve()
INPUT_JSON = REPO_DIR / "all_stock_data.json"
OUTPUT_TXT = REPO_DIR / "selected.txt"

# --- Git autosave settings ---
GIT_AUTOSAVE = True          # 자동 저장 켜기/끄기
GIT_REMOTE = "origin"        # 원격 이름
GIT_BRANCH = "main"          # 푸시할 브랜치

def git_autosave(repo_dir: Path, msg: str) -> None:
    """
    변경된 파일이 있으면 add/commit/push 수행.
    변경 사항이 없으면 조용히 패스.
    """
    try:
        cwd = str(repo_dir)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

        if not status:
            print("[GIT] 변경 사항 없음 → 스킵")
            return

        subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=cwd, check=True)
        subprocess.run(["git", "push", GIT_REMOTE, GIT_BRANCH], cwd=cwd, check=True)
        print(f"[GIT] push 완료 → {GIT_REMOTE}/{GIT_BRANCH}")
    except subprocess.CalledProcessError as e:
        print(f"[GIT][ERR] {e}")
    except Exception as e:
        print(f"[GIT][ERR] {type(e).__name__}: {e}")


# =========================
# 절대 필터 프리셋 (시총/주가/거래대금)
# =========================
MODE = "LIGHT"  # 선택: "LIGHT" / "NORMAL" / "STRICT"
USE_ABSOLUTE_FILTER = True  # True면 필터 적용, False면 해제

PRESETS = {
    "LIGHT": {   # 완화형
        "PRICE_MIN": 3000,
        "MCAP_MIN_WON": int(5e10),     # 500억
        "VOL_MIN_0": 100_000,
        "VALUE_MIN_0_WON": int(1e9),   # 10억
    },
    "NORMAL": {  # 기본형
        "PRICE_MIN": 5000,
        "MCAP_MIN_WON": int(1e11),     # 1000억
        "VOL_MIN_0": 200_000,
        "VALUE_MIN_0_WON": int(2e9),   # 20억
    },
    "STRICT": {  # 강화형
        "PRICE_MIN": 10_000,
        "MCAP_MIN_WON": int(2e11),     # 2000억
        "VOL_MIN_0": 300_000,
        "VALUE_MIN_0_WON": int(3e9),   # 30억
    },
}

if MODE not in PRESETS:
    raise ValueError(f"Unknown MODE: {MODE}")

P = PRESETS[MODE]

# =========================
# 기술 조건 파라미터
# =========================
INFLECT_LOOKBACK_20 = 40
SLOPE_EPS_UP = 1e-9

WIN_3MA = 40
VOL_K = 1.0
ALLOW_UNDER_PCT = 0.0
NEED_LAST_UP = False
DB_MIN_GAP = 3
DB_MAX_GAP = 30
DOJI_MAX_RATIO = 0.10
SHORT_UPPER_WICK_RATIO = 0.25
ANCHOR_OVERLAP_RATIO = 0.50

EXCLUDE_KEYWORDS = [
    "ETF", "ETN", "리츠", "REIT", "스팩", "SPAC", "우",   # ← 콤마 누락 버그 수정
    "우선주", "우B", "우C", "인버스", "레버리지", "선물", "풋", "콜",
    "TRUST", "PLUS", "RISE", "KODEX", "TIGER", "KOSEF", "HANARO", "미국",
    "ACE", "액티브", "KIWOOM", "SOL", "채권"
]

def is_excluded_name(name: str) -> bool:
    """
    종목명이 ETF, 리츠, 스팩, 우선주 등 제외 대상일 경우 True 반환.
    영문/한글 혼합표기 모두 대응.
    """
    up = (name or "").upper()
    return any(k in up for k in EXCLUDE_KEYWORDS)


# =========================
# 유틸 함수
# =========================
def is_excluded_name(name: str) -> bool:
    up = (name or "").upper()
    return any(k.upper() in up for k in EXCLUDE_KEYWORDS)

def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def volume_ma(s: pd.Series, w: int = 20) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()

def _to_number(x):
    """문자 숫자('1,234', '5.6e7', '123억', '12,345원')도 안전 변환"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    try:
        s = str(x).strip()
        s = s.replace(",", "").replace("원", "").replace("KRW", "")
        if s.endswith("억"):
            base = s[:-1].strip()
            return float(base) * 1e8
        return float(s)
    except Exception:
        return np.nan

def to_df(ohlcv_list: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv_list)
    if "date" in df.columns:
        df = df.sort_values("date")
    df = df.reset_index(drop=True)
    # 숫자 컬럼 안전 변환
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_number)
    # NA 처리
    if {"open","high","low","close"}.issubset(df.columns):
        df[["open","high","low","close"]] = df[["open","high","low","close"]].fillna(method="ffill").fillna(method="bfill")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype(float)
    return df

def local_minima_idx(s: pd.Series, start: int, end: int) -> List[int]:
    idxs = []
    start = max(start, 1)
    end = min(end, len(s) - 2)
    for i in range(start, end + 1):
        if s.iloc[i] <= s.iloc[i - 1] and s.iloc[i] <= s.iloc[i + 1]:
            idxs.append(i)
    return idxs

def _last_trading_idx(df: pd.DataFrame) -> int:
    """최근 1~7봉 내에서 거래량>0인 마지막 실제 거래 봉 인덱스를 반환. 없으면 -1(마지막)."""
    if "volume" in df.columns and len(df) > 0:
        vols = df["volume"]
        for i in range(1, min(7, len(vols)) + 1):
            try:
                if vols.iloc[-i] and vols.iloc[-i] > 0:
                    return -i
            except Exception:
                continue
    return -1

def _normalize_to_won(value: Any, *, guess_unit: str = "auto") -> float:
    """값이 억 단위로 저장된 경우 원 단위로 환산. 문자열(콤마/단위 포함)도 안전 파싱."""
    if value is None:
        return 0.0
    # 사전 정규화
    if not isinstance(value, (int, float, np.number)):
        try:
            s = str(value).strip().replace(",", "").replace("원", "").replace("KRW", "")
            if s.endswith("억"):
                base = s[:-1].strip()
                value = float(base) * 1e8
            else:
                value = float(s)
        except Exception:
            return 0.0
    v = float(value)
    if guess_unit == "won":
        return v
    if guess_unit == "eok":
        return v * 1e8
    # auto 휴리스틱: 너무 작으면 억 단위로 간주
    if v < 1e9:
        return v * 1e8
    return v

# =========================
# 절대 필터 (MODE 프리셋)
# =========================
def pass_absolute_filters(info: Dict[str, Any], df: pd.DataFrame) -> bool:
    # 0) 마지막 실제 거래봉 기준으로 계산 (휴일/0거래 보정)
    idx = _last_trading_idx(df)

    # 1) 가격
    last_close = float(df["close"].iloc[idx]) if len(df) else 0.0
    if last_close < P["PRICE_MIN"]:
        return False

    # 2) 시총 (원 단위 정규화)
    market_cap_raw = info.get("market_cap", None)
    if market_cap_raw is None and "shares_outstanding" in info:
        try:
            shares = _normalize_to_won(info.get("shares_outstanding"), guess_unit="won")
            market_cap_raw = last_close * shares
        except Exception:
            market_cap_raw = None
    market_cap_won = _normalize_to_won(market_cap_raw, guess_unit="auto") if market_cap_raw is not None else 0.0
    if market_cap_won < P["MCAP_MIN_WON"]:
        return False

    # 3) 거래량/거래대금 (문자 케이스 포함 보정)
    vol0 = float(df["volume"].iloc[idx]) if len(df) else 0.0
    vol_money_raw = info.get("vol_money")
    if vol_money_raw is None:
        vol_money_won = last_close * max(vol0, 0)
    else:
        vol_money_won = _normalize_to_won(vol_money_raw, guess_unit="auto")

    vol_ok = (vol0 >= P["VOL_MIN_0"])
    val_ok = (vol_money_won >= P["VALUE_MIN_0_WON"])

    if MODE == "LIGHT":
        # 라이트: OR (둘 중 하나만 통과)
        if not (vol_ok or val_ok):
            return False
    else:
        # 노멀/스트릭트: AND (둘 다 필요)
        if not (vol_ok and val_ok):
            return False

    return True

# =========================
# 20MA 상승변곡 (필수)
# =========================
def cond_20ma_inflect_up_required(df: pd.DataFrame) -> bool:
    ma20 = rolling_ma(df["close"], 20)
    d = ma20.diff()
    end = len(df) - 1
    start = max(2, end - INFLECT_LOOKBACK_20)

    hit = -1
    for i in range(start, end + 1):
        prev, cur = d.iloc[i - 1], d.iloc[i]
        if (prev <= 0) and (cur > SLOPE_EPS_UP):
            hit = i
            break

    # 변곡 없으면 False
    if hit == -1:
        return False

    # 변곡 이후 현재까지 우상향 유지해야 통과
    if not (d.iloc[-1] > 0):
        return False

    return True

# =========================
# 3MA: (전제) 상승변곡 발생  AND
#      (A) 그 봉에서 안착  OR  (B) 변곡 직후 상승구간 내 안착
#  - 쌍바닥: 외바닥 밑 제외, 케이스1~3 허용
#  - 최근성: '안착 발생 봉'이 0~1봉 이내
# =========================
def cond_3ma_turning_point_capture(
    df: pd.DataFrame,
    window: int = 40,              # 탐색창
    vol_k: float = 1.0,            # 안착봉 거래량 >= vol20 * k
    allow_under_pct: float = 0.0,  # 종가가 MA3 아래 허용 비율
    need_last_up: bool = False,    # 안착봉이 전봉 대비 상승 필요 여부
    db_min_gap: int = 3,
    db_max_gap: int = 30,
    doji_max_ratio: float = 0.10,
    short_upper_wick_ratio: float = 0.25,
    anchor_overlap_ratio: float = 0.50,
    max_anchor_delay: int = 1,     # 변곡 후 안착 허용 지연(봉) [A or B]
    slope_eps: float = 1e-6,       # 기울기 오차 허용
) -> bool:

    close, open_, high, low, vol = df["close"], df["open"], df["high"], df["low"], df["volume"]
    ma3 = close.rolling(3, min_periods=3).mean()
    vol20 = vol.rolling(20, min_periods=1).mean()
    d = ma3.diff()

    end = len(df) - 1
    start = max(2, end - window)

    # ---------- 외바닥/쌍바닥 ----------
    def _local_mins(s: pd.Series, sidx: int, eidx: int) -> List[int]:
        out = []
        sidx = max(sidx, 1)
        eidx = min(eidx, len(s) - 2)
        for i in range(sidx, eidx + 1):
            if s.iloc[i] <= s.iloc[i-1] and s.iloc[i] <= s.iloc[i+1]:
                out.append(i)
        return out

    mins = _local_mins(ma3, start, end)
    outer_bottom_idx = mins[0] if mins else None
    last_double_pair = None
    if len(mins) >= 2:
        b1, b2 = mins[-2], mins[-1]
        gap = b2 - b1
        if db_min_gap <= gap <= db_max_gap:
            m1, m2 = float(ma3.iloc[b1]), float(ma3.iloc[b2])
            if outer_bottom_idx is None or float(ma3.iloc[outer_bottom_idx]) <= m2:
                if m2 >= m1:
                    last_double_pair = (b1, b2)  # CASE1~3 허용, 외바닥 밑만 제외

    # ---------- (전제) 가장 최근 상승변곡 찾기 ----------
    turn_idx = -1
    for i in range(end, max(start + 1, 2) - 1, -1):  # 뒤에서부터(최근 변곡)
        prev, cur = d.iloc[i - 1], d.iloc[i]
        if (prev <= 0 + slope_eps) and (cur > 0 + slope_eps):
            turn_idx = i
            break
    if turn_idx == -1:
        return False

    # 쌍바닥이 있다면 변곡은 두 번째 저점 이후여야
    if last_double_pair is not None and turn_idx <= last_double_pair[1]:
        return False

    # ---------- 안착 판정 함수 ----------
    def _anchor_ok_at(idx: int) -> bool:
        m_now = float(ma3.iloc[idx])
        o_now, c_now = float(open_.iloc[idx]), float(close.iloc[idx])
        h_now, l_now = float(high.iloc[idx]), float(low.iloc[idx])
        body_top, body_bottom = max(o_now, c_now), min(o_now, c_now)
        body_size = max(0.0, body_top - body_bottom)
        range_size = max(1e-9, h_now - l_now)

        if body_size > 0:
            overlap = max(0.0, body_top - max(body_bottom, m_now))
            overlap_ratio = overlap / body_size
        else:
            overlap_ratio = 0.0

        is_doji = (body_size / range_size) <= doji_max_ratio
        is_bear = c_now < o_now
        upper_wick = max(0.0, h_now - body_top)
        short_upper = (upper_wick / range_size) <= short_upper_wick_ratio
        ma_ok = (c_now >= m_now * (1 - allow_under_pct))

        cond = (
            (overlap_ratio >= anchor_overlap_ratio)
            or (is_doji and ma_ok)
            or (is_bear and short_upper and ma_ok)
        )
        if not cond:
            return False

        # 거래량
        v_now, v20_now = int(vol.iloc[idx]), float(vol20.iloc[idx])
        if v_now < v20_now * vol_k:
            return False

        # (옵션) 직전 종가 대비 상승
        if need_last_up and idx >= 1 and not (close.iloc[idx] > close.iloc[idx - 1]):
            return False

        return True

    # ---------- (A) 변곡 그 봉에서 안착 OR (B) 변곡 후 상승구간 내 안착 ----------
    anchor_idx = None
    scan_end = min(end, turn_idx + max_anchor_delay)
    for i in range(turn_idx, scan_end + 1):
        # "상승구간" 필터: 해당 봉의 기울기가 양(+)
        if d.iloc[i] <= 0 + slope_eps:
            continue
        if _anchor_ok_at(i):
            anchor_idx = i
            break

    if anchor_idx is None:
        return False

    # ---------- 최근성: 안착봉이 0~1봉 이내 ----------
    if (end - anchor_idx) > 1:
        return False

    return True


# =========================
# 메인
# =========================
def main():
    if not INPUT_JSON.exists():
        print(f"[ERR] 입력 파일이 없습니다: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    total = len(raw)
    selected: List[Tuple[str, str]] = []

    # 디버그 카운터
    c_total = c_excluded = c_short = c_schema = 0
    c_abs_fail = c_20_fail = c_3_fail = 0

    for idx, (code, info) in enumerate(raw.items(), 1):
        name = info.get("name", "")
        c_total += 1

        # 🔥 ETF/리츠/스팩/우선주 등 제외
        if is_excluded_name(name):
            c_excluded += 1
            continue

        ohlcv = info.get("ohlcv", [])
        if not ohlcv or len(ohlcv) < 60:
            c_short += 1
            continue

        df = to_df(ohlcv)
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            c_schema += 1
            continue

        print(f"\r[검색중] {idx}/{total}  ({name})", end="")

        # ✅ 절대 필터: (단위/휴일 보정 + LIGHT OR 로직)
        if USE_ABSOLUTE_FILTER and not pass_absolute_filters(info, df):
            c_abs_fail += 1
            continue

        ok20 = cond_20ma_inflect_up_required(df)
        if not ok20:
            c_20_fail += 1
            continue

        ok3 = cond_3ma_turning_point_capture(df)
        if not ok3:
            c_3_fail += 1
            continue

        selected.append((code, name))

    selected.sort(key=lambda x: x[1])
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for code, name in selected:
            f.write(f"{code}\t{name}\n")

    print(f"\n[DONE:{MODE}] 종목 수: {len(selected)} → {OUTPUT_TXT}")
    # 단계별 통계
    print(f"[STATS] total={c_total} excluded={c_excluded} short_df={c_short} schema_miss={c_schema}")
    print(f"[STATS] abs_fail={c_abs_fail} 20ma_fail={c_20_fail} 3ma_fail={c_3_fail}")

    # === GitHub autosave ===
    if GIT_AUTOSAVE:
        commit_msg = f"update selected ({MODE})"
        os.chdir(REPO_DIR)
        ensure_gitignore_full(REPO_DIR)

        include_targets = ["selected.txt", "condition.py"]
        exclude_patterns = [
            "*.bak", "*.json", "selected_debug.json",
            "__pycache__/", "*.pyc", ".idea/", ".vscode/"
        ]

        # .gitignore 자동 갱신
        gitignore_path = REPO_DIR / ".gitignore"
        existing = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
        new_lines = [p for p in exclude_patterns if p not in existing]
        if new_lines:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            print(f"[GIT] .gitignore 업데이트 완료 → {gitignore_path}")

        # 커밋 대상만 add
        for fname in include_targets:
            path = REPO_DIR / fname
            if path.exists():
                subprocess.run(["git", "add", str(path)], check=False)

        # 커밋 + 푸시
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        print(f"[GIT] push 완료 → origin/main (files: {include_targets})")

# =========================
# Git ignore 자동생성 (.bak / 대용량 JSON 제외)
# =========================
def ensure_gitignore_full(repo_dir: Path):
    """
    .gitignore에 백업/데이터/시스템 파일 제외 패턴 추가 (자동)
    """
    gitignore_path = repo_dir / ".gitignore"
    patterns = [
        "*.bak", "*_bak.json", "*_backup.json", "*.json.bak",
        "all_stock_data_*.json.bak", "all_stock_data.json",
        "selected_debug.json", "__pycache__/", "*.pyc", ".idea/", ".vscode/", ".DS_Store"
    ]

    existing = []
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8").splitlines()

    new_lines = [p for p in patterns if p not in existing]
    if new_lines:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[GIT] .gitignore 업데이트 완료 → {gitignore_path}")
    else:
        print("[GIT] .gitignore 이미 최신 상태입니다.")

# =========================
# 메인 실행부
# =========================
if __name__ == "__main__":
    ensure_gitignore_full(REPO_DIR)  # ✅ 실행 시 자동 반영
    main()
