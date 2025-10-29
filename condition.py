# -*- coding: utf-8 -*-
"""
condition.py — 20MA 상승변곡(필수) + 3MA '상승전환 시점' 즉시 포착 + 쌍바닥(외바닥 밑 제외)
- 경로: C:\work\mygit
- 입력: all_stock_data.json
- 출력: selected.txt
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import subprocess
import os
import argparse
import sys
import re  # 시총/상장주식수 파서용

# =========================
# 모드 설정 (각자 독립 설정 가능)
# =========================
MODE_3MA = "STRICT"           # "LIGHT" / "NORMAL" / "STRICT"  → 3MA 안착 강도(0봉/1봉) 등
MODE_ABS = "LIGHT"            # "LIGHT" / "NORMAL" / "STRICT"  → 절대필터 강도(가격/시총/거래량/대금)
USE_ABSOLUTE_FILTER = True    # True: 절대필터 적용 / False: 미적용

# =========================
# 경로/입출력
# =========================
REPO_DIR = Path(r"C:\work\mygit").resolve()
INPUT_JSON = REPO_DIR / "all_stock_data.json"
OUTPUT_TXT = REPO_DIR / "selected.txt"

# --- Git autosave settings ---
GIT_AUTOSAVE = True           # 자동 저장 켜기/끄기
GIT_REMOTE = "origin"         # 원격 이름
GIT_BRANCH = "main"           # 푸시할 브랜치

def git_autosave(repo_dir: Path, msg: str) -> None:
    """변경된 파일이 있으면 add/commit/push 수행. 변경 사항이 없으면 조용히 패스."""
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
# 모드/필터 프리셋
# =========================
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

# =========================
# 기술조건 프리셋 (모드별)
# =========================
TECH_PRESETS = {
    "LIGHT": {
        "INFLECT_LOOKBACK_20": 50,
        "SLOPE_EPS_UP": 0.0,
        "WIN_3MA": 50,
        "VOL_K": 0.8,
        "ALLOW_UNDER_PCT": 0.02,
        "NEED_LAST_UP": False,
        "DB_MIN_GAP": 2,
        "DB_MAX_GAP": 32,
        "DOJI_MAX_RATIO": 0.12,
        "SHORT_UPPER_WICK_RATIO": 0.35,
        "ANCHOR_OVERLAP_RATIO": 0.40,
        "MAX_ANCHOR_DELAY": 2,
        "SLOPE_EPS_3MA": 1e-6,
        "RECENT_MAX_AGE": 1,   # 0~1봉 허용
    },
    "NORMAL": {
        "INFLECT_LOOKBACK_20": 40,
        "SLOPE_EPS_UP": 1e-9,
        "WIN_3MA": 40,
        "VOL_K": 1.0,
        "ALLOW_UNDER_PCT": 0.0,
        "NEED_LAST_UP": False,
        "DB_MIN_GAP": 3,
        "DB_MAX_GAP": 30,
        "DOJI_MAX_RATIO": 0.10,
        "SHORT_UPPER_WICK_RATIO": 0.25,
        "ANCHOR_OVERLAP_RATIO": 0.50,
        "MAX_ANCHOR_DELAY": 1,
        "SLOPE_EPS_3MA": 1e-6,
        "RECENT_MAX_AGE": 1,   # 0~1봉 허용
    },
    "STRICT": {
        "INFLECT_LOOKBACK_20": 35,
        "SLOPE_EPS_UP": 1e-8,
        "WIN_3MA": 30,
        "VOL_K": 1.2,
        "ALLOW_UNDER_PCT": 0.0,
        "NEED_LAST_UP": True,
        "DB_MIN_GAP": 4,
        "DB_MAX_GAP": 28,
        "DOJI_MAX_RATIO": 0.08,
        "SHORT_UPPER_WICK_RATIO": 0.20,
        "ANCHOR_OVERLAP_RATIO": 0.60,
        "MAX_ANCHOR_DELAY": 0,
        "SLOPE_EPS_3MA": 1e-6,
        "RECENT_MAX_AGE": 0,   # 0봉만
    },
}

def parse_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "t", "yes", "y", "on"):
        return True
    if val in ("0", "false", "f", "no", "n", "off"):
        return False
    return default

def get_config_from_cli_env():
    """
    파일 상단 토글을 기본값으로 사용.
    환경변수/CLI 인자가 있으면 그 값으로 override.
    하위호환: --mode 혹은 MODE 로 두 모드(MODE_3MA/MODE_ABS) 동시에 설정 가능.
    """
    # ----- 환경변수 우선 읽기 -----
    # 글로벌 MODE가 있으면 두 모드에 기본으로 전파 (하위호환)
    env_mode_global = os.environ.get("MODE", "").strip() or None
    env_mode_3ma = os.environ.get("MODE_3MA", "").strip() or env_mode_global or MODE_3MA
    env_mode_abs = os.environ.get("MODE_ABS", "").strip() or env_mode_global or MODE_ABS
    env_abs_flag = parse_bool_env("ABS_FILTER", USE_ABSOLUTE_FILTER)

    parser = argparse.ArgumentParser(add_help=True)
    # 하위호환: --mode 하나로 두 모드 동시 지정 가능
    parser.add_argument("--mode", choices=["LIGHT", "NORMAL", "STRICT"], help="하위호환: 3MA/ABS를 동시에 이 모드로 설정")
    # 독립 지정
    parser.add_argument("--mode-3ma", "--mode3", dest="mode_3ma",
                        choices=["LIGHT", "NORMAL", "STRICT"],
                        default=env_mode_3ma,
                        help="3MA 기술조건 모드 (안착 신선도 등)")
    parser.add_argument("--mode-abs", "--modeabs", dest="mode_abs",
                        choices=["LIGHT", "NORMAL", "STRICT"],
                        default=env_mode_abs,
                        help="절대필터 모드 (가격/시총/거래량/대금)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abs-filter", dest="abs_filter", action="store_true",
                       help="절대필터 사용")
    group.add_argument("--no-abs-filter", dest="abs_filter", action="store_false",
                       help="절대필터 미사용")
    parser.set_defaults(abs_filter=env_abs_flag)

    args, _ = parser.parse_known_args(sys.argv[1:])

    # --mode가 들어오면 두 모드에 동시 적용 (명시적 개별 지정이 있으면 그 값이 우선)
    if args.mode:
        base = args.mode.upper()
        mode_3ma = (args.mode_3ma or base).upper()
        mode_abs = (args.mode_abs or base).upper()
    else:
        mode_3ma = (args.mode_3ma or env_mode_3ma).upper()
        mode_abs = (args.mode_abs or env_mode_abs).upper()

    # 최종 검증
    if mode_3ma not in TECH_PRESETS:
        raise ValueError(f"Unknown 3MA MODE: {mode_3ma}")
    if mode_abs not in PRESETS:
        raise ValueError(f"Unknown ABS MODE: {mode_abs}")

    use_abs = bool(args.abs_filter)
    return mode_3ma, mode_abs, use_abs

# =========================
# 제외 대상 (ETF/리츠/스팩/우선주 등)
# =========================
EXCLUDE_KEYWORDS = [
    "ETF", "ETN", "리츠", "REIT", "스팩", "SPAC", "우",
    "우선주", "우B", "우C", "인버스", "레버리지", "선물", "풋", "콜",
    "TRUST", "PLUS", "RISE", "KODEX", "TIGER", "KOSEF", "HANARO", "미국",
    "ACE", "액티브", "KIWOOM", "SOL", "채권"
]

def is_excluded_name(name: str) -> bool:
    up = (name or "").upper()
    return any(k in up for k in EXCLUDE_KEYWORDS)

# =========================
# 유틸 함수
# =========================
def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

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

# =========================
# 시총/상장주식수 파서 & 조회 (강화판)
# =========================
_JO_RE  = re.compile(r"([\d\.,]+)\s*조")
_EOK_RE = re.compile(r"([\d\.,]+)\s*억")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_UNIT_RE = re.compile(r"(만|억)\s*주")
_NUM_ONLY_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def _to_float_num(s: str) -> float:
    return float(str(s).replace(",", ""))

def _parse_market_cap(raw) -> Optional[float]:
    """
    다양한 한국형 표기 -> 원(KRW) 실수로 변환
    허용 예: "3.2조", "3조 5000억", "3조5,000억", "1,234,567,890,000", "3.2조원", "5000억 원"
    0/음수는 None 처리
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.number)):
        v = float(raw)
        return v if v > 0 else None  # 0/음수 무시
    try:
        s = str(raw).strip().upper()
        s = s.replace("KRW", "").replace("WON", "").replace("원", "")
        s = s.replace(" ", "")
        # 복합: 조+억
        m_j = _JO_RE.search(s)
        m_e = _EOK_RE.search(s)
        if (m_j is not None) or (m_e is not None):
            jo  = _to_float_num(m_j.group(1)) if m_j else 0.0
            eok = _to_float_num(m_e.group(1)) if m_e else 0.0
            v = jo * 1e12 + eok * 1e8
            return v if v > 0 else None
        # 일반 숫자
        nums = _NUM_RE.findall(s)
        if nums:
            v = _to_float_num(nums[0])
            return v if v > 0 else None
        return None
    except Exception:
        return None

# 상장주식수 파서 (만주/억주 포함)
def _parse_shares(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.number)):
        v = float(raw)
        return v if v > 0 else None
    try:
        s = str(raw).strip().replace(",", "")
        m = _UNIT_RE.search(s)
        if m:
            unit = m.group(1)
            nums = _NUM_ONLY_RE.findall(s)
            if not nums:
                return None
            n = float(nums[0])
            mul = 1e4 if unit == "만" else 1e8  # 만주/억주
            v = n * mul
            return v if v > 0 else None
        nums = _NUM_ONLY_RE.findall(s)
        if nums:
            v = float(nums[0])
            return v if v > 0 else None
        return None
    except Exception:
        return None

def get_market_cap(info: Dict[str, Any]) -> Optional[float]:
    """가능한 모든 경로에서 시총을 찾는다. 못 찾으면 None."""
    paths = [
        ("market_cap",),
        ("extra", "market_cap"),
        ("opt10001", "market_cap"),
        ("cap",),
        ("opt10001", "시가총액"),
        ("extra", "시가총액"),
        ("시가총액",),
        ("mktcap",), ("marketcap",), ("market_cap_krw",),
    ]
    for path in paths:
        cur = info
        ok = True
        for k in path:
            if not (isinstance(cur, dict) and (k in cur)):
                ok = False
                break
            cur = cur[k]
        if ok:
            val = _parse_market_cap(cur)
            if val is not None:
                return val

    # 최후 수단: dict 전체 스캔
    try:
        stack = [info]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    key = str(k)
                    if ("cap" in key.lower()) or ("시가총" in key):
                        val = _parse_market_cap(v)
                        if val is not None:
                            return val
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    except Exception:
        pass
    return None

# 상장주식수 조회
_SHARES_KEYS = [
    ("opt10001","상장주식"),
    ("opt10001","상장주식수"),
    ("extra","shares_outstanding"),
    ("extra","shares"),
    ("shares_outstanding",),
    ("상장주식수",),
    ("상장주식",),
    ("listed_shares",),
]
def get_listed_shares(info: Dict[str, Any]) -> Optional[float]:
    for path in _SHARES_KEYS:
        cur = info
        ok = True
        for k in path:
            if not (isinstance(cur, dict) and (k in cur)):
                ok = False; break
            cur = cur[k]
        if ok:
            v = _parse_shares(cur)
            if v is not None:
                return v
    try:
        stack = [info]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    key = str(k)
                    if ("shares" in key.lower()) or ("상장주" in key):
                        vv = _parse_shares(v)
                        if vv is not None:
                            return vv
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    except Exception:
        pass
    return None

# =========================
# 절대 필터 (MODE 프리셋) — 기술조건 이후 최종 단계에서 적용
# =========================
VOL20_MIN_FACTOR = 0.6   # LIGHT: 20일 평균 거래량 보조
VALUE20_MIN_FACTOR = 0.6 # LIGHT: 20일 평균 거래대금 보조

ABS_C_PRICE = 0
ABS_C_MCAP  = 0
ABS_C_VOL   = 0
ABS_C_VAL   = 0

def pass_absolute_filters(info: Dict[str, Any], df: pd.DataFrame, P: Dict[str, Any], MODE: str) -> bool:
    global ABS_C_PRICE, ABS_C_MCAP, ABS_C_VOL, ABS_C_VAL

    if df is None or len(df) == 0:
        return False

    idx = _last_trading_idx(df)
    last_close = float(df["close"].iloc[idx]) if len(df) else 0.0
    if not np.isfinite(last_close) or last_close <= 0:
        ABS_C_PRICE += 1
        print(f"[ABS-DBG] price_fail  close={last_close:.0f}  min={P['PRICE_MIN']}")
        return False

    # (1) 가격 하한
    if last_close < P["PRICE_MIN"]:
        ABS_C_PRICE += 1
        print(f"[ABS-DBG] price_fail  close={last_close:.0f}  min={P['PRICE_MIN']}")
        return False

    # (2) 시총: 값이 ‘있고 >0’일 때 비교, 없으면 보정 시도
    market_cap_won = get_market_cap(info)
    if market_cap_won is None:
        shares = get_listed_shares(info)
        if shares and shares > 0:
            market_cap_won = shares * last_close
            print(f"[ABS-DBG] mcap_fallback  shares={shares:.0f}  close={last_close:.0f}  mcap≈{market_cap_won:.0f}")
    if (market_cap_won is not None) and (market_cap_won <= 0):
        market_cap_won = None  # 안전

    if market_cap_won is not None:
        if market_cap_won < P["MCAP_MIN_WON"]:
            ABS_C_MCAP += 1
            print(f"[ABS-DBG] mcap_fail   mcap={market_cap_won:.0f}  min={P['MCAP_MIN_WON']}")
            return False

    # (3) 거래량/거래대금: 당일 + 20일 평균 보조 기준
    vol0 = float(df["volume"].iloc[idx]) if len(df) else 0.0
    vol20 = float(df["volume"].rolling(20, min_periods=1).mean().iloc[idx]) if len(df) else 0.0

    val0 = last_close * max(vol0, 0.0)
    val20 = last_close * max(vol20, 0.0)

    VOL_MIN_0 = P["VOL_MIN_0"]
    VAL_MIN_0 = P["VALUE_MIN_0_WON"]

    if MODE == "LIGHT":
        vol_ok = (vol0 >= VOL_MIN_0) or (vol20 >= VOL_MIN_0 * VOL20_MIN_FACTOR)
        val_ok = (val0 >= VAL_MIN_0) or (val20 >= VAL_MIN_0 * VALUE20_MIN_FACTOR)
        if not (vol_ok or val_ok):
            if not vol_ok:
                ABS_C_VOL += 1
            if not val_ok:
                ABS_C_VAL += 1
            print(
                f"[ABS-DBG] vol/val_fail  vol0={vol0:.0f} vol20={vol20:.0f}  "
                f"VAL0={val0/1e8:.2f}억 VAL20={val20/1e8:.2f}억  "
                f"VOL_MIN={VOL_MIN_0} VAL_MIN={VAL_MIN_0} "
                f"factors=({VOL20_MIN_FACTOR},{VALUE20_MIN_FACTOR})"
            )
            return False
    else:
        # NORMAL/STRICT: 둘 다 충족(AND)
        if vol0 < VOL_MIN_0:
            ABS_C_VOL += 1
            print(f"[ABS-DBG] vol_fail   vol0={vol0:.0f}  min={VOL_MIN_0}")
            return False
        if val0 < VAL_MIN_0:
            ABS_C_VAL += 1
            print(f"[ABS-DBG] val_fail   val0={val0/1e8:.2f}억  min={VAL_MIN_0/1e8:.2f}억")
            return False

    return True

# =========================
# 20MA 상승변곡 (필수) — 모드 파라미터 주입형
# =========================
def cond_20ma_inflect_up_required(df: pd.DataFrame,
                                  lookback: int,
                                  slope_eps_up: float) -> bool:
    ma20 = rolling_ma(df["close"], 20)
    d = ma20.diff()
    end = len(df) - 1
    start = max(2, end - lookback)

    hit = -1
    for i in range(start, end + 1):
        prev, cur = d.iloc[i - 1], d.iloc[i]
        if (prev <= 0) and (cur > slope_eps_up):
            hit = i
            break

    if hit == -1:
        return False
    if not (d.iloc[-1] > 0):
        return False
    return True

# =========================
# 3MA '상승전환 시점' 즉시 포착 (+쌍바닥: 외바닥 밑 제외)
# =========================
def cond_3ma_turning_point_capture(
    df: pd.DataFrame,
    window: int = 40,
    vol_k: float = 1.0,
    allow_under_pct: float = 0.0,
    need_last_up: bool = False,
    db_min_gap: int = 3,
    db_max_gap: int = 30,
    doji_max_ratio: float = 0.10,
    short_upper_wick_ratio: float = 0.25,
    anchor_overlap_ratio: float = 0.50,
    max_anchor_delay: int = 1,
    slope_eps: float = 1e-6,
    recent_max_age: int = 1,   # ‘안착 후 허용 지연 봉수’ (0이면 0봉만)
) -> bool:

    close, open_, high, low, vol = df["close"], df["open"], df["high"], df["low"], df["volume"]
    ma3 = close.rolling(3, min_periods=3).mean()
    vol20 = vol.rolling(20, min_periods=1).mean()
    d = ma3.diff()

    end = len(df) - 1
    start = max(2, end - window)

    # --- 쌍바닥 탐지 (외바닥 밑 제외 로직 포함) ---
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
                    last_double_pair = (b1, b2)

    # --- (전제) 최근 상승변곡 ---
    turn_idx = -1
    for i in range(end, max(start + 1, 2) - 1, -1):
        prev, cur = d.iloc[i - 1], d.iloc[i]
        if (prev <= 0 + slope_eps) and (cur > 0 + slope_eps):
            turn_idx = i
            break
    if turn_idx == -1:
        return False

    # 쌍바닥이 있으면 변곡은 두 번째 저점 이후여야
    if last_double_pair is not None and turn_idx <= last_double_pair[1]:
        return False

    # --- 안착 판정 ---
    def _anchor_ok_at(idx: int) -> bool:
        m_now = float(ma3.iloc[idx])
        o_now, c_now = float(open_.iloc[idx]), float(close.iloc[idx])
        h_now, l_now = float(high.iloc[idx]), float(low.iloc[idx])
        body_top, body_bottom = max(o_now, c_now), min(o_now, c_now)
        body_size = max(0.0, body_top - body_bottom)
        range_size = max(1e-9, h_now - l_now)

        # 몸통이 MA3 위로 겹친 비율
        overlap = max(0.0, body_top - max(body_bottom, m_now))
        overlap_ratio = 0.0 if body_size == 0 else (overlap / body_size)
        if overlap_ratio < anchor_overlap_ratio:
            return False

        # 윗꼬리 과도 방지
        _upper = max(0.0, h_now - body_top)
        if body_size > 0 and (_upper / max(1e-9, range_size)) > short_upper_wick_ratio:
            return False

        # 거래량 기준
        v_now, v20_now = int(vol.iloc[idx]), float(vol20.iloc[idx])
        if v_now < v20_now * vol_k:
            return False

        # 마지막 봉 상승 요구
        if need_last_up and idx >= 1 and not (close.iloc[idx] > close.iloc[idx - 1]):
            return False

        # MA3 약간 하회 허용 (필요시)
        if allow_under_pct > 0 and close.iloc[idx] < m_now * (1 - allow_under_pct):
            return False

        return True

    # --- (A) 변곡 그 봉에서 안착 OR (B) 변곡 후 상승구간 내 안착 ---
    anchor_idx = None
    scan_end = min(end, turn_idx + max(0, max_anchor_delay))
    for i in range(turn_idx, scan_end + 1):
        if d.iloc[i] <= 0 + slope_eps:
            continue
        if _anchor_ok_at(i):
            anchor_idx = i
            break

    if anchor_idx is None:
        return False

    # --- 최근성: 안착 recent_max_age 봉 이내 ---
    if (end - anchor_idx) > recent_max_age:
        return False

    return True

# =========================
# 디버그/깃IGNORE 유틸
# =========================
DEBUG_MCAP = False
DEBUG_SAMPLE = 8
_dbg_cnt = 0

def _dbg_mcap_once(code, info):
    global _dbg_cnt
    if not DEBUG_MCAP or _dbg_cnt >= DEBUG_SAMPLE:
        return
    print(f"[MCAP-DBG] {code} mcap={get_market_cap(info)}")
    _dbg_cnt += 1

def ensure_gitignore_full(repo_dir: Path):
    """ .gitignore에 백업/데이터/시스템 파일 제외 패턴 추가 (자동) """
    gitignore_path = repo_dir / ".gitignore"
    patterns = [
        "*.bak", "*_bak.json", "*_backup.json", "*.json.bak",
        "all_stock_data_*.json.bak", "all_stock_data.json",
        "selected_debug.json", "__pycache__/", "*.pyc", ".idea/", ".vscode/", ".DS_Store"
    ]
    existing = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
    new_lines = [p for p in patterns if p not in existing]
    if new_lines:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[GIT] .gitignore 업데이트 완료 → {gitignore_path}")
    else:
        print("[GIT] .gitignore 이미 최신 상태입니다.")

# =========================
# 메인
# =========================
def main():
    # ---- 모드/필터 설정 읽기 ----
    MODE_3MA_rt, MODE_ABS_rt, USE_ABS_rt = get_config_from_cli_env()
    MODE_3MA_FINAL = MODE_3MA_rt
    MODE_ABS_FINAL = MODE_ABS_rt
    USE_ABSOLUTE_FILTER = USE_ABS_rt

    P = PRESETS[MODE_ABS_FINAL]          # 절대필터용 프리셋
    TP = TECH_PRESETS[MODE_3MA_FINAL]    # 기술조건(3MA/20MA)용 프리셋

    print(f"[CONFIG] 3MA_MODE={MODE_3MA_FINAL}  ABS_MODE={MODE_ABS_FINAL}  USE_ABSOLUTE_FILTER={USE_ABSOLUTE_FILTER}")
    sys.stdout.flush()

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

        # 진행률 출력 (유지)
        if idx == 1:
            print("")
        print(f"\r[검색중] {idx}/{total}  ({name})", end="", flush=True)

        if DEBUG_MCAP and idx <= DEBUG_SAMPLE:
            _dbg_mcap_once(code, info)

        # ✅ 기술 조건 먼저 (모드별 파라미터 적용)
        ok20 = cond_20ma_inflect_up_required(
            df,
            lookback=TP["INFLECT_LOOKBACK_20"],
            slope_eps_up=TP["SLOPE_EPS_UP"]
        )
        if not ok20:
            c_20_fail += 1
            continue

        ok3 = cond_3ma_turning_point_capture(
            df,
            window=TP["WIN_3MA"],
            vol_k=TP["VOL_K"],
            allow_under_pct=TP["ALLOW_UNDER_PCT"],
            need_last_up=TP["NEED_LAST_UP"],
            db_min_gap=TP["DB_MIN_GAP"],
            db_max_gap=TP["DB_MAX_GAP"],
            doji_max_ratio=TP["DOJI_MAX_RATIO"],
            short_upper_wick_ratio=TP["SHORT_UPPER_WICK_RATIO"],
            anchor_overlap_ratio=TP["ANCHOR_OVERLAP_RATIO"],
            max_anchor_delay=TP["MAX_ANCHOR_DELAY"],
            slope_eps=TP["SLOPE_EPS_3MA"],
            recent_max_age=TP.get("RECENT_MAX_AGE", 1),
        )
        if not ok3:
            c_3_fail += 1
            continue

        # ✅ 절대 필터 (시총 누락→보정, LIGHT는 완화 로직)
        if USE_ABSOLUTE_FILTER and not pass_absolute_filters(info, df, P, MODE_ABS_FINAL):
            c_abs_fail += 1
            continue

        selected.append((code, name))

    selected.sort(key=lambda x: x[1])
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for code, name in selected:
            f.write(f"{code}\t{name}\n")

    print(f"\n[DONE:{MODE_3MA_FINAL}/{MODE_ABS_FINAL}] 종목 수: {len(selected)} → {OUTPUT_TXT}")
    # 단계별 통계
    print(f"[STATS] total={c_total} excluded={c_excluded} short_df={c_short} schema_miss={c_schema}")
    print(f"[STATS] abs_fail={c_abs_fail} 20ma_fail={c_20_fail} 3ma_fail={c_3_fail}")
    if USE_ABSOLUTE_FILTER:
        print(f"[ABS-DETAIL] price_fail={ABS_C_PRICE} mcap_fail={ABS_C_MCAP} vol_fail={ABS_C_VOL} val_fail={ABS_C_VAL}")

    # === GitHub autosave ===
    if GIT_AUTOSAVE:
        commit_msg = f"update selected ({MODE_3MA_FINAL}/{MODE_ABS_FINAL})"
        os.chdir(REPO_DIR)
        ensure_gitignore_full(REPO_DIR)

        include_targets = ["selected.txt", "condition.py"]
        exclude_patterns = [
            "*.bak", "*.json", "selected_debug.json",
            "__pycache__/", "*.pyc", ".idea/", ".vscode/"
        ]

        # .gitignore 자동 갱신 (수정 완료 버전)
        gitignore_path = REPO_DIR / ".gitignore"
        existing = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
        new_lines = [p for p in exclude_patterns if p not in existing]
        if new_lines:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            print(f"[GIT] .gitignore 업데이트 완료 → {gitignore_path}")
        else:
            print("[GIT] .gitignore 이미 최신 상태입니다.")

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
# 메인 실행부
# =========================
if __name__ == "__main__":
    try:
        ensure_gitignore_full(REPO_DIR)  # 실행 시 자동 반영
    except Exception as e:
        print(f"[GIT] .gitignore 처리 중 에러: {e}")
    main()
