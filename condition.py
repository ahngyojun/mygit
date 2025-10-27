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

# =========================
# 모드 설정 (여기서만 바꾸면 됨)
# =========================
MODE = "lIGHT"               # "LIGHT" / "NORMAL" / "STRICT"
USE_ABSOLUTE_FILTER = True    # True: 필터 적용 / False: 해제

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
# 기술조건 프리셋 (모드별로 다르게)
# =========================
TECH_PRESETS = {
    "LIGHT": {
        "INFLECT_LOOKBACK_20": 50,   # 20MA 변곡 탐색 폭 (넓게)
        "SLOPE_EPS_UP": 0.0,
        "WIN_3MA": 50,               # 3MA 탐색 윈도우
        "VOL_K": 0.8,                # 안착봉 거래량 기준 (v_now >= v20*VOL_K)
        "ALLOW_UNDER_PCT": 0.02,     # 3MA 약간 하회 허용
        "NEED_LAST_UP": False,
        "DB_MIN_GAP": 2,
        "DB_MAX_GAP": 32,
        "DOJI_MAX_RATIO": 0.12,
        "SHORT_UPPER_WICK_RATIO": 0.35,
        "ANCHOR_OVERLAP_RATIO": 0.40, # 몸통 40% 이상 MA3 위
        "MAX_ANCHOR_DELAY": 2,        # 변곡 후 0~2봉 내 안착 허용
        "SLOPE_EPS_3MA": 1e-6,
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
    },
    "STRICT": {
        "INFLECT_LOOKBACK_20": 35,   # 타이트
        "SLOPE_EPS_UP": 1e-8,
        "WIN_3MA": 30,
        "VOL_K": 1.2,
        "ALLOW_UNDER_PCT": 0.0,
        "NEED_LAST_UP": True,        # 마지막 봉 상승 요구
        "DB_MIN_GAP": 4,
        "DB_MAX_GAP": 28,
        "DOJI_MAX_RATIO": 0.08,
        "SHORT_UPPER_WICK_RATIO": 0.20,
        "ANCHOR_OVERLAP_RATIO": 0.60, # 몸통 60% 이상 MA3 위
        "MAX_ANCHOR_DELAY": 0,        # 변곡 그 봉에서 바로 안착 요구
        "SLOPE_EPS_3MA": 1e-6,
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
    파일 상단 토글(MODE/USE_ABSOLUTE_FILTER)을 기본값으로 사용.
    환경변수/CLI 인자가 있으면 그 값으로 override.
    """
    env_mode = os.environ.get("MODE", MODE).upper()
    env_abs  = parse_bool_env("ABS_FILTER", USE_ABSOLUTE_FILTER)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", choices=["LIGHT", "NORMAL", "STRICT"], default=env_mode,
                        help="검색 강도 모드 (LIGHT/NORMAL/STRICT). 기본=파일 상단 MODE 또는 환경변수 MODE")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--abs-filter", dest="abs_filter", action="store_true",
                       help="절대필터 사용(최종 단계에서 시총/거래량/거래대금 체크)")
    group.add_argument("--no-abs-filter", dest="abs_filter", action="store_false",
                       help="절대필터 미사용")
    parser.set_defaults(abs_filter=env_abs)

    args, _ = parser.parse_known_args(sys.argv[1:])
    mode = args.mode.upper()
    if mode not in PRESETS:
        raise ValueError(f"Unknown MODE: {mode}")
    use_abs = bool(args.abs_filter)
    return mode, use_abs

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
    """
    종목명이 ETF, 리츠, 스팩, 우선주 등 제외 대상일 경우 True 반환.
    영문/한글 혼합표기 모두 대응.
    """
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

# ---- 시총 파서 & 다중 경로 조회 ----
def _parse_market_cap(raw) -> Optional[float]:
    """원 단위 float로 반환. 없거나 파싱 실패면 None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.number)):
        return float(raw)  # 숫자는 원으로 간주
    try:
        s = str(raw).strip().replace(",", "").replace("KRW", "").replace("원", "")
        if s.endswith("억"):
            base = float(s[:-1].strip())
            return base * 1e8
        return float(s)
    except Exception:
        return None

def get_market_cap(info: Dict[str, Any]) -> Optional[float]:
    """가능한 모든 경로에서 시총을 찾는다. 못 찾으면 None."""
    paths = [
        ("market_cap",),
        ("extra", "market_cap"),
        ("opt10001", "market_cap"),
        ("cap",),  # 과거 호환
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
    return None  # 누락 시 미적용

# =========================
# 절대 필터 (MODE 프리셋) — 기술조건 이후 최종 단계에서 적용
# =========================
# LIGHT 모드에서 20일 평균으로 보조 허용
VOL20_MIN_FACTOR = 0.6         # 20일 평균 거래량 ≥ (VOL_MIN_0 * 이 비율) 이면 OK
VALUE20_MIN_FACTOR = 0.6       # 20일 평균 거래대금 ≥ (VALUE_MIN_0_WON * 이 비율) 이면 OK

# 절대필터 상세 진단 카운터
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
        return False

    # (1) 가격 하한
    if last_close < P["PRICE_MIN"]:
        ABS_C_PRICE += 1
        return False

    # (2) 시총 (있을 때만 적용 — 누락이면 미적용)
    market_cap_won = get_market_cap(info)
    if market_cap_won is not None:
        if market_cap_won < P["MCAP_MIN_WON"]:
            ABS_C_MCAP += 1
            return False

    # (3) 거래량/거래대금: 당일 + 20일 평균 보조 기준
    vol0 = float(df["volume"].iloc[idx]) if len(df) else 0.0
    vol20 = float(df["volume"].rolling(20, min_periods=1).mean().iloc[idx]) if len(df) else 0.0

    val0 = last_close * max(vol0, 0.0)   # info['vol_money'] 누락 대비
    val20 = last_close * max(vol20, 0.0) # 20MA 가격 아님: 보수적으로 현재가 기준

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
            return False
    else:
        # NORMAL/STRICT: 둘 다 충족(AND)
        if vol0 < VOL_MIN_0:
            ABS_C_VOL += 1
            return False
        if val0 < VAL_MIN_0:
            ABS_C_VAL += 1
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

        # 윗꼬리 과도 방지 (요구시 활성화 가능)
        _upper = max(0.0, h_now - body_top)
        if body_size > 0 and (_upper / max(1e-9, range_size)) > short_upper_wick_ratio:
            return False

        # 도지 과도 방지
        if range_size > 0 and (body_size / range_size) < (1 - doji_max_ratio):
            pass  # body가 너무 작으면 걸러낼 수도 있음 (옵션화 가능)

        # 거래량 기준
        v_now, v20_now = int(vol.iloc[idx]), float(vol20.iloc[idx])
        if v_now < v20_now * vol_k:
            return False

        # 마지막 봉 상승 요구
        if need_last_up and idx >= 1 and not (close.iloc[idx] > close.iloc[idx - 1]):
            return False

        # MA3 약간 하회 허용 (필요시)
        if allow_under_pct > 0:
            if close.iloc[idx] < m_now * (1 - allow_under_pct):
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

    # --- 최근성: 안착봉이 0~1봉 이내 ---
    if (end - anchor_idx) > 1:
        return False

    return True

# =========================
# 메인
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

def main():
    # ---- 모드/필터 설정 읽기 ----
    MODE_runtime, USE_ABSOLUTE_FILTER_runtime = get_config_from_cli_env()
    # 지역 변수로 고정 (아래 로그/커밋 메시지에 사용)
    MODE = MODE_runtime
    USE_ABSOLUTE_FILTER = USE_ABSOLUTE_FILTER_runtime

    P = PRESETS[MODE]
    TP = TECH_PRESETS[MODE]

    print(f"[CONFIG] MODE={MODE}  USE_ABSOLUTE_FILTER={USE_ABSOLUTE_FILTER}")
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
        )
        if not ok3:
            c_3_fail += 1
            continue

        # ✅ 절대 필터는 최종 단계 (시총 누락시 미적용, LIGHT는 완화 로직)
        if USE_ABSOLUTE_FILTER and not pass_absolute_filters(info, df, P, MODE):
            c_abs_fail += 1
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
    if USE_ABSOLUTE_FILTER:
        print(f"[ABS-DETAIL] price_fail={ABS_C_PRICE} mcap_fail={ABS_C_MCAP} vol_fail={ABS_C_VOL} val_fail={ABS_C_VAL}")

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
