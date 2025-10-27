# -*- coding: utf-8 -*-
"""
condition.py — 20MA 상승변곡(필수) + 3MA '안착/쌍바닥/신선도' + 절대필터 안정화
- 경로: C:\work\mygit
- 입력: all_stock_data.json
- 출력: selected.txt
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os
import math
import numpy as np
import pandas as pd
import subprocess
import datetime as dt

# =========================
# 경로/입출력
# =========================
REPO_DIR = Path(r"C:\work\mygit").resolve()
INPUT_JSON = REPO_DIR / "all_stock_data.json"
OUTPUT_TXT = REPO_DIR / "selected.txt"

# =========================
# 실행/로깅 옵션
# =========================
MODE = os.environ.get("MODE", "LIGHT").upper()  # LIGHT / TIGHT 등
VERBOSE = bool(int(os.environ.get("VERBOSE", "1")))

# --- Git autosave settings ---
GIT_AUTOSAVE = os.environ.get("GIT_AUTOSAVE", "1") == "1"
GIT_REMOTE = os.environ.get("GIT_REMOTE", "origin")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")

# =========================
# 절대필터 파라미터 (환경변수로 튜닝)
# =========================
ABS_STRICT = os.environ.get("ABS_STRICT", "0") == "1"  # True=결측시 제외, False=결측시 통과
MIN_PRICE = float(os.environ.get("MIN_PRICE", "1000"))        # 최저 가격(원)
MAX_PRICE = float(os.environ.get("MAX_PRICE", "9999999999"))  # 최고 가격(원)
MIN_TURNOVER = float(os.environ.get("MIN_TURNOVER", "1e8"))   # 일 거래대금 최소(원) 기본=1억
MIN_MKTCAP_WON = float(os.environ.get("MIN_MKTCAP_WON", "5e10"))  # 시총 최소(원) 기본=500억
EXCLUDE_ST = os.environ.get("EXCLUDE_ST", "1") == "1"         # 관리/투자주의 등 제외

# =========================
# 유틸
# =========================
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def _normalize_currency_to_won(x: Optional[float]) -> Optional[float]:
    """
    원/억 혼용값을 원 단위로 정규화.
    휴리스틱:
    - 1e9 이상이면 이미 '원'으로 본다.
    - 1e2 ~ 1e6 사이면 억단위로 보고 *1e8
    - 그 외는 값 자체를 신뢰.
    """
    if x is None:
        return None
    if x >= 1e9:          # 원 단위로 충분히 큼
        return x
    if 1e2 <= x <= 1e6:   # 억 단위로 흔히 쓰는 범위(대략적)
        return x * 1e8
    return x

def _ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def is_etf_name(name: str) -> bool:
    if not name:
        return False
    bad = ("ETF", "ETN", "리츠", "REIT", "스팩", "SPAC", "인프라", "선물", "Inverse", "인버스")
    name_up = str(name).upper()
    return any(t in name_up for t in bad)

def is_warning_flags(flags: List[str]) -> bool:
    # 종목 플래그(예: 관리종목, 투자주의 등)가 전달되는 경우 제외
    if not flags:
        return False
    bad = ("관리", "투자주의", "투자경고", "정리매매")
    return any(any(b in f for b in bad) for f in flags)

def git_autosave(repo_dir: Path, msg: str) -> None:
    try:
        subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", msg], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "push", GIT_REMOTE, GIT_BRANCH], check=True)
        if VERBOSE:
            print(f"[GIT] push 완료 → {GIT_REMOTE}/{GIT_BRANCH} (msg: {msg})")
    except subprocess.CalledProcessError as e:
        print(f"[GIT] 오류: {e}")

# =========================
# 절대필터 (결측치 내성 + 단위정규화 + 유연모드)
# =========================
def compute_derived_fields(info: Dict, df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    info: {"name","mktcap","shares","last_price","turnover","flags", ...}
    df: OHLCV DataFrame (date, open, high, low, close, volume, ...)

    반환: {"mktcap_won","turnover_won","last_price","shares"}
    """
    name = info.get("name", "")
    last_row = df.iloc[-1] if len(df) else pd.Series(dtype=float)

    # last_price 우선순위: info.last_price -> df.close[-1]
    last_price = _safe_float(info.get("last_price"))
    if last_price is None and "close" in last_row:
        last_price = _safe_float(last_row["close"])

    # turnover(일 거래대금): info.turnover -> df.close[-1]*df.volume[-1]
    turnover = _safe_float(info.get("turnover"))
    if turnover is None and "close" in df.columns and "volume" in df.columns and len(df):
        _close = _safe_float(df["close"].iloc[-1])
        _vol = _safe_float(df["volume"].iloc[-1])
        if _close is not None and _vol is not None:
            turnover = _close * _vol

    # shares: info.shares
    shares = _safe_float(info.get("shares"))

    # mktcap: info.mktcap -> last_price*shares
    mktcap = _safe_float(info.get("mktcap"))
    if mktcap is None and last_price is not None and shares is not None:
        mktcap = last_price * shares

    # 단위 정규화
    mktcap_won = _normalize_currency_to_won(mktcap)
    turnover_won = _normalize_currency_to_won(turnover)

    return {
        "mktcap_won": mktcap_won,
        "turnover_won": turnover_won,
        "last_price": last_price,
        "shares": shares,
        "name": name
    }

def absolute_filters(info: Dict, df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    절대필터: ETF/ETN/리츠/스팩 제외, 가격/거래대금/시총 하한 등.
    - ABS_STRICT=False(기본): 핵심값 결측시 '보류 통과'
    - ABS_STRICT=True: 핵심값 결측시 '제외'
    """
    reasons: List[str] = []
    name = str(info.get("name", ""))

    # (0) 유형 제외
    if is_etf_name(name):
        reasons.append("ETF/ETN/리츠/스팩")
        return False, reasons

    # (1) 경고 플래그
    if EXCLUDE_ST and is_warning_flags(info.get("flags", [])):
        reasons.append("경고/관리/주의 플래그")
        return False, reasons

    # (2) 수치 조건
    fields = compute_derived_fields(info, df)
    price = fields["last_price"]
    turnover_won = fields["turnover_won"]
    mktcap_won = fields["mktcap_won"]

    # 가격
    if price is None:
        if ABS_STRICT:
            reasons.append("가격결측")
            return False, reasons
    else:
        if price < MIN_PRICE:
            reasons.append(f"가격<{int(MIN_PRICE)}")
        if price > MAX_PRICE:
            reasons.append("가격>MAX")

    # 거래대금
    if turnover_won is None:
        if ABS_STRICT:
            reasons.append("거래대금결측")
            return False, reasons
    else:
        if turnover_won < MIN_TURNOVER:
            reasons.append(f"거래대금<{int(MIN_TURNOVER)}")

    # 시총
    if mktcap_won is None:
        if ABS_STRICT:
            reasons.append("시총결측")
            return False, reasons
    else:
        if mktcap_won < MIN_MKTCAP_WON:
            reasons.append(f"시총<{int(MIN_MKTCAP_WON)}")

    if reasons:
        # 이유가 하나라도 있으면 제외
        return False, reasons
    return True, reasons

# =========================
# 20MA 상승 변곡 (필수)
# =========================
def cond_20ma_inflect_up(df: pd.DataFrame,
                         min_len: int = 25,
                         slope_eps: float = -1e-9,
                         return_reason: bool = True) -> Optional[Dict]:
    """
    전제조건: 20MA 기울기 음→양 변곡(마지막 봉 기준)
    """
    if not {"close"}.issubset(df.columns) or len(df) < min_len:
        return None
    ma20 = _ma(df["close"], 20)
    slope = ma20.diff()
    if ma20.isna().iloc[-1] or slope.isna().iloc[-1]:
        return None
    turned_up = (slope.iloc[-2] <= slope_eps) and (slope.iloc[-1] > 0)
    if not turned_up:
        return None
    return {"ok": True, "reason": "20ma_upturn"}

# =========================
# 3MA: 안착/쌍바닥/신선도
# =========================
def cond_3ma_fresh_combo(df: pd.DataFrame,
                         window: int = 18,
                         cross_window: int = 3,
                         min_up_days: int = 2,
                         allow_under_pct: float = 0.0,
                         need_last_up: bool = False,
                         return_reason: bool = True) -> Optional[Dict]:
    """
    - '갓 안착' 신선도: 최근 cross_window 이내에 종가가 3MA 위로 최초 재진입
    - 쌍바닥/변곡 등은 간단화(세부 로직은 기존과 동일 가정)
    """
    if not {"close"}.issubset(df.columns) or len(df) < 25:
        return None

    close = df["close"].astype(float)
    ma3 = _ma(close, 3)

    # 최근 window 범위로 제한
    sub = df.tail(max(window, cross_window+5)).copy()
    sub_close = sub["close"].astype(float)
    sub_ma3 = _ma(sub_close, 3)

    # 최근 cross_window 이내에 아래→위 교차가 있었는지
    cross_idx = []
    for i in range(1, len(sub_ma3)):
        if pd.notna(sub_ma3.iloc[i]) and pd.notna(sub_ma3.iloc[i-1]):
            prev = sub_close.iloc[i-1] - sub_ma3.iloc[i-1]
            now = sub_close.iloc[i] - sub_ma3.iloc[i]
            if prev <= 0 and now > 0:
                cross_idx.append(sub.index[i])

    fresh_ok = False
    if cross_idx:
        # 마지막 교차가 cross_window 이내여야 함
        last_cross_pos = list(sub.index).index(cross_idx[-1])
        if (len(sub) - 1 - last_cross_pos) <= cross_window:
            fresh_ok = True

    # 마지막 봉 조건
    last_close = sub_close.iloc[-1]
    last_ma3 = sub_ma3.iloc[-1]
    if pd.isna(last_ma3):
        return None

    if allow_under_pct <= 0:
        if last_close < last_ma3:
            return None
    else:
        if last_close < last_ma3 * (1 - allow_under_pct):
            return None

    if need_last_up:
        if len(sub_close) >= 2 and not (last_close > sub_close.iloc[-2]):
            return None

    if not fresh_ok:
        return None

    return {"ok": True, "reason": "3ma_fresh_cross"}

# =========================
# 종합 조건(20MA 필수 + 3MA 시그널)
# =========================
def cond_combo(df: pd.DataFrame) -> Optional[Dict]:
    a = cond_20ma_inflect_up(df)
    if not (a and a.get("ok")):
        return None
    b = cond_3ma_fresh_combo(df)
    if b and b.get("ok"):
        return {"ok": True, "cond": f"{a['reason']} + {b['reason']}"}
    return None

# =========================
# 평가 & 저장
# =========================
def eval_symbol(code: str, info: dict) -> List[Tuple[str, str, int, str]]:
    name = info.get("name", code)
    ohlcv = info.get("ohlcv", [])
    if not ohlcv:
        return []
    df = pd.DataFrame(ohlcv)
    # 컬럼 표준화
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        except Exception:
            pass
        df = df.sort_values("date")
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 25:
        return []

    # 절대필터
    ok_abs, reasons_abs = absolute_filters(info, df)
    if not ok_abs:
        return []

    # 전략 조건
    res = cond_combo(df)
    if res and res.get("ok"):
        last_close = int(df["close"].iloc[-1])
        return [(code, name, last_close, res["cond"])]
    return []

def run_all_to_selected(all_json_path: Path = INPUT_JSON, out_txt: Path = OUTPUT_TXT) -> Dict[str, int]:
    if not all_json_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없음: {all_json_path}")
    with all_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    excluded = 0
    schema_miss = 0
    abs_fail = 0
    pass_abs = 0

    results: List[Tuple[str, str, int, str]] = []
    for code, info in data.items():
        try:
            name = info.get("name", code)
            ohlcv = info.get("ohlcv", [])
            if not ohlcv:
                schema_miss += 1
                continue
            df = pd.DataFrame(ohlcv)
            df = df.dropna(subset=["close"]) if "close" in df.columns else df
            if len(df) < 25:
                excluded += 1
                continue

            ok_abs, _reasons = absolute_filters(info, df)
            if not ok_abs:
                abs_fail += 1
                continue
            else:
                pass_abs += 1

            r = eval_symbol(code, info)
            if r:
                results.extend(r)
        except Exception as e:
            if VERBOSE:
                print(f"[WARN] {code}({info.get('name','')}) 처리 중 오류: {e}")

    # 저장 (정렬: 가격 내림차순)
    results.sort(key=lambda x: (-x[2], x[1]))
    with out_txt.open("w", encoding="utf-8") as f:
        for code, name, price, cond in results:
            f.write(f"{code}\t{name}\t{price}\t{cond}\n")

    if VERBOSE:
        print(f"[DONE:{MODE}] 종목 수: {len(results)} → {out_txt}")
        print(f"[STATS] total={total} excluded={excluded} schema_miss={schema_miss}")
        print(f"[STATS] abs_fail={abs_fail} pass_abs={pass_abs}")

    # === GitHub autosave ===
    if GIT_AUTOSAVE:
        commit_msg = f"update selected ({MODE})"
        git_autosave(REPO_DIR, commit_msg)

    return {
        "total": total,
        "results": len(results),
        "excluded": excluded,
        "schema_miss": schema_miss,
        "abs_fail": abs_fail,
        "pass_abs": pass_abs,
    }

# =========================
# 진입점
# =========================
if __name__ == "__main__":
    run_all_to_selected(INPUT_JSON, OUTPUT_TXT)
