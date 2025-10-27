# -*- coding: utf-8 -*-
# stockdata_import.py — Kiwoom OpenAPI 데이터 수집 (속도개선판, Py38 호환)
# 기능: 증분 갱신(UPDATE) + 신규상장 자동 편입 + 최근 240일 롤링 유지
# 개선: 배치저장, ETF/ETN/리츠/스팩 사전제외, 오늘자 보유시 스킵
# 추가: opt10001(현재가/시총/상장주식수/거래량) 수집, 거래대금 저장, 저장 원자화 + 재시도,
#       Git autosave, .gitignore 자동갱신, TR 속도제한(Throttle) + 백오프 재시도
# 실행: python C:\work\mygit\stockdata_import.py

import sys
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ===== 콘솔 즉시출력 =====
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget
import subprocess

# ================== 설정 ==================
REPO_DIR = Path(r"C:\work\mygit").resolve()
REPO_DIR.mkdir(parents=True, exist_ok=True)

SAVE_FILE = REPO_DIR / "all_stock_data.json"
TEMP_FILE = REPO_DIR / "all_stock_data.json.tmp"

MODE = os.environ.get("MODE", "UPDATE").upper()  # UPDATE(기본) / FULL
RECENT_FETCH = int(os.environ.get("RECENT_FETCH", "15"))
MAX_DAYS = int(os.environ.get("MAX_DAYS", "240"))
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "0.25"))
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))  # 0=off → N개만 시험
NEW_BOOTSTRAP_MAX = int(os.environ.get("NEW_BOOTSTRAP_MAX", "50"))
REMOVE_DELISTED = os.environ.get("REMOVE_DELISTED", "0") == "1"
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "40"))  # ⬅ 배치저장 단위

# ===== TR 속도 제한(Throttle) & 재시도 =====
REQUEST_GAP_SEC    = 0.70   # 각 TR 사이 최소 간격(초) 0.6~0.9 권장
LONG_PAUSE_EVERY   = 120    # N회마다
LONG_PAUSE_SEC     = 6.0    # 길게 쉬기(서버 쿨다운)
TR_TIMEOUT_SEC     = 7.0    # 한 TR 대기 타임아웃
MAX_TR_RETRIES     = 4      # 과도요청/무응답 시 재시도 횟수
BACKOFF_BASE_SEC   = 2.0    # 재시도 백오프 시작값(2,4,6,...)

# ================== Git 자동설정 / 자동저장 ==================
GIT_AUTOSAVE = True          # 자동 저장 on/off
GIT_REMOTE = "origin"        # 원격 이름
GIT_BRANCH = "main"          # 기본 브랜치 (현재 브랜치 자동 감지)

def ensure_gitignore_full(repo_dir: Path):
    """
    .gitignore에 백업/데이터/시스템 파일 제외 패턴 추가 (자동)
    """
    gitignore_path = repo_dir / ".gitignore"
    patterns = [
        # Backup & cache (대용량)
        "*.bak", "*_bak.json", "*_backup.json", "*.json.bak",
        "all_stock_data_*.json.bak",
        # Local data (대용량/캐시성)
        "all_stock_data.json",
        "selected_debug.json",
        # System / IDE
        "__pycache__/", "*.pyc", ".idea/", ".vscode/", ".DS_Store",
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

def _run_git(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def git_autosave(repo_dir: Path, msg: str, patterns: Optional[List[str]] = None) -> None:
    """
    변경 사항이 있으면 add/commit/push.
    patterns 지정 시 해당 패턴만 add (예: ['*.py']) → JSON/BAK는 커밋 안 함.
    """
    try:
        cwd = str(repo_dir)
        if not (repo_dir / ".git").exists():
            print("[GIT] .git 폴더 없음 → autosave 스킵")
            return
        cur = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
        current_branch = (cur.stdout or "").strip() or GIT_BRANCH
        if patterns:
            for pat in patterns:
                _run_git(["git", "add", pat], cwd)
        else:
            _run_git(["git", "add", "-A"], cwd)
        diffq = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd)
        if diffq.returncode == 0:
            print("[GIT] 변경 사항 없음 → 스킵")
            return
        cr = _run_git(["git", "commit", "-m", msg], cwd)
        if cr.returncode != 0:
            print(f"[GIT][COMMIT ERR]\n{cr.stderr}")
            return
        rr = _run_git(["git", "remote"], cwd)
        remotes = (rr.stdout or "").split()
        if GIT_REMOTE not in remotes:
            print(f"[GIT] 원격 '{GIT_REMOTE}' 없음 → push 스킵")
            return
        target_branch = current_branch or GIT_BRANCH
        pr = _run_git(["git", "push", GIT_REMOTE, target_branch], cwd)
        if pr.returncode != 0:
            print(f"[GIT][PUSH ERR]\n{pr.stderr}")
        else:
            print(f"[GIT] push 완료 → {GIT_REMOTE}/{target_branch}")
    except Exception as e:
        print(f"[GIT][ERR] {type(e).__name__}: {e}")

# ================== ETF/ETN/리츠/스팩 제외 ==================
_ETF_BRANDS = {
    "ETF","ETN","KODEX","TIGER","KBSTAR","KOSEF","ARIRANG","HANARO","ACE",
    "KINDEX","TIMEFOLIO","TREX","SMART","FOCUS","MARO","SOL","PLUS","RISE",
    "ITF","HVOL","QV","HANBIT","NEOS","TOME"
}
_ETF_STYLE = {
    "INVERSE","레버리지","인버스","커버드콜","TRF","합성","선물","WTI","BRENT",
    "GOLD","SILVER","NICKEL","COPPER","GAS","원유","금현물","은현물","구리","니켈",
    "천연가스","S&P","NASDAQ","나스닥","미국","중국","유로스톡스","선진국","신흥국",
    "TOP10","TOP5","퀄리티","모멘텀","밸류","고배당","배당","채권혼합","채권","금리"
}
def is_etf_name(name: str) -> bool:
    """ETF/ETN/리츠/스팩/우선주/대표지수형 키워드 제외"""
    if not name:
        return False
    u = str(name).upper().strip()
    if any(k in u for k in _ETF_BRANDS):
        return True
    if any(k in u for k in _ETF_STYLE):
        return True
    if any(k in str(name) for k in ("우", "스팩", "리츠")):
        return True
    return False

# ================== 유틸 ==================
def _to_int(s: str) -> int:
    """ '1,234,567' / '+123' / ' -1,000 ' 등을 정수로 안전 변환 """
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    s = s.replace("(", "-").replace(")", "")
    filtered = "".join(ch for ch in s if ch.isdigit() or ch in "+-")
    try:
        return int(filtered)
    except Exception:
        try:
            return int(s.replace(",", "").replace("+", "").replace(" ", ""))
        except Exception:
            return 0

def atomic_save(obj: dict, path: Path, tmp_path: Path, *, retries: int = 5, delay: float = 0.2):
    """
    임시파일에 쓴 뒤 원자적 교체. Windows 파일 잠금 이슈에 대비해 재시도.
    """
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            os.replace(str(tmp_path), str(path))
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
        except Exception as e:
            last_err = e
            time.sleep(delay)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
    except Exception as e:
        raise last_err or e

def start_tmp_housekeeping():
    """비정상 종료로 남은 tmp 정리/복원."""
    if TEMP_FILE.exists():
        if not SAVE_FILE.exists():
            print("⚠️ 이전 tmp 복원 → 본 파일로 승격")
            os.replace(str(TEMP_FILE), str(SAVE_FILE))
        else:
            print("⚠️ 이전 tmp 정리(삭제)")
            try:
                TEMP_FILE.unlink()
            except Exception as e:
                print("   tmp 삭제 실패:", e)

def normalize_all_data(all_data: Dict[str, Dict], *, save_after=True) -> int:
    """
    정렬/중복제거/롤링 유지 + 간단 보정:
      - OHLCV 중복 제거 및 최신 MAX_DAYS만 유지
      - market_cap==0 이고 price*shares>0이면 계산값으로 백필
    """
    changed = 0
    for code, v in list(all_data.items()):
        ohlcv = v.get("ohlcv", [])
        if not isinstance(ohlcv, list):
            all_data.pop(code, None)
            changed += 1
            continue

        # 1) OHLCV 정렬/중복 제거/롤링 유지
        seen, uniq = set(), []
        for r in sorted(ohlcv, key=lambda x: x.get("date", "")):
            d = r.get("date")
            if d and d not in seen:
                uniq.append(r)
                seen.add(d)
        trimmed = uniq[-MAX_DAYS:] if MAX_DAYS > 0 else uniq
        if len(trimmed) != len(ohlcv):
            all_data[code]["ohlcv"] = trimmed
            changed += 1

        # 2) 시총 백필
        price = abs(int(v.get("price", 0))) if isinstance(v.get("price", 0), int) else abs(_to_int(v.get("price", 0)))
        shares = int(v.get("shares_outstanding", 0)) if isinstance(v.get("shares_outstanding", 0), int) else _to_int(v.get("shares_outstanding", 0))
        mcap = int(v.get("market_cap", 0)) if isinstance(v.get("market_cap", 0), int) else _to_int(v.get("market_cap", 0))
        if mcap <= 0 and price > 0 and shares > 0:
            all_data[code]["market_cap"] = price * shares
            changed += 1

        # 3) 거래대금 백필
        vol = int(v.get("volume", 0)) if isinstance(v.get("volume", 0), int) else _to_int(v.get("volume", 0))
        tv = int(v.get("trading_value", 0)) if isinstance(v.get("trading_value", 0), int) else _to_int(v.get("trading_value", 0))
        if tv <= 0 and price > 0 and vol > 0:
            all_data[code]["trading_value"] = price * vol
            changed += 1

    if save_after and changed:
        atomic_save(all_data, SAVE_FILE, TEMP_FILE)
        print(f">> 데이터 정규화: {changed}종목 ({MAX_DAYS}개 유지/백필)")
    return changed

# ================== 키움 래퍼 ==================
class Kiwoom:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        self.ocx.OnEventConnect.connect(self._on_login)
        self.ocx.OnReceiveTrData.connect(self._on_receive_tr_data)
        self.login_ok = False
        self.data_ready = False
        self.tr_data: List[Dict] = []

        # throttle 상태
        self._last_tr_time = 0.0
        self._tr_count = 0

    # ---- throttle & retry ----
    def _throttle(self):
        # 최소 간격
        since = time.time() - self._last_tr_time
        if since < REQUEST_GAP_SEC:
            time.sleep(REQUEST_GAP_SEC - since)
        # 잔여 대기시간 반영
        try:
            remain_ms = int(self.ocx.dynamicCall("GetCommRemainTime()"))
            if remain_ms > 0:
                time.sleep(remain_ms / 1000.0 + 0.05)
        except Exception:
            pass
        # 주기적 휴식
        self._tr_count += 1
        if self._tr_count % LONG_PAUSE_EVERY == 0:
            time.sleep(LONG_PAUSE_SEC)

    def _with_retry(self, send_tr_func, parse_func, *, desc: str):
        for attempt in range(1, MAX_TR_RETRIES + 1):
            try:
                self.data_ready = False
                self._throttle()
                send_tr_func()

                t0 = time.time()
                while not self.data_ready:
                    self.app.processEvents()
                    if time.time() - t0 > TR_TIMEOUT_SEC:
                        raise TimeoutError(f"{desc} timeout")
                    time.sleep(0.05)

                out = parse_func()
                self._last_tr_time = time.time()
                # 비응답/빈응답은 재시도
                if out is None or (hasattr(out, "__len__") and len(out) == 0):
                    raise RuntimeError(f"{desc} empty")
                return out

            except Exception as e:
                wait = BACKOFF_BASE_SEC * attempt  # 2,4,6,8...
                print(f"⚠️ {desc} 재시도 {attempt}/{MAX_TR_RETRIES} ({e}), {wait:.1f}s 대기")
                time.sleep(wait)
        return None

    # ---- 기본 기능 ----
    def connect(self):
        print(">> 키움 로그인 시도중...")
        self.ocx.dynamicCall("CommConnect()")
        self.app.exec_()

    def _on_login(self, err_code):
        print(">> 로그인 성공" if err_code == 0 else f">> 로그인 실패 ({err_code})")
        self.login_ok = (err_code == 0)
        self.app.quit()

    def _on_receive_tr_data(self, screen_no, rqname, trcode, recordname, prev_next):
        # opt10081(일봉)만 파싱; opt10001은 개별 메서드에서 직접 꺼냄
        try:
            cnt = self.ocx.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        except Exception:
            cnt = 0
        parsed: List[Dict] = []
        def get(field, i):
            return self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode, rqname, i, field
            ).strip()
        for i in range(cnt):
            def to_int(s):
                try:
                    return abs(int(s))
                except:
                    return 0
            date = get("일자", i)
            open_ = to_int(get("시가", i))
            high  = to_int(get("고가", i))
            low   = to_int(get("저가", i))
            close = to_int(get("현재가", i))
            vol   = to_int(get("거래량", i))
            if date:
                parsed.append({"date": date, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
        self.tr_data = parsed
        self.data_ready = True

    def ensure_login(self):
        if self.ocx.dynamicCall("GetConnectState()") == 0:
            print("⚠️ 세션 만료 → 재로그인")
            self.connect()

    def get_code_list(self) -> List[str]:
        kospi = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["0"]).split(";")
        kosdaq = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["10"]).split(";")
        return [c for c in kospi + kosdaq if c]

    def get_master_name(self, code: str) -> str:
        return self.ocx.dynamicCall("GetMasterCodeName(QString)", [code])

    def get_ohlcv(self, code: str, n: int = 15) -> List[Dict]:
        """최근 n개(오름차순 반환)."""
        def _send():
            self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
            self.ocx.dynamicCall("SetInputValue(QString, QString)", "기준일자", "")
            self.ocx.dynamicCall("SetInputValue(QString, QString)", "수정주가구분", "1")
            self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", "opt10081_req", "opt10081", 0, "0101")
        def _parse():
            data = sorted(self.tr_data, key=lambda x: x["date"])
            return data[-n:] if n > 0 else data
        out = self._with_retry(_send, _parse, desc=f"{code} opt10081")
        return out or []

    def get_basic_info(self, code: str) -> dict:
        """
        opt10001 주식기본정보요청
        반환: {'price': 현재가, 'market_cap': 시가총액(원), 'shares_outstanding': 상장주식수,
              'volume': 거래량, 'trading_value': 거래대금(원)}
        (보강) 시총 필드 누락/편차 대응: '시가총액(억)' 환산 + price*shares 보정
        """
        def _send():
            self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
            self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", "opt10001_req", "opt10001", 0, "1101")

        def _parse():
            def g(field):
                return self.ocx.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    "opt10001", "opt10001_req", 0, field
                ).strip()

            price  = _to_int(g("현재가"))
            vol    = _to_int(g("거래량"))
            shares = _to_int(g("상장주식수"))

            mcap_raw = _to_int(g("시가총액"))  # 원 단위일 수도, 빈문자열일 수도
            if mcap_raw <= 0:
                mcap_eok = _to_int(g("시가총액(억)"))
                if mcap_eok > 0:
                    mcap_raw = mcap_eok * 100_000_000  # 억 → 원

            # 계산 보정
            calc_mcap = abs(price) * max(0, shares)
            market_cap = mcap_raw if mcap_raw > 0 else calc_mcap

            # 응답값 vs 계산값 차이 클 때(>5%) 계산값 채택
            if mcap_raw > 0 and calc_mcap > 0:
                diff = abs(mcap_raw - calc_mcap) / max(mcap_raw, calc_mcap)
                if diff > 0.05:
                    market_cap = calc_mcap
                    print(f"(i) {code} 시총 보정: resp={mcap_raw:,} → calc={calc_mcap:,}")

            trading_value = abs(price) * max(0, vol)

            # 경고 로깅
            if shares == 0 or market_cap == 0:
                print(f"(w) {code} opt10001 누락 감지 → shares={shares}, mcap={market_cap}, price={price}")

            return {
                "price": price,
                "market_cap": market_cap,
                "shares_outstanding": shares,
                "volume": vol,
                "trading_value": trading_value,
            }

        out = self._with_retry(_send, _parse, desc=f"{code} opt10001")
        return out or {"price": 0, "market_cap": 0, "shares_outstanding": 0, "volume": 0, "trading_value": 0}

# ================== 메인 ==================
def main():
    # .gitignore 자동 갱신 (대용량/백업/캐시 제외)
    ensure_gitignore_full(REPO_DIR)

    start_tmp_housekeeping()

    # 1) 백업 + 오래된 백업 정리
    if SAVE_FILE.exists():
        bak = SAVE_FILE.with_name(f"all_stock_data_{datetime.now():%Y%m%d_%H%M%S}.json.bak")
        bak.write_bytes(SAVE_FILE.read_bytes())
        print(">> 백업 생성:", bak)
        backups = sorted(SAVE_FILE.parent.glob("all_stock_data_*.json.bak"), key=os.path.getmtime)
        if len(backups) > 3:
            for old in backups[:-3]:
                try:
                    old.unlink()
                    print(f">> 오래된 백업 삭제: {old.name}")
                except Exception as e:
                    print(f"(i) 백업 삭제 실패: {old.name} ({e})")

    # 2) 저장본 로드
    all_data: Dict[str, Dict] = {}
    if SAVE_FILE.exists():
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        print(f">> 저장본 로드: {len(all_data):,} 종목")
    else:
        print(f">> 저장본 없음: {SAVE_FILE}")

    # 3) 키움 연결
    kiwoom = Kiwoom()
    kiwoom.connect()

    # 4) 전체 코드 목록
    ALL_CODES = set(kiwoom.get_code_list())
    EXISTING = set(all_data.keys())

    if MODE == "UPDATE" and not all_data:
        raise RuntimeError("UPDATE인데 저장본이 비었습니다.")

    # 신규 상장 자동 편입
    new_codes = sorted(ALL_CODES - EXISTING)
    if MODE == "UPDATE" and new_codes:
        orig_cnt = len(new_codes)
        new_codes = new_codes[:NEW_BOOTSTRAP_MAX]
        print(f"※ 신규 상장 감지: {orig_cnt}개 중 {len(new_codes)}개 편입 예정")

    # 상장폐지 제거(옵션)
    delisted = sorted(EXISTING - ALL_CODES)
    if REMOVE_DELISTED and delisted:
        for c in delisted:
            all_data.pop(c, None)
        atomic_save(all_data, SAVE_FILE, TEMP_FILE)
        print(f"※ 상장폐지 {len(delisted)}개 제거")

    # 실행 대상
    if MODE == "UPDATE":
        loop_codes = list(EXISTING) + new_codes
    else:
        loop_codes = sorted(list(ALL_CODES))

    if DRY_RUN > 0:
        loop_codes = loop_codes[:DRY_RUN]
        print(f"※ DRY_RUN={DRY_RUN} → {len(loop_codes)}종목만")

    total = len(loop_codes)
    print(f">> 총 {total}종목 처리 예정")

    if all_data:
        normalize_all_data(all_data, save_after=True)

    # ===== 본 루프 =====
    changed = 0
    TODAY = datetime.now().strftime("%Y%m%d")
    for idx, code in enumerate(loop_codes, 1):
        kiwoom.ensure_login()
        name = kiwoom.get_master_name(code) or ""
        if not name:
            continue
        if is_etf_name(name):
            print(f"[{idx}/{total}] {code} {name} → ETF/ETN/리츠/스팩 제외")
            continue

        # ✅ 기본정보(시총/가격/거래량/거래대금) 재활용: 기존 데이터가 있으면 opt10001 생략
        prev = all_data.get(code, {})
        need_basic = (
            ("market_cap" not in prev) or
            (prev.get("market_cap", 0) <= 0) or
            (code in new_codes) or
            (MODE == "FULL")
        )

        if need_basic:
            # 신규/누락/풀모드만 opt10001 호출
            base = kiwoom.get_basic_info(code)
        else:
            # 기존 저장본 재활용
            base = {
                "price": prev.get("price", 0),
                "market_cap": prev.get("market_cap", 0),
                "shares_outstanding": prev.get("shares_outstanding", 0),
                "volume": prev.get("volume", 0),
                "trading_value": prev.get("trading_value", 0) or abs(prev.get("price", 0)) * max(0, prev.get("volume", 0)),
            }

        print(f"[{idx}/{total}] {code} {name} 데이터 요청 중...")

        try:
            exist = sorted(all_data.get(code, {"ohlcv": []})["ohlcv"], key=lambda x: x["date"])
            last_date = exist[-1]["date"] if exist else None

            # 오늘자까지 있으면 스킵 (이름/기본정보 갱신만 반영)
            if MODE == "UPDATE" and last_date and last_date >= TODAY:
                if code in all_data and all_data[code].get("name") != name:
                    all_data[code]["name"] = name
                if code in all_data:
                    all_data[code]["price"] = base.get("price", all_data[code].get("price", 0))
                    all_data[code]["market_cap"] = base.get("market_cap", all_data[code].get("market_cap", 0))
                    all_data[code]["shares_outstanding"] = base.get("shares_outstanding", all_data[code].get("shares_outstanding", 0))
                    all_data[code]["volume"] = base.get("volume", all_data[code].get("volume", 0))
                    all_data[code]["trading_value"] = base.get("trading_value", all_data[code].get("trading_value", 0))
                print(f"  건너뜀: 오늘자까지 보유 (last={last_date})")
                continue

            # 신규 vs 기존
            if code in all_data:
                recent = kiwoom.get_ohlcv(code, n=RECENT_FETCH)
            else:
                bootstrap_n = MAX_DAYS if MAX_DAYS > 0 else 240
                recent = kiwoom.get_ohlcv(code, n=bootstrap_n)

            # 증분 병합
            seen = {x["date"] for x in exist}
            to_add = [r for r in recent if r["date"] not in seen]
            merged = exist + to_add
            if MAX_DAYS > 0:
                merged = merged[-MAX_DAYS:]

            all_data[code] = {
                "name": name,
                "price": base.get("price", 0),
                "market_cap": base.get("market_cap", 0),
                "shares_outstanding": base.get("shares_outstanding", 0),
                "volume": base.get("volume", 0),
                "trading_value": base.get("trading_value", 0),
                "ohlcv": merged,
            }
            changed += 1

            # N개마다 배치 저장
            if changed % SAVE_EVERY == 0:
                atomic_save(all_data, SAVE_FILE, TEMP_FILE)
                print(f"  ⏺ 배치 저장 ({changed} changes)")

            print(f"  저장 예정: +{len(to_add)}개, 총 {len(merged)}개 (last={merged[-1]['date'] if merged else 'NA'})")

        except Exception as e:
            print(f"  >> 실패: {code} {name} - {e}")

        time.sleep(SLEEP_SEC)

    # 마지막 저장
    if changed:
        atomic_save(all_data, SAVE_FILE, TEMP_FILE)
    print("✅ 전체 완료 →", SAVE_FILE)

    # === GitHub autosave: .py만 커밋/푸시 (JSON/BAK는 .gitignore로 제외) ===
    if GIT_AUTOSAVE:
        git_autosave(REPO_DIR, msg=f"update stockdata_import ({MODE})", patterns=["*.py"])

# ================== 엔트리 ==================
if __name__ == "__main__":
    main()
