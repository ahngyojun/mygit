# -*- coding: utf-8 -*-
# stockdata_import.py — Kiwoom OpenAPI 데이터 수집 (속도개선판)
# 기능: 증분 갱신(UPDATE) + 신규상장 자동 편입 + 최근 240일 롤링 유지
# 개선: 배치저장, ETF/ETN/리츠/스팩 사전제외, 오늘자 보유시 스킵
# 실행: python C:\work\mygit\stockdata_import.py

import sys
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# ===== 콘솔 즉시출력 =====
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget

# ================== 설정 ==================
REPO_DIR = Path(r"C:\work\mygit").resolve()
REPO_DIR.mkdir(parents=True, exist_ok=True)

SAVE_FILE = REPO_DIR / "all_stock_data.json"
TEMP_FILE = REPO_DIR / "all_stock_data.json.tmp"

MODE = os.environ.get("MODE", "UPDATE").upper()  # UPDATE(기본) / FULL
RECENT_FETCH = int(os.environ.get("RECENT_FETCH", "15"))
MAX_DAYS = int(os.environ.get("MAX_DAYS", "240"))
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "0.25"))
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))  # 0=off
NEW_BOOTSTRAP_MAX = int(os.environ.get("NEW_BOOTSTRAP_MAX", "50"))
REMOVE_DELISTED = os.environ.get("REMOVE_DELISTED", "0") == "1"
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "40"))  # ⬅ 배치저장 단위

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
    if not name: return False
    u = str(name).upper().strip()
    if any(k in u for k in _ETF_BRANDS): return True
    if any(k in u for k in _ETF_STYLE): return True
    if any(k in str(name) for k in ("우", "스팩", "리츠")): return True
    return False

# ================== 유틸 ==================
def atomic_save(obj: dict, path: Path, tmp_path: Path):
    """임시파일에 쓴 뒤 원자적 교체."""
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_path), str(path))

def start_tmp_housekeeping():
    """비정상 종료로 남은 tmp 정리/복원."""
    if TEMP_FILE.exists():
        if not SAVE_FILE.exists():
            print("⚠️ 이전 tmp 복원 → 본 파일로 승격")
            os.replace(str(TEMP_FILE), str(SAVE_FILE))
        else:
            print("⚠️ 이전 tmp 정리(삭제)")
            try: TEMP_FILE.unlink()
            except Exception as e: print("   tmp 삭제 실패:", e)

def normalize_all_data(all_data: Dict[str, Dict], *, save_after=True) -> int:
    """정렬/중복제거/롤링 유지."""
    changed = 0
    for code, v in list(all_data.items()):
        ohlcv = v.get("ohlcv", [])
        if not isinstance(ohlcv, list):
            all_data.pop(code, None)
            changed += 1
            continue
        seen, uniq = set(), []
        for r in sorted(ohlcv, key=lambda x: x.get("date", "")):
            d = r.get("date")
            if d and d not in seen:
                uniq.append(r); seen.add(d)
        trimmed = uniq[-MAX_DAYS:] if MAX_DAYS > 0 else uniq
        if len(trimmed) != len(ohlcv):
            all_data[code]["ohlcv"] = trimmed; changed += 1
    if save_after and changed:
        atomic_save(all_data, SAVE_FILE, TEMP_FILE)
        print(f">> 데이터 정규화: {changed}종목 ({MAX_DAYS}개 유지)")
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

    def connect(self):
        print(">> 키움 로그인 시도중...")
        self.ocx.dynamicCall("CommConnect()")
        self.app.exec_()

    def _on_login(self, err_code):
        print(">> 로그인 성공" if err_code == 0 else f">> 로그인 실패 ({err_code})")
        self.login_ok = (err_code == 0)
        self.app.quit()

    def _on_receive_tr_data(self, screen_no, rqname, trcode, recordname, prev_next):
        cnt = self.ocx.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        parsed: List[Dict] = []
        def get(field, i):
            return self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode, rqname, i, field
            ).strip()
        for i in range(cnt):
            def to_int(s):
                try: return abs(int(s))
                except: return 0
            date = get("일자", i)
            open_ = to_int(get("시가", i))
            high  = to_int(get("고가", i))
            low   = to_int(get("저가", i))
            close = to_int(get("현재가", i))
            vol   = to_int(get("거래량", i))
            if date:
                parsed.append({
                    "date": date, "open": open_, "high": high,
                    "low": low, "close": close, "volume": vol
                })
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
        self.data_ready = False
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "기준일자", "")
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "수정주가구분", "1")
        self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)",
                             "opt10081_req", "opt10081", 0, "0101")
        while not self.data_ready:
            self.app.processEvents()
            time.sleep(0.05)
        data = sorted(self.tr_data, key=lambda x: x["date"])
        return data[-n:] if n > 0 else data

# ================== 메인 ==================
def main():
    start_tmp_housekeeping()

    # 1) 백업
    if SAVE_FILE.exists():
        bak = SAVE_FILE.with_name(f"all_stock_data_{datetime.now():%Y%m%d_%H%M%S}.json.bak")
        bak.write_bytes(SAVE_FILE.read_bytes())
        print(">> 백업 생성:", bak)

        # --- 오래된 백업 자동 정리 (최근 3개만 유지) ---
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

        print(f"[{idx}/{total}] {code} {name} 데이터 요청 중...")

        try:
            exist = sorted(all_data.get(code, {"ohlcv": []})["ohlcv"], key=lambda x: x["date"])
            last_date = exist[-1]["date"] if exist else None

            # 오늘자까지 있으면 스킵
            if MODE == "UPDATE" and last_date and last_date >= TODAY:
                if code in all_data and all_data[code].get("name") != name:
                    all_data[code]["name"] = name
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

            all_data[code] = {"name": name, "ohlcv": merged}
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

# ================== 엔트리 ==================
if __name__ == "__main__":
    main()
