# -*- coding: utf-8 -*-
# stockdata_import.py — resume + daily incremental update
import sys
import json
import time
import os
from pathlib import Path
from typing import Dict, List
from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget

# ===== 저장 경로(조건/깃과 동일 폴더로 통일) =====
REPO_DIR = Path(r"C:\work\mygit").resolve()
REPO_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = str((REPO_DIR / "all_stock_data.json").resolve())

# ===== 실행 모드 =====
# MODE=FULL   : 처음 구축/중단 재개 (이미 저장된 종목은 건너뜀)
# MODE=UPDATE : 매일 업데이트 (기존 저장 종목만 N개만 받아 병합)   ← 기본
MODE = os.environ.get("MODE", "UPDATE").upper()

# UPDATE 시, 각 종목에서 최신 몇 개 캔들만 받아 병합할지 (너무 작게 하면 휴장/공휴일에 빈 업데이트 가능)
COUNT_LIMIT = int(os.environ.get("COUNT_LIMIT", "12"))

# 조회 간격(초) — 과다조회 방지
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "0.25"))


class Kiwoom:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")

        self.ocx.OnEventConnect.connect(self._on_login)
        self.ocx.OnReceiveTrData.connect(self._on_receive_tr_data)

        self.login_ok = False
        self.tr_data = None
        self.data_ready = False

    # 로그인
    def connect(self):
        self.ocx.dynamicCall("CommConnect()")
        self.app.exec_()

    def _on_login(self, err_code):
        if err_code == 0:
            print(">> 로그인 성공")
            self.login_ok = True
        else:
            print(">> 로그인 실패")
        self.app.quit()

    def ensure_login(self):
        if self.ocx.dynamicCall("GetConnectState()") == 0:
            print("⚠ 세션 만료 → 재로그인")
            self.connect()

    def get_code_list(self) -> List[str]:
        kospi = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["0"]).split(";")
        kosdaq = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["10"]).split(";")
        return list(filter(None, kospi + kosdaq))

    def get_stock_name(self, code: str) -> str:
        return self.ocx.dynamicCall("GetMasterCodeName(QString)", [code])

    def is_valid_stock(self, code: str, name: str) -> bool:
        # 간단 제외 (세부 ETF/ETN 필터는 condition.py에서 재확인)
        for k in ("우", "ETF", "ETN", "리츠", "스팩"):
            if k in name:
                return False
        status = self.ocx.dynamicCall("GetMasterStockState(QString)", [code])
        if "정지" in status:
            return False
        return True

    def get_ohlcv(self, code: str, count_limit: int = 120) -> List[Dict]:
        """opt10081: 일봉 OHLCV. 최신순으로 수신 → 최신 count_limit개만 잘라 반환."""
        self.tr_data = []
        self.data_ready = False

        self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "기준일자", "")
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "수정주가구분", "1")
        self.ocx.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            "opt10081_req", "opt10081", 0, "0101"
        )
        while not self.data_ready:
            self.app.processEvents()

        # 최신순 → 최신 count_limit 개만
        data = self.tr_data[:max(1, int(count_limit))]
        return data

    def _on_receive_tr_data(self, screen_no, rqname, trcode, recordname, prev_next):
        count = self.ocx.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        result = []
        for i in range(count):
            date = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                        trcode, rqname, i, "일자").strip()
            open_ = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                         trcode, rqname, i, "시가").strip()
            high = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                        trcode, rqname, i, "고가").strip()
            low = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                       trcode, rqname, i, "저가").strip()
            close = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                         trcode, rqname, i, "현재가").strip()
            volume = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                          trcode, rqname, i, "거래량").strip()

            try:
                result.append({
                    "date": date,
                    "open": int(open_),
                    "high": int(high),
                    "low": int(low),
                    "close": int(close),
                    "volume": int(volume)
                })
            except Exception:
                # 숫자 파싱 실패 row 스킵
                continue

        # 키움은 최신이 앞쪽에 위치하는 배열(최근 → 과거)로 들어옴
        self.tr_data = result
        self.data_ready = True


def load_all() -> Dict:
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def atomic_save(obj: Dict):
    tmp = SAVE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SAVE_PATH)


def merge_by_date(existing: List[Dict], fetched: List[Dict], keep_max: int = 240) -> List[Dict]:
    """
    날짜키 기준으로 fetched가 최신을 덮어쓰도록 병합.
    저장 포맷은 최신→과거 순서가 아니라 condition.py에서 정렬하므로 순서 자유.
    여기서는 간단히 '날짜 사전'로 통합.
    """
    by_date: Dict[str, Dict] = {row["date"]: row for row in existing if "date" in row}
    for row in fetched:
        d = row.get("date")
        if d:
            by_date[d] = row  # 덮어쓰기
    # 날짜 역순(최신→과거)로 정렬 후 최대 keep_max개만 유지
    merged = sorted(by_date.values(), key=lambda r: r["date"], reverse=True)[:keep_max]
    return merged


if __name__ == "__main__":
    print(">> 저장 경로:", SAVE_PATH)
    kiwoom = Kiwoom()
    kiwoom.connect()  # 로그인

    all_data = load_all()
    all_codes = kiwoom.get_code_list()

    if MODE == "FULL":
        # 이어 저장(Resume): 기존에 없는 종목만 수집
        done = set(all_data.keys())
        codes = [c for c in all_codes if c not in done]
        print(f"[FULL] 총 {len(all_codes)}개 중, 이미 저장 {len(done)}개 → 수집 {len(codes)}개")
    else:
        # UPDATE: 이미 저장된 종목만 최신 N개 받아 병합
        codes = list(all_data.keys())
        print(f"[UPDATE] 기존 저장 종목 {len(codes)}개만 최신 {COUNT_LIMIT}개 캔들 병합")

    # 선택적으로 DRY_RUN
    DRY_RUN = os.environ.get("DRY_RUN", "0") in ("1", "true", "yes")
    if DRY_RUN:
        codes = codes[:10]
        print(f"(DRY_RUN) {len(codes)}개만 실행")

    for idx, code in enumerate(codes, 1):
        name = kiwoom.get_stock_name(code)
        if MODE == "FULL" and not kiwoom.is_valid_stock(code, name):
            continue

        kiwoom.ensure_login()
        print(f"[{idx}/{len(codes)}] {code} {name} 요청...", end="")
        try:
            data = kiwoom.get_ohlcv(code, count_limit=(120 if MODE == "FULL" else COUNT_LIMIT))
            if MODE == "FULL":
                all_data[code] = {"name": name, "ohlcv": data}
            else:
                # UPDATE: 병합
                existing = all_data.get(code, {"name": name, "ohlcv": []})
                merged = merge_by_date(existing.get("ohlcv", []), data, keep_max=240)
                all_data[code] = {"name": name, "ohlcv": merged}
            atomic_save(all_data)
            print(" 저장완료")
        except Exception as e:
            print(f" 실패: {e}")
        time.sleep(SLEEP_SEC)

    print("✅ 완료:", SAVE_PATH)
