# -*- coding: utf-8 -*-
# stockdata_import.py
import sys
import json
import time
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget


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

    def get_code_list(self):
        kospi = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["0"]).split(";")
        kosdaq = self.ocx.dynamicCall("GetCodeListByMarket(QString)", ["10"]).split(";")
        codes = list(filter(None, kospi + kosdaq))
        print(f">> 총 종목 수: {len(codes)}")
        return codes

    def get_stock_name(self, code):
        return self.ocx.dynamicCall("GetMasterCodeName(QString)", [code])

    def is_valid_stock(self, code, name):
        # 간단 제외(추가 필터는 condition.py에서)
        keywords = ["우", "ETF", "ETN", "리츠", "스팩"]
        if any(k in name for k in keywords):
            return False
        status = self.ocx.dynamicCall("GetMasterStockState(QString)", [code])
        if "정지" in status:
            return False
        return True

    def get_ohlcv(self, code):
        self.tr_data = []
        self.data_ready = False

        self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "기준일자", "")
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "수정주가구분", "1")
        self.ocx.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            "opt10081_req",
            "opt10081",
            0,
            "0101",
        )
        while not self.data_ready:
            self.app.processEvents()
        return self.tr_data

    def _on_receive_tr_data(self, screen_no, rqname, trcode, recordname, prev_next):
        count = self.ocx.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        result = []
        for i in range(count):
            date = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "일자",
            ).strip()
            open_ = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "시가",
            ).strip()
            high = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "고가",
            ).strip()
            low = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "저가",
            ).strip()
            close = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "현재가",
            ).strip()
            volume = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                trcode,
                rqname,
                i,
                "거래량",
            ).strip()

            result.append(
                {
                    "date": date,
                    "open": int(open_),
                    "high": int(high),
                    "low": int(low),
                    "close": int(close),
                    "volume": int(volume),
                }
            )

        self.tr_data = result[:120]
        self.data_ready = True


if __name__ == "__main__":
    # 저장 디렉토리 고정: C:\work\mygit
    REPO_DIR = Path(r"C:\work\mygit").resolve()
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = str((REPO_DIR / "all_stock_data.json").resolve())
    print(">> 저장 경로:", SAVE_PATH)

    kiwoom = Kiwoom()
    kiwoom.connect()

    # 기존 JSON 있으면 로드
    all_data = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    done = set(all_data.keys())

    def ensure_login():
        if kiwoom.ocx.dynamicCall("GetConnectState()") == 0:
            print("⚠ 세션 만료 → 재로그인")
            kiwoom.connect()

    codes = [c for c in kiwoom.get_code_list() if c not in done]
    total = len(done) + len(codes)
    print(f"총 {total}개 중, 이미 저장 {len(done)}개, 남은 {len(codes)}개 처리 예정.")

    DRY_RUN = False
    if DRY_RUN:
        codes = codes[:10]
        print(f"DRY_RUN: {len(codes)}개만 시험 실행")

    for idx, code in enumerate(codes, 1):
        name = kiwoom.get_stock_name(code)
        if not kiwoom.is_valid_stock(code, name):
            continue
        ensure_login()
        print(f"[{idx}/{len(codes)}] {code} {name} 요청...")
        try:
            ohlcv = kiwoom.get_ohlcv(code)
            all_data[code] = {"name": name, "ohlcv": ohlcv}
            tmp = SAVE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, SAVE_PATH)
            print(f"  저장 완료: {code}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  >> 실패: {code} {name} - {e}")

    print("✅ 저장 완료:", SAVE_PATH)
