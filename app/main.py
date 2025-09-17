from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import pandas as pd
import joblib
from .schema import TxItem, PredictResponse
import io

ARTIFACT_PATH = "models/fraud_xgb_artifact.pkl"
artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
threshold = artifact["threshold"]

app = FastAPI(title="Fraud Detector API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=List[PredictResponse])
def predict(items: List[TxItem]):
    X = pd.DataFrame([i.model_dump() for i in items])
    proba = model.predict_proba(X)[:, 1]
    label = (proba >= threshold).astype(int)
    return [PredictResponse(proba=float(p), label=int(l)) for p, l in zip(proba, label)]

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), return_csv: bool = True):
    # 1) 確認副檔名
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    # 2) 讀取 CSV（處理常見編碼）
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))                 # 預設 UTF-8
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="cp950")  # Windows-繁中常見
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Read CSV error: {e}")

    # 3) 欄位檢查（必要欄位需與訓練時一致）
    required_cols = [
        "cc_num", "trans_date_trans_time", "merchant", "category", "gender",
        "city", "amt", "unix_time", "lat", "long", "merch_lat", "merch_long"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    # 4) 推論
    try:
        proba = model.predict_proba(df)[:, 1]
        label = (proba >= threshold).astype(int)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    # 5A) 回傳 CSV：在原檔加上欄位
    if return_csv:
        out = df.copy()
        out["fraud_proba"] = proba
        out["fraud_label"] = label
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="pred_{file.filename}"'}
        )

    # 5B) 回傳 JSON（僅結果）
    result = [{"proba": float(p), "label": int(l)} for p, l in zip(proba, label)]
    return JSONResponse(content=result)