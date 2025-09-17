from pydantic import BaseModel, Field
from typing import Optional

class TxItem(BaseModel):
    # 請確保這些欄位名稱與你訓練時 DataFrame 的原始欄位一致
    cc_num: int
    trans_date_trans_time: str        # "YYYY-MM-DD HH:MM:SS"
    category: str
    gender: str
    city: str
    amt: float
    unix_time: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    merchant: str

class PredictResponse(BaseModel):
    proba: float = Field(..., description="Fraud probability")
    label: int = Field(..., description="1=fraud, 0=normal")
