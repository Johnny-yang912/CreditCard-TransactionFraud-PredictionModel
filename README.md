# 信用卡交易詐欺偵測 (Credit Card Transaction Fraud Detection)

---

## 專案簡介
本專案目標為建立一個 **信用卡交易詐欺偵測模型**，完整涵蓋從資料前處理、Pipeline 建模、特徵工程、模型調參到 API 部署的全流程。  
透過逐步優化特徵與調整模型超參數，最終達成穩定的詐欺辨識能力，並以 API 形式部署，方便外部系統串接。

---

## 專案資源

- [完整 Notebooks: 信用卡交易詐欺預測.ipynb](./信用卡交易詐欺偵測.ipynb)
- 工具包:
  - [時間特徵轉換器: time_transformer_tools.py](./time_transformer_tools.py)
  - [地理特徵轉換器: locat_transformer_tools.py](./locat_transformer_tools.py)

---

## 專案流程

### 1. Baseline 模型
- 以交易金額、交易時間等基本特徵，先建立基礎模型進行初步評估。  
- 作為後續特徵工程與調參的比較基準。

### 2. 加入時間特徵工程
- 將交易時間 `trans_date_trans_time` 拆解為年/月/日/時/分/秒。  
- 衍生特徵：
  - **時段分組**：early morning / morning / afternoon / night  
  - **交易間隔**：同一張卡與前一筆交易的秒數差  
  - **異常時段標記**：若交易時間落在使用者常見時段區間外，則標記為 1  

### 3. 加入地理特徵工程
- 使用交易地點與商店地點的經緯度資訊。  
- 衍生特徵：
  - **交易者與商店的距離**  
  - **跨城市/跨地區交易標記**

### 4. 調參與閾值優化
- 使用 **RandomizedSearchCV** 對 XGBoost 模型進行超參數調整。  
- 透過 **PR-AUC 與 F2-score** 作為主要評估指標，重視 **Recall 與 Precision 的平衡**。  
- 額外進行 **決策閾值調整**，確保 Precision 在高水準下仍能保有最大化的 Recall。

### 5. 模型成果
- **最終分數**：
  - PR-AUC: **0.8477**
  - F2-score: **0.7642**

### 6. API 部署
- 使用 **FastAPI + Uvicorn** 建立推論服務。  
- 兩種輸入方式：
  - **單筆交易 JSON 輸入** → 即時回傳詐欺機率與標籤  
  - **整份 CSV 上傳** → 批次回傳預測結果並下載  

---

## 專案亮點
- **Pipeline 建模 + 特徵工程**：  
  以 Scikit-learn 的 Pipeline 與 ColumnTransformer 將資料前處理、特徵工程、建模流程自動化，確保可重複與可擴展性。
- **進階特徵工程**：  
  加入交易時間特徵與地理特徵，大幅提升模型效能。
- **模型調參 + 閾值優化**：  
  不僅調整模型參數，更針對實務需求調整判斷閾值。
- **API 部署**：  
  讓模型能以 RESTful API 形式提供服務，支援即時與批次推論。

---

## 專案結構
```
CreditCard-TransactionFraud-PredictionModel/
│── app/ # FastAPI API 服務程式碼
│── models/ # 模型與閾值檔案（不含大檔）
│── time_transformer_tools.py # 時間特徵工程工具
│── locat_transformer_tools.py # 地理特徵工程工具
│── requirements.txt # 依賴套件
│── README.md # 專案說明文件
```

---

## 使用方式

### 1. 安裝套件
```
pip install -r requirements.txt
```

### 2. 啟動 API
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
### 3. 呼叫 API
```
單筆交易：
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '[
          {
            "cc_num": 1234567890123456,
            "trans_date_trans_time": "2019-01-01 08:15:00",
            "merchant": "M12345",
            "category": "gas_transport",
            "gender": "F",
            "city": "Taipei",
            "amt": 42.5,
            "unix_time": 1546320900,
            "lat": 25.033,
            "long": 121.5654,
            "merch_lat": 25.0478,
            "merch_long": 121.5319
          }
        ]'
```
### 批次 CSV：
上傳整份 CSV，回傳預測結果並下載。

## 成果總結
本專案展示了 從資料前處理 → 特徵工程 → 模型建置與優化 → API 部署 的完整流程，並最終在不平衡資料下達成 PR-AUC 0.8477、F2-score 0.7642 的表現，展現詐欺偵測在實務應用中的可行性。

## 注意事項與補充說明
- 本專案資料來源為 Kaggle 公開資料集，屬於模擬資料，僅用於展示 **特徵工程、機器學習建模與部署流程**，請勿直接套用於真實金融場景。  
- 由於本地端算力限制，未能進行更細緻的特徵工程與全面的參數搜尋。若算力資源允許，後續可延伸方向包括：  
  - 更多進階特徵設計（如使用者行為習慣偏離度、群體統計特徵等）。  
  - 更大規模與精細的超參數搜尋（如提高 `n_iter`、使用 `GridSearchCV` 或進階工具如 Optuna）。  

## 資料來源
Kaggle-Credit Card Transactions Fraud Detection Dataset
