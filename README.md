# ZongBot - 加密貨幣預測交易系統

一個基於 PyTorch 的智能加密貨幣價格預測和 Discord 交易信號系統。

## 🎯 項目概述

ZongBot 是一個端到端的機器學習系統，用於：
- 🔄 自動從 Binance 爬取 15+ 加密貨幣數據
- 🧠 使用深度學習模型預測價格走勢和波動率
- 💬 通過 Discord Bot 推送交易信號
- ☁️ 在 GCP VM 上持續運行和自動更新

## 📋 系統架構

```
ZongBot 系統架構
├── Phase 1: 數據層 (當前)
│   ├── Binance API 爬蟲 → 獲取 15+ 幣種
│   ├── 時間框架: 15m, 1h, 4h
│   └── 數據上傳到 HuggingFace
├── Phase 2: 特徵工程
│   ├── 15+ 技術指標
│   └── 波動率計算
├── Phase 3: 模型訓練
│   ├── LSTM/GRU 時間序列模型
│   ├── 方向預測 + 波動率預測
│   └── 模型上傳到 HuggingFace
├── Phase 4: Discord Bot
│   └── 實時交易信號推送
└── Phase 5: GCP 部署
    └── VM 自動化運行
```

## 🚀 快速開始

### 環境設置

```bash
# 克隆倉庫
git clone https://github.com/caizongxun/zongbot.git
cd zongbot

# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # 或 Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 配置 API Keys

創建 `.env` 文件：
```bash
cp .env.example .env
```

編輯 `.env` 填入你的認證信息：
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
HUGGINGFACE_TOKEN=your_hf_token
DISCORD_TOKEN=your_discord_bot_token
```

### 運行數據爬蟲

```bash
python -m src.data.binance_fetcher
```

## 📁 項目結構

```
zongbot/
├── README.md                      # 項目說明
├── requirements.txt               # Python 依賴
├── .env.example                   # 環境變量模板
├── .gitignore                     # Git 忽略文件
├── config/
│   ├── config.yaml               # 全局配置
│   ├── symbols.json              # 監控的加密貨幣列表
│   └── indicators.json           # 技術指標配置
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance_fetcher.py   # Binance 數據爬蟲
│   │   ├── data_processor.py    # 數據清洗處理
│   │   ├── hf_uploader.py       # HuggingFace 上傳
│   │   └── storage.py           # 本地存儲管理
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py # 特徵提取 (Phase 2)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py             # 模型架構 (Phase 3)
│   │   ├── train.py             # 訓練腳本 (Phase 3)
│   │   └── inference.py         # 推理 (Phase 4)
│   ├── bot/
│   │   ├── __init__.py
│   │   └── discord_bot.py       # Discord 機器人 (Phase 4)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # 日誌工具
│       └── config.py            # 配置管理
├── notebooks/
│   ├── 01_eda.ipynb             # 探索性數據分析
│   ├── 02_feature_analysis.ipynb # 特徵分析
│   └── 03_model_evaluation.ipynb # 模型評估
├── tests/
│   ├── __init__.py
│   ├── test_fetcher.py
│   └── test_processor.py
└── scripts/
    ├── deploy.sh                 # 部署腳本
    └── monitor.sh                # 監控腳本
```

## 📊 監控的加密貨幣

當前配置監控以下 15 種加密貨幣：
- BTC (Bitcoin) - BTCUSDT
- ETH (Ethereum) - ETHUSDT
- BNB (Binance Coin) - BNBUSDT
- ADA (Cardano) - ADAUSDT
- DOGE (Dogecoin) - DOGEUSDT
- SOL (Solana) - SOLUSDT
- POLY (Polygon) - POLYUSDT
- LINK (Chainlink) - LINKUSDT
- XRP (Ripple) - XRPUSDT
- LTC (Litecoin) - LTCUSDT
- AVAX (Avalanche) - AVAXUSDT
- MATIC (Polygon) - MATICUSDT
- UNI (Uniswap) - UNIUSDT
- ATOM (Cosmos) - ATOMUSDT
- FTM (Fantom) - FTMUSDT

## ⏱️ 時間框架

系統支持三種 K 線時間框架：
- **15m** (15 分鐘) - 短期交易信號
- **1h** (1 小時) - 中短期趨勢
- **4h** (4 小時) - 中期趨勢

## 🔧 開發進度

- [x] Phase 1: 數據爬蟲基礎設施
- [ ] Phase 2: 特徵工程和指標計算
- [ ] Phase 3: 模型訓練
- [ ] Phase 4: Discord Bot 開發
- [ ] Phase 5: GCP VM 部署

## 📦 依賴庫

主要依賴：
- **binance-connector** - Binance API 交互
- **pandas** - 數據處理
- **numpy** - 數值計算
- **torch** - 深度學習框架
- **huggingface-hub** - 數據和模型存儲
- **ccxt** - 加密貨幣交易 API
- **python-dotenv** - 環境變量管理

## 📝 使用說明

### 1. 獲取 API Keys

#### Binance API
1. 登錄 [Binance](https://www.binance.com)
2. 賬戶 → API 管理
3. 創建新的 API Key
4. 複製 API Key 和 Secret Key

#### HuggingFace Token
1. 登錄 [HuggingFace](https://huggingface.co)
2. 設置 → Access Tokens
3. 創建新 token（有寫入權限）

#### Discord Bot Token
1. 進入 [Discord Developer Portal](https://discord.com/developers/applications)
2. 創建新應用
3. Bot → Add Bot
4. 複製 Token

### 2. 環境配置

```bash
cp .env.example .env
# 編輯 .env 填入所有 API Keys
```

### 3. 運行爬蟲

```bash
# 一次性爬取數據
python -m src.data.binance_fetcher --mode once

# 持續運行（定時爬取）
python -m src.data.binance_fetcher --mode continuous --interval 300
```

## 🤝 貢獻指南

歡迎提交 Pull Requests！請確保：
- 代碼遵循 PEP 8 規範
- 添加相應的測試
- 更新文檔

## 📄 許可證

MIT License

## 📞 聯繫方式

如有問題或建議，請提交 Issue 或聯繫開發者。
