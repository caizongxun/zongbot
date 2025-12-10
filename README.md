# ZongBot - 加密貨幣預測交易系統

一個完整的端到端機器學習交易系統，結合 Binance API、深度學習模型、HuggingFace 整合和 Discord 機器人。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 項目概述

ZongBot 是一個五階段的加密貨幣交易預測系統：

- **Phase 1** ✓ 數據層：從 Binance 自動收集 15+ 加密貨幣的多時間框架數據
- **Phase 2** ✓ 特徵工程：計算 15+ 技術指標和波動率指標
- **Phase 3** ✓ 模型層：LSTM/GRU/Attention 神經網絡用於雙任務預測
- **Phase 4** ✓ 信號層：Discord 機器人自動推送交易信號
- **Phase 5** ✓ 部署層：GCP VM 自動化調度和編排

## 📊 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    ZongBot Complete System                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Data Collection & Processing                      │
│  ├─ Binance API (15 symbols × 3 timeframes)               │
│  ├─ Data Cleaning & Validation                             │
│  └─ HuggingFace Dataset Upload                             │
│                                                             │
│  Phase 2: Feature Engineering                              │
│  ├─ 10+ Moving Average Indicators                          │
│  ├─ 3+ Momentum Indicators (RSI, MACD, Stochastic)        │
│  ├─ 3+ Volatility Indicators (BB, ATR, Parkinson)         │
│  └─ Volume & Trend Indicators (OBV, ADX)                  │
│                                                             │
│  Phase 3: Deep Learning Models                             │
│  ├─ LSTM Predictor                                         │
│  ├─ GRU Predictor                                          │
│  └─ Attention-enhanced LSTM                                │
│  Outputs: [Direction (3-class) + Volatility (continuous)]  │
│                                                             │
│  Phase 4: Signal Broadcasting                              │
│  └─ Discord Bot with Real-time Updates                     │
│                                                             │
│  Phase 5: VM Automation                                    │
│  ├─ Data Collection Scheduler (4h interval)               │
│  ├─ Model Retraining Scheduler (Weekly)                    │
│  └─ Inference Scheduler (15min interval)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速開始

### 前置要求
- Python 3.10+
- CUDA 11.8+ (GPU 推薦但非必需)
- Binance API Keys
- HuggingFace 帳戶
- Discord Bot Token (Phase 4)

### 本地開發

```bash
# 克隆倉庫
git clone https://github.com/caizongxun/zongbot.git
cd zongbot

# 創建虛擬環境
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate          # Windows

# 安裝依賴
pip install -r requirements.txt

# 配置環境
cp .env.example .env
nano .env  # 編輯並填入你的 API Keys

# 運行系統
python orchestration/main.py
```

### Docker 快速部署

```bash
# 構建並運行
docker-compose up -d

# 查看日誌
docker-compose logs -f zongbot

# 停止
docker-compose down
```

## 📋 監控的加密貨幣 (15 種)

| 排名 | 符號 | 名稱 | 類型 |
|-----|------|------|------|
| 1 | BTCUSDT | Bitcoin | 主流幣 |
| 2 | ETHUSDT | Ethereum | 主流幣 |
| 3 | BNBUSDT | Binance Coin | 交易所幣 |
| 4 | ADAUSDT | Cardano | Layer 1 |
| 5 | DOGEUSDT | Dogecoin | 迷因幣 |
| 6 | SOLUSDT | Solana | Layer 1 |
| 7 | POLYUSDT | Polygon | Layer 2 |
| 8 | LINKUSDT | Chainlink | Oracle |
| 9 | XRPUSDT | Ripple | 支付幣 |
| 10 | LTCUSDT | Litecoin | Layer 1 |
| 11 | AVAXUSDT | Avalanche | Layer 1 |
| 12 | MATICUSDT | Polygon | Layer 2 |
| 13 | UNIUSDT | Uniswap | DEX |
| 14 | ATOMUSDT | Cosmos | Layer 1 |
| 15 | FTMUSDT | Fantom | Layer 1 |

## ⏰ 時間框架

系統監控三種時間框架的 K 線數據：

- **15分鐘 (15m)**: 短期交易信號
- **1小時 (1h)**: 中短期趨勢
- **4小時 (4h)**: 中期走勢

## 🔧 系統配置

### 自動化計劃

| 任務 | 頻率 | 說明 |
|------|------|------|
| 數據收集 | 每 4 小時 | 從 Binance 拉取最新 OHLCV 數據 |
| 模型重訓練 | 週日 2 AM UTC | 使用最新數據重新訓練模型 |
| 推理和信號 | 每 15 分鐘 | 運行推理生成交易信號 |
| HF 數據同步 | 每 4 小時 | 上傳新數據到 HuggingFace |
| HF 模型同步 | 週日 3 AM UTC | 上傳訓練後的模型到 HF |

### 信號篩選標準

```python
信號要求:
  - 模型置信度 ≥ 60%
  - 預測波動率 ≤ 2%
  - 排除 NEUTRAL 方向
  - 冷卻時間: 5 分鐘 (同一幣種)
```

## 📁 項目結構

```
zongbot/
├── config/                 # 配置文件
│   ├── config.yaml        # 全局配置
│   ├── symbols.json       # 監控幣種
│   └── indicators.json    # 技術指標配置
├── src/
│   ├── data/              # Phase 1: 數據層
│   │   ├── binance_fetcher.py
│   │   ├── data_processor.py
│   │   └── hf_uploader.py
│   ├── features/          # Phase 2: 特徵層
│   │   └── feature_engineering.py
│   ├── models/            # Phase 3: 模型層
│   │   ├── model.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── bot/               # Phase 4: 信號層
│   │   └── discord_bot.py
│   ├── orchestration/     # Phase 5: 編排層
│   │   └── scheduler.py
│   └── utils/
│       ├── logger.py
│       ├── config.py
│       └── validators.py
├── orchestration/
│   └── main.py            # 統一入口
├── scripts/               # 部署腳本
│   ├── deploy.sh
│   └── zongbot.service
├── Dockerfile             # Docker 配置
├── docker-compose.yml
├── DEPLOYMENT.md          # 部署指南
└── README.md             # 本文件
```

## 🎓 技術棧

### 數據和 API
- **Binance Connector**: 交易對 OHLCV 數據
- **CCXT**: 多交易所支持 (可選)
- **pandas/NumPy**: 數據處理

### 機器學習
- **PyTorch**: 神經網絡框架
- **scikit-learn**: 數據預處理
- **torchmetrics**: 性能評估

### 數據存儲
- **HuggingFace Datasets**: 數據版本管理
- **HuggingFace Models**: 模型版本管理
- **SQLite/PostgreSQL**: 本地數據 (可選)
- **Redis**: 緩存 (可選)

### 自動化和部署
- **APScheduler**: 任務調度
- **Docker & Docker Compose**: 容器化
- **Systemd**: Linux 服務管理
- **GCP Compute Engine**: VM 部署

### 通知和監控
- **discord.py**: Discord 機器人
- **python-json-logger**: 結構化日誌
- **Prometheus**: 指標收集 (可選)

## 📈 性能指標

系統跟踪以下指標：

```
預測性能:
  - 方向準確率 (Direction Accuracy)
  - 波動率 MAE (Mean Absolute Error)
  - 信號勝率 (Win Rate)
  - 夏普比率 (Sharpe Ratio)

系統性能:
  - API 調用延遲
  - 數據收集耗時
  - 推理耗時
  - Discord 消息發送速度
```

## 🔒 安全最佳實踐

1. **API 密鑰管理**
   - 使用 `.env` 文件 (本地)
   - GCP Secret Manager (生產環境)
   - 定期輪換密鑰

2. **網絡安全**
   - 限制 Binance API IP 白名單
   - 使用 VPC 隔離
   - 定期更新依賴

3. **數據保護**
   - HTTPS 用於所有 API 調用
   - 加密敏感數據
   - 定期備份

## 📚 文檔

- [部署指南](DEPLOYMENT.md) - GCP VM 完整部署
- [配置指南](CONFIG.md) - 詳細配置說明
- [API 文檔](API.md) - REST API 規範
- [故障排除](TROUBLESHOOTING.md) - 常見問題

## 🤝 貢獻

歡迎提交 Pull Requests！請確保：

- 代碼遵循 PEP 8 規範
- 添加適當的單元測試
- 更新相關文檔

## ⚠️ 免責聲明

本系統用於教育和研究目的。使用本系統進行實際交易時，請自擔風險。過去的表現不代表未來結果。

## 📞 支持

遇到問題？

1. 查看 [故障排除指南](TROUBLESHOOTING.md)
2. 檢查日誌：`sudo journalctl -u zongbot -f`
3. 在 GitHub 提交 Issue

## 📄 許可證

MIT License - 詳見 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

**Zong Xun Cai** - [GitHub](https://github.com/caizongxun)

## 🙏 致謝

感謝所有貢獻者和開源社區的支持！

---

⭐ 如果這個項目對你有幫助，請給一個 Star！
