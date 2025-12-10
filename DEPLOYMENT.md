# ZongBot Deployment Guide

完整的 Phase 1-5 部署指南，用於在 GCP VM 上運行 ZongBot 系統。

## 快速開始

### 本地開發

```bash
# 克隆並設置
git clone https://github.com/caizongxun/zongbot.git
cd zongbot

# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt

# 配置環境
cp .env.example .env
# 編輯 .env 填入你的 API Keys

# 運行系統
python orchestration/main.py
```

### Docker 部署

```bash
# 構建和運行
docker-compose up -d

# 查看日誌
docker-compose logs -f zongbot

# 停止
docker-compose down
```

## GCP VM 部署

### 1. 建立 VM 實例

```bash
gcloud compute instances create zongbot-vm \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --machine-type=e2-medium \
  --zone=us-central1-a
```

### 2. SSH 連接

```bash
gcloud compute ssh zongbot-vm --zone=us-central1-a
```

### 3. 執行部署腳本

```bash
# 克隆倉庫
git clone https://github.com/caizongxun/zongbot.git ~/zongbot
cd ~/zongbot

# 執行部署
bash scripts/deploy.sh

# 配置環境變量
cp .env.example .env
# 編輯 .env
nano .env

# 設置 Systemd 服務
sudo cp scripts/zongbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable zongbot
sudo systemctl start zongbot

# 檢查狀態
sudo systemctl status zongbot
```

## 配置

### 必需的 API Keys

編輯 `.env` 文件：

```bash
# Binance (必需)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# HuggingFace (必需)
HUGGINGFACE_TOKEN=your_token

# Discord (可選，Phase 4)
DISCORD_TOKEN=your_token
DISCORD_CHANNEL_ID=channel_id

# 環境設置
DEMO_MODE=false  # 設為 false 進行完整運行
ENVIRONMENT=production
```

## 系統結構

### Phase 1: 數據層
- `src/data/binance_fetcher.py` - Binance API 爬蟲
- `src/data/data_processor.py` - 數據清洗
- `src/data/hf_uploader.py` - HuggingFace 整合

### Phase 2: 特徵層
- `src/features/feature_engineering.py` - 15+ 技術指標

### Phase 3: 模型層
- `src/models/model.py` - LSTM/GRU/Attention 模型
- `src/models/train.py` - 訓練腳本
- `src/models/inference.py` - 推理引擎

### Phase 4: 信號層
- `src/bot/discord_bot.py` - Discord 機器人

### Phase 5: 自動化層
- `src/orchestration/scheduler.py` - 調度器
- `orchestration/main.py` - 主程式

## 自動化計劃

VM 自動執行以下任務：

```
- 每 4 小時：從 Binance 收集新數據 → 上傳到 HF
- 每週日 2 AM：使用最新數據重新訓練模型 → 上傳到 HF
- 每 15 分鐘：運行推理 → 生成交易信號 → 發送到 Discord
```

## 監控

### 查看日誌

```bash
# 實時日誌
sudo journalctl -u zongbot -f

# 最後 100 行
sudo journalctl -u zongbot -n 100

# 日期範圍
sudo journalctl -u zongbot --since "2023-12-10 00:00:00" --until "2023-12-11 00:00:00"
```

### Docker 日誌

```bash
# 查看容器日誌
docker logs -f zongbot

# 查看特定時間範圍
docker logs --since 10m zongbot
```

## 故障排除

### 無法連接 Binance API

1. 檢查 API Keys 正確性
2. 確認 IP 地址在 Binance API 白名單中
3. 檢查網絡連接

```bash
curl -I https://api.binance.com
```

### HuggingFace 上傳失敗

1. 驗證 token
2. 確保倉庫存在
3. 檢查磁盤空間

```bash
df -h
```

### Discord 信號未發送

1. 驗證 Discord Token 和 Channel ID
2. 檢查 Bot 有發送消息權限
3. 查看日誌中的錯誤

## 性能優化

### CPU 優化

- 調整 `config.yaml` 中的 `parallel_workers`
- 使用 GRU 而不是 LSTM（更快）

### 內存優化

- 降低批次大小
- 使用 int8 量化模型

### 存儲優化

- 定期清理舊數據
- 使用 Parquet 格式（更小）

## 安全最佳實踐

1. **保護 API Keys**
   - 不要上傳 `.env` 到 Git
   - 使用 GCP Secret Manager
   - 定期輪換密鑰

2. **防火牆規則**
   ```bash
   gcloud compute firewall-rules create zongbot-bot \
     --allow=tcp:8000 \
     --source-ranges=YOUR_IP
   ```

3. **定期備份**
   ```bash
   gsutil -m cp -r gs://zongbot-backup/* ./backup/
   ```

## 成本估算 (GCP)

| 項目 | 預估成本/月 |
|-----|----------|
| e2-medium VM | $20-30 |
| Cloud Storage | $1-5 |
| Cloud SQL (可選) | $50+ |
| **總計** | **$25-85** |

## 支援

如有問題，請：
1. 檢查日誌：`sudo journalctl -u zongbot`
2. 在 GitHub 提交 Issue
3. 查看 README.md 的常見問題

## 許可證

MIT License
