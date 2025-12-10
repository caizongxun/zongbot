#!/bin/bash

# ZongBot Deployment Script for GCP VM
# This script sets up the complete ZongBot system on a GCP VM

set -e  # Exit on error

echo "======================================="
echo "ZongBot Deployment Script"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/$(whoami)/zongbot"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="/var/log/zongbot"
SYSTEMD_DIR="/etc/systemd/system"

echo -e "${YELLOW}[1/6]${NC} Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3.10 python3-pip python3-venv git curl wget

echo -e "${YELLOW}[2/6]${NC} Cloning ZongBot repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull origin main
else
    git clone https://github.com/caizongxun/zongbot.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo -e "${YELLOW}[3/6]${NC} Setting up Python virtual environment..."
python3.10 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}[4/6]${NC} Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}[5/6]${NC} Creating log directory..."
sudo mkdir -p "$LOG_DIR"
sudo chown "$(whoami):$(whoami)" "$LOG_DIR"

echo -e "${YELLOW}[6/6]${NC} Setting up environment configuration..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo ".env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials:"
    echo "  cp $PROJECT_DIR/.env.example $PROJECT_DIR/.env"
    echo "  nano $PROJECT_DIR/.env"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Configure .env file with your API keys"
echo "  2. Run: source $VENV_DIR/bin/activate"
echo "  3. Test: python main.py"
echo "  4. Setup systemd service: bash scripts/setup_systemd.sh"
echo ""
