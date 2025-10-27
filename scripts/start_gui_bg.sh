#!/usr/bin/env bash
#
# Start ImageWorks GUI in background using nohup
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}🖼️  ImageWorks GUI Control Center - Background Mode${NC}"
echo "=========================================="
echo ""

# Check if already running
if pgrep -f "streamlit run.*imageworks/gui/app.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  GUI is already running${NC}"
    echo ""
    echo "To stop it, run: ./scripts/stop_gui_bg.sh"
    echo "To view logs, run: tail -f logs/gui.log"
    exit 1
fi

echo -e "${GREEN}✅ Starting GUI in background...${NC}"

# Start with nohup
nohup uv run streamlit run src/imageworks/gui/app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false \
    --server.headless=true \
    > "$LOG_DIR/gui.log" 2>&1 &

PID=$!

# Wait a moment for startup
sleep 3

# Check if still running
if kill -0 $PID 2>/dev/null; then
    echo -e "${GREEN}✅ GUI started successfully (PID: $PID)${NC}"
    echo ""
    echo -e "${BLUE}📱 App running at: http://localhost:8501${NC}"
    echo ""
    echo "Commands:"
    echo "  View logs:  tail -f logs/gui.log"
    echo "  Stop GUI:   ./scripts/stop_gui_bg.sh"
    echo "  Check PID:  pgrep -f 'streamlit.*imageworks'"
    echo ""

    # Save PID
    echo $PID > "$LOG_DIR/gui.pid"
else
    echo -e "${RED}❌ Failed to start GUI${NC}"
    echo "Check logs: cat logs/gui.log"
    exit 1
fi
