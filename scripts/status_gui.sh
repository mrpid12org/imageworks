#!/usr/bin/env bash
#
# Check status of ImageWorks GUI
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🖼️  ImageWorks GUI Status${NC}"
echo "=========================="
echo ""

# Check if running
PIDS=$(pgrep -f "streamlit run.*imageworks/gui/app.py" || true)

if [ -z "$PIDS" ]; then
    echo -e "${RED}❌ Status: Not running${NC}"
    echo ""
    echo "To start: ./scripts/start_gui_bg.sh"
    exit 0
fi

echo -e "${GREEN}✅ Status: Running${NC}"
echo ""

# Show process details
for PID in $PIDS; do
    echo "Process ID: $PID"

    # Show CPU/Memory
    ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null || true
done

echo ""
echo -e "${BLUE}📱 URL: http://localhost:8501${NC}"
echo ""

# Check if port is listening
if netstat -tuln 2>/dev/null | grep -q ":8501 "; then
    echo -e "${GREEN}✅ Port 8501: Listening${NC}"
else
    echo -e "${YELLOW}⚠️  Port 8501: Not listening (still starting?)${NC}"
fi

echo ""
echo "Commands:"
echo "  View logs:  tail -f logs/gui.log"
echo "  Stop GUI:   ./scripts/stop_gui_bg.sh"
echo "  Restart:    ./scripts/stop_gui_bg.sh && ./scripts/start_gui_bg.sh"
