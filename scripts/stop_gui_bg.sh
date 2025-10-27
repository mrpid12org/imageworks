#!/usr/bin/env bash
#
# Stop ImageWorks GUI background process
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🛑 Stopping ImageWorks GUI..."
echo ""

# Find and kill the process
PIDS=$(pgrep -f "streamlit run.*imageworks/gui/app.py" || true)

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}⚠️  GUI is not running${NC}"
    exit 0
fi

for PID in $PIDS; do
    echo "Killing process $PID..."
    kill $PID 2>/dev/null || true
done

# Wait for processes to stop
sleep 2

# Force kill if still running
PIDS=$(pgrep -f "streamlit run.*imageworks/gui/app.py" || true)
if [ -n "$PIDS" ]; then
    echo "Force killing remaining processes..."
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null || true
    done
fi

echo -e "${GREEN}✅ GUI stopped${NC}"
echo ""

# Remove PID file
rm -f logs/gui.pid
