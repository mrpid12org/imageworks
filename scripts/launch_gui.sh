#!/usr/bin/env bash
#
# Launch ImageWorks GUI Control Center
#

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🖼️  ImageWorks GUI Control Center${NC}"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: 'uv' not found${NC}"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${GREEN}✅ Starting Streamlit app...${NC}"
echo ""

# Launch with uv (disable server.headless to prevent browser auto-open issues)
uv run streamlit run src/imageworks/gui/app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false \
    --server.headless=true \
    "$@"

echo ""
echo -e "${BLUE}📱 App running at: http://localhost:8501${NC}"
echo -e "${YELLOW}💡 Open this URL in your browser manually${NC}"
echo ""
