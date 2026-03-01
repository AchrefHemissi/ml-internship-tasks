#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh  —  Environment setup for Linux / macOS
# Run once from the project root before opening any notebook.
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit immediately on any error

echo ""
echo "============================================================"
echo "  Traffic Sign Recognition  —  Environment Setup (pip)"
echo "============================================================"
echo ""

# 1. Create virtual environment
echo "[1/5] Creating virtual environment (.venv) ..."
python3 -m venv .venv
echo "      Done."

# 2. Activate it
echo "[2/5] Activating virtual environment ..."
source .venv/bin/activate

# 3. Upgrade pip
echo "[3/5] Upgrading pip ..."
pip install --upgrade pip --quiet

# 4. Install dependencies
echo "[4/5] Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

# 5. Register Jupyter kernel
echo "[5/5] Registering Jupyter kernel (traffic-signs) ..."
python -m ipykernel install --user --name traffic-signs --display-name "traffic-signs"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Place dataset files in  data/"
echo "           Train.csv, Meta.csv, Test.csv + image folders"
echo "    2. Run:  source .venv/bin/activate"
echo "             jupyter notebook"
echo "    3. Open notebooks/ in order: 01 -> 02 -> 03 -> 04 -> 05 -> 06"
echo "============================================================"
echo ""
