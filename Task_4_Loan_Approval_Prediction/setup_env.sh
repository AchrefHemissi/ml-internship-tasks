#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh  —  Environment setup for Linux / macOS / WSL
# Requires: uv  (https://docs.astral.sh/uv/getting-started/installation/)
# Run once from the project root before opening any notebook.
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit immediately on any error

echo ""
echo "============================================================"
echo "  Loan Approval Prediction  —  Environment Setup  (uv)"
echo "============================================================"
echo ""

# 0. Check uv is available
if ! command -v uv &> /dev/null; then
  echo "ERROR: 'uv' is not installed."
  echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
echo "uv version: $(uv --version)"
echo ""

# 1. Create virtual environment
echo "[1/4] Creating virtual environment (.venv) ..."
uv venv .venv
echo "      Done."

# 2. Activate it
echo "[2/4] Activating virtual environment ..."
source .venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing dependencies from requirements.txt ..."
uv pip install -r requirements.txt

# 4. Register Jupyter kernel
echo "[4/4] Registering Jupyter kernel (loan-approval) ..."
python -m ipykernel install --user --name loan-approval --display-name "loan-approval"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Download the dataset from Kaggle:"
echo "         kaggle datasets download -d architsharma01/loan-approval-prediction-dataset"
echo "                                   -p data/ --unzip"
echo "       OR manually place  loan_approval_dataset.csv  inside  data/"
echo ""
echo "    2. Run:  source .venv/bin/activate"
echo "             jupyter notebook"
echo ""
echo "    3. Open notebooks in order: 01 -> 02 -> 03 -> 04 -> 05 -> 06"
echo "       (select the 'loan-approval' kernel)"
echo "============================================================"
echo ""
