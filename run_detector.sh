#!/bin/bash
clear
echo ""
echo " ============================================================"
echo "  FACE MASK DETECTOR — Auto Setup & Launch"
echo " ============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " [ERROR] Python 3 not found. Install from python.org"
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo " [1/3] Creating virtual environment ..."
    python3 -m venv venv
    echo "       Done."
fi

# Activate
source venv/bin/activate

# Dependencies
echo " [2/3] Checking dependencies ..."
pip install -q -r requirements.txt
echo "       Done."

# Run
echo " [3/3] Launching detector ..."
echo ""
echo " Controls: Q=Quit  S=Screenshot  SPACE=Pause  +/-=Sensitivity"
echo " ============================================================"
echo ""
python realtime_mask_detector.py "$@"

echo ""
echo " Session ended."
