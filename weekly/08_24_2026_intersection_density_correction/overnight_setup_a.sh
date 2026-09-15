#!/bin/bash
cd "$(dirname "$0")"
LOG=overnight_setup_a_summary.log
echo "=== Setup (a) overnight run started: $(date) ===" >> "$LOG"

echo "SIFT start: $(date)" >> "$LOG"
python3 -u schemes/sift_setup_a_test.py --n-candidates 2000 --n-test-queries 10000 \
  --run-dir results/overnight_setup_a_sift > sift_setup_a_overnight.log 2>&1
SIFT_EXIT=$?
if [ $SIFT_EXIT -eq 0 ]; then
  echo "SIFT completed: $(date)" >> "$LOG"
else
  echo "SIFT FAILED (exit $SIFT_EXIT): $(date)" >> "$LOG"
fi

echo "ESCI start: $(date)" >> "$LOG"
python3 -u schemes/esci_setup_a_test.py --n-candidates 2000 --n-test-queries 50000 \
  --run-dir results/overnight_setup_a_esci > esci_setup_a_overnight.log 2>&1
ESCI_EXIT=$?
if [ $ESCI_EXIT -eq 0 ]; then
  echo "ESCI completed: $(date)" >> "$LOG"
else
  echo "ESCI FAILED (exit $ESCI_EXIT): $(date)" >> "$LOG"
fi

echo "Analysis phase start: $(date)" >> "$LOG"
python3 -u analysis/setup_a_analysis.py > setup_a_analysis_overnight.log 2>&1
ANALYSIS_EXIT=$?
if [ $ANALYSIS_EXIT -eq 0 ]; then
  echo "Analysis completed: $(date)" >> "$LOG"
else
  echo "Analysis FAILED (exit $ANALYSIS_EXIT): $(date)" >> "$LOG"
fi

echo "=== Setup (a) overnight run finished: $(date) ===" >> "$LOG"
