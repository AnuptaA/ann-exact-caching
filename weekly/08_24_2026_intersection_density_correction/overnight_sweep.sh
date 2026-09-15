#!/bin/bash
cd "$(dirname "$0")"
LOG=overnight_summary.log
echo "=== Overnight sweep started: $(date) ===" >> "$LOG"

echo "SIFT start: $(date)" >> "$LOG"
python3 -u schemes/sift_synth_sweep_run_experiment.py --run-dir results/overnight_sift > sift_overnight.log 2>&1
SIFT_EXIT=$?
if [ $SIFT_EXIT -eq 0 ]; then
  echo "SIFT completed: $(date)" >> "$LOG"
else
  echo "SIFT FAILED (exit $SIFT_EXIT): $(date)" >> "$LOG"
fi

echo "ESCI start: $(date)" >> "$LOG"
python3 -u schemes/esci_synth_sweep_run_experiment.py --run-dir results/overnight_esci > esci_overnight.log 2>&1
ESCI_EXIT=$?
if [ $ESCI_EXIT -eq 0 ]; then
  echo "ESCI completed: $(date)" >> "$LOG"
else
  echo "ESCI FAILED (exit $ESCI_EXIT): $(date)" >> "$LOG"
fi

echo "Uniform baseline start: $(date)" >> "$LOG"
python3 -u common/part2_uniform_baseline_run_experiment.py --run-dir results/overnight_uniform_baseline > uniform_baseline_overnight.log 2>&1
BASELINE_EXIT=$?
if [ $BASELINE_EXIT -eq 0 ]; then
  echo "Uniform baseline completed: $(date)" >> "$LOG"
else
  echo "Uniform baseline FAILED (exit $BASELINE_EXIT): $(date)" >> "$LOG"
fi

echo "Analysis phase start: $(date)" >> "$LOG"
python3 -u analysis/overnight_analysis.py > analysis_overnight.log 2>&1
ANALYSIS_EXIT=$?
if [ $ANALYSIS_EXIT -eq 0 ]; then
  echo "Analysis completed: $(date)" >> "$LOG"
else
  echo "Analysis FAILED (exit $ANALYSIS_EXIT): $(date)" >> "$LOG"
fi

echo "=== Overnight sweep finished: $(date) ===" >> "$LOG"
