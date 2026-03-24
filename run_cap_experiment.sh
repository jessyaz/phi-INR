#!/usr/bin/env bash
set -e

echo "═══════════════════════════════════════════"
echo "Paire 1 — baselines LSTM (parallel)"
echo "═══════════════════════════════════════════"

python baseline/baseline_lstm.py baseline_lstm.model.mode=static_only &
PID1=$!

python baseline/baseline_lstm.py baseline_lstm.model.mode=flow_only &
PID2=$!

wait $PID1 || { echo "ERREUR : static_only (PID $PID1)"; exit 1; }
wait $PID2 || { echo "ERREUR : flow_only   (PID $PID2)"; exit 1; }

echo "✓ Paire 1 terminée"

echo "═══════════════════════════════════════════"
echo "Paire 2 — INR (parallel)"
echo "═══════════════════════════════════════════"

python inr_forecast.py inr.use_context=True inr.control=static_only &
PID3=$!

python inr_forecast.py inr.use_context=False &
PID4=$!

wait $PID3 || { echo "ERREUR : inr static_only  (PID $PID3)"; exit 1; }
wait $PID4 || { echo "ERREUR : inr no_context   (PID $PID4)"; exit 1; }

echo "✓ Paire 2 terminée"
echo "═══════════════════════════════════════════"
echo "Toutes les expériences sont terminées."