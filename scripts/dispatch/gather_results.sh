#!/bin/bash
# Gather ALL evaluation JSONs from every rig back onto the 4090 (canonical repo).
#
# Why this is a required step, not housekeeping: results are written by whichever
# host ran the job, so the 4-bit grid is currently split across three machines with
# THREE DIFFERENT paths. Every 998_rebuffal analysis and visualization script reads
# from the local evaluations/ tree only -- until this runs, any analysis silently
# sees a fraction of the grid and reports it as the whole thing.
#
#   4090      /mnt/KS_2TB/PARA/Projects/quantization/qat-transfer/evaluations
#   behemoth  ~/data/PARA/Projects/quantization/qat-transfer/evaluations
#   3090Ti    /mnt/KS_960GB/PARA/Projects/quantization/qat-transfer/evaluations
#             (note: NOT /mnt/WD_4TB, which is only its CHECKPOINT_BASE_PATH)
#
# JSON only -- tiny. Runs after reconcile so it captures the repaired runs too.
set -u
SP=/tmp/claude-1000/-mnt-KS-2TB-PARA-Projects-quantization-qat-transfer/ab33a2a3-75dc-4425-941b-52636164991d/scratchpad
LOCAL=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer/evaluations
BEH='behemoth:/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer/evaluations/'
TI='rig-3090-ti:/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer/evaluations/'

echo "### [$(date +%T)] waiting for reconcile to finish"
while ! grep -q 'RECONCILE COMPLETE' $SP/runlogs/reconcile.log 2>/dev/null; do sleep 120; done

count() { find $LOCAL -path '*bits=4*' -name eval_results.json 2>/dev/null | wc -l; }
echo "### [$(date +%T)] before gather: $(count) 4-bit JSONs locally"

echo "### [$(date +%T)] pulling behemoth"
rsync -a --partial "$BEH" "$LOCAL/"
echo "### [$(date +%T)] pulling 3090ti"
rsync -a --partial "$TI" "$LOCAL/"

echo "### [$(date +%T)] after gather: $(count) 4-bit JSONs locally"
V=$(find $LOCAL/vision/ilharco_timm_supervised/001_qat_transfer -path '*qat=bits=4*' -path '*ptq=bits=4*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l)
T=$(find $LOCAL/text/ilharco_automodelforsequenceclassification/001_qat_transfer -path '*qat=bits=4*' -path '*ptq=bits=4*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l)
echo "### [$(date +%T)] vision 4-bit grid cells: $V/484"
echo "### [$(date +%T)] text   4-bit grid cells: $T/484"
[ "$V" -lt 484 ] && echo "### WARNING vision grid incomplete"
[ "$T" -lt 484 ] && echo "### WARNING text grid incomplete"
echo "### [$(date +%T)] GATHER COMPLETE"
