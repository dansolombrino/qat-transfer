#!/usr/bin/env bash
# run: model=bert-large,split=val,alpha=-1.5,receiver=Emotion,donors=all   experiment: 998_rebuttal/003_lambda_sensitivity/001_signed_bert
# wave: 20260802-152022   rig: behemoth   gpu: 2
# GPU auth: user-granted 2026-08-02 for wave 20260802-152022: gpu 0,2,4,5,6,7
set -uo pipefail
cd "$(dirname "$0")/../../../../../.." || exit 1

RUN_ID_FLAT=model=bert-large,split=val,alpha=-1.5,receiver=Emotion,donors=all
EVAL_DIR=evaluations/998_rebuttal/003_lambda_sensitivity/001_signed_bert/model=bert-large/split=val/alpha=-1.5/receiver=Emotion/donors=all
LOG_DIR=logs/998_rebuttal/003_lambda_sensitivity/001_signed_bert/model=bert-large,split=val,alpha=-1.5,receiver=Emotion,donors=all/wave_20260802-152022
ARTIFACT=evaluations/998_rebuttal/003_lambda_sensitivity/001_signed_bert/model=bert-large/split=val/alpha=-1.5/receiver=Emotion/donors=all/complete.json

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES=2
BEHEMOTH_AUTHORIZED_GPUS=0,2,4,5,6,7
for d in ${CUDA_VISIBLE_DEVICES//,/ }; do
  case ",$BEHEMOTH_AUTHORIZED_GPUS," in
    *",$d,"*) ;;
    *) echo "[abort] gpu $d is not authorized (authorized: $BEHEMOTH_AUTHORIZED_GPUS)" >&2; exit 1 ;;
  esac
done
export WAVE_ID=20260802-152022
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

.venv/bin/python code/experiments/998_rebuttal/003_lambda_sensitivity/001_signed_bert/run_row.py model=bert-large split=val alpha=-1.5 receiver=Emotion donors=all 2>&1   | tee "$LOG_DIR/wave_behemoth_gpu2-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi
if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'PYEOF'
import datetime, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s) + "\n")
t.replace(p)
PYEOF
fi
exit "$rc"
