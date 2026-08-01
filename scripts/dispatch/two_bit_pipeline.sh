#!/bin/bash
# 2-BIT PIPELINE, vision + text. Starts only after the 4-bit work has fully landed
# (gather complete), so it never contends with it.
#
# Reuses everything that already exists rather than recomputing:
#   vision fp_ptq / pretrained_ptq at 2 bits ....... 22 each, ALREADY DONE
#   text 2-bit QAT ckpts for gemma and Qwen ........ 11 each, ALREADY DONE
#   text matched 2-bit grid for Qwen ............... 121 pairs, ALREADY DONE
# So: train 22 vision + 22 text (the two BERTs only); 22 vision + 44 text gate
# rows; 22 vision + 33 text grid items (484 + 363 pairs).
#
# Phases are barriers on purpose: a grid must not start before its checkpoints
# exist, and the Stage-2 gate is what says whether the grid is worth reading.
# All remote calls go through rssh.sh (behemoth runs MaxSessions 10, no sudo).
set -u
SP=/tmp/claude-1000/-mnt-KS-2TB-PARA-Projects-quantization-qat-transfer/ab33a2a3-75dc-4425-941b-52636164991d/scratchpad
RSSH="bash $SP/rssh.sh"
LOG=$SP/runlogs/twobit; mkdir -p $LOG
DONE=$SP/runlogs/twobit_done.txt; : > $DONE
BROOT=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
BPY=.venv/bin/python

DSV="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR10 CIFAR100 STL10 Food101 Flowers102 FER2013 PCAM OxfordIIITPet RenderedSST2 EMNIST FashionMNIST KMNIST TinyImageNet ImageNet"
ALLV=$(echo $DSV | tr ' ' ',')
DST="Emotion IMDB Banking77 AmazonReviewsClassification AmazonCounterfactual MassiveIntent MassiveScenario MTOPDomain MTOPIntent ToxicConversations TweetSentimentExtraction"
ALLT=$(echo $DST | tr ' ' ',')

echo "### [$(date +%T)] 2-bit pipeline waiting for 4-bit GATHER COMPLETE"
while ! grep -q 'GATHER COMPLETE' $SP/runlogs/gather.log 2>/dev/null; do sleep 180; done
echo "### [$(date +%T)] 4-bit finished -- starting 2-bit"

QF=""; LK=""; LBL=""
pop() { flock "$LK" bash -c 'if [ -s "$1" ]; then head -1 "$1"; sed -i 1d "$1"; exit 0; fi; exit 1' _ "$QF"; }

feeder() {
  local NAME=$1 GPU=$2
  while item=$(pop); do
    echo "### [$(date +%T)] $LBL $NAME gpu$GPU <- ${item%% *}"
    # item is a full runner invocation; substitute this feeder's GPU for the "0"
    local cmd=$(echo "$item" | sed "s| 0 $BPY | $GPU $BPY |")
    if $RSSH behemoth "bash ~/runners/$cmd" >> "$LOG/$LBL.$NAME.log" 2>&1; then
      echo "$LBL|$NAME|OK|$item" >> $DONE
    else
      echo "$LBL|$NAME|FAILED|$item" >> $DONE
    fi
  done
  echo "### [$(date +%T)] $LBL $NAME drained"
}

run_phase() {   # run_phase <queuefile> <label>
  QF=$1; LBL=$2; LK="${QF%.txt}.lock"
  echo "### [$(date +%T)] $LBL start: $(wc -l < $QF) items"
  feeder b0 0 & feeder b2 2 & feeder b6 6 & feeder b7 7 &
  wait
  echo "### [$(date +%T)] $LBL COMPLETE ($(grep -c "^$LBL|" $DONE) recorded, $(grep -c "^$LBL|.*|FAILED|" $DONE) failed)"
}

# ------------------------------------------------- PHASE 1: 2-bit QAT training
Q=$SP/q2_train.txt; : > $Q
for DS in $DSV; do echo "v_qat.sh 2 $DS 0 $BPY $BROOT" >> $Q; done
for DS in $DST; do
  echo "t_qat.sh 2 google-bert/bert-base-uncased $DS classifier 0 $BPY $BROOT" >> $Q
  echo "t_qat.sh 2 google-bert/bert-large-uncased $DS classifier 0 $BPY $BROOT" >> $Q
done
run_phase $Q PHASE1_TRAIN

# --------------------------------------------------- PHASE 2: 2-bit Stage-2 gate
Q=$SP/q2_gate.txt; : > $Q
for DS in $DSV; do echo "v_s2.sh 2 $DS 0 $BPY $BROOT" >> $Q; done
for M in "google-bert/bert-base-uncased:classifier" "google-bert/bert-large-uncased:classifier" "google/embeddinggemma-300m:score" "Qwen/Qwen3-Embedding-0.6B:score"; do
  MODEL=${M%%:*}; SKIP=${M##*:}
  for DS in $DST; do echo "t_s2.sh 2 $MODEL $DS $SKIP 0 $BPY $BROOT" >> $Q; done
done
run_phase $Q PHASE2_GATE

# ------------------------------------------------------- PHASE 3: 2-bit grids
Q=$SP/q2_grid.txt; : > $Q
for T in $DSV; do echo "v_qv.sh 2 $T $ALLV 0 $BPY $BROOT" >> $Q; done
# Qwen's matched 2-bit grid (121 pairs) already exists -- deliberately excluded
for M in "google/embeddinggemma-300m:score" "google-bert/bert-base-uncased:classifier" "google-bert/bert-large-uncased:classifier"; do
  MODEL=${M%%:*}; SKIP=${M##*:}
  for T in $DST; do echo "t_qv.sh 2 $MODEL $T $ALLT $SKIP 0 $BPY $BROOT" >> $Q; done
done
run_phase $Q PHASE3_GRID

echo "### [$(date +%T)] 2-BIT PIPELINE COMPLETE"
