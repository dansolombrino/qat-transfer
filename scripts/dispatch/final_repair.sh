#!/bin/bash
# FINAL REPAIR + GATHER. Covers everything reconcile.sh v2 did NOT:
#   * 4-bit TEXT grid cells (Stage 3 truncations)
#   * 2-bit vision + text checkpoints, gate rows, grid cells
#   * a second gather (the first ran while text Stage 3 was at 11/44 and warned
#     "text grid incomplete 125/484")
#
# Two design fixes over reconcile.sh v1:
#   1. IN-FLIGHT TRACKING. v1 re-checked "does the artifact exist?" every 180s with
#      no record of what it had already launched, so any item longer than the sweep
#      interval got launched repeatedly -- that produced 35 duplicate processes on
#      one GPU. Here each repair is launched at most once per pass.
#   2. Verification is by ARTIFACT COUNT, never exit status. Of 8 "FAILED" items
#      observed, 4 had completed successfully and 3 were partial -- the label
#      carries no information.
set -u
SP=/tmp/claude-1000/-mnt-KS-2TB-PARA-Projects-quantization-qat-transfer/ab33a2a3-75dc-4425-941b-52636164991d/scratchpad
RSSH="bash $SP/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
LOG=$SP/runlogs/final_repair
mkdir -p $LOG
LAUNCHED=$SP/runlogs/final_repair_launched.txt; touch $LAUNCHED

DST="Emotion IMDB Banking77 AmazonReviewsClassification AmazonCounterfactual MassiveIntent MassiveScenario MTOPDomain MTOPIntent ToxicConversations TweetSentimentExtraction"
ALLT=$(echo $DST | tr ' ' ',')
DSV="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR10 CIFAR100 STL10 Food101 Flowers102 FER2013 PCAM OxfordIIITPet RenderedSST2 EMNIST FashionMNIST KMNIST TinyImageNet ImageNet"
ALLV=$(echo $DSV | tr ' ' ',')
MODELS="Qwen/Qwen3-Embedding-0.6B:Qwen_Qwen3_Embedding_0.6B:score google-bert/bert-large-uncased:google_bert_bert_large_uncased:classifier google/embeddinggemma-300m:google_embeddinggemma_300m:score google-bert/bert-base-uncased:google_bert_bert_base_uncased:classifier"

echo "### [$(date +%T)] waiting for text Stage 3 to drain"
while [ -s $SP/q_text3.txt ]; do sleep 120; done
echo "### [$(date +%T)] waiting for 2-bit pipeline to complete"
while ! grep -q '2-BIT PIPELINE COMPLETE' $SP/runlogs/twobit.log 2>/dev/null; do sleep 300; done
sleep 300   # let in-flight items land

# --------------------------------------------- 4-bit TEXT grid cells (expect 11)
echo "### [$(date +%T)] PASS 1: 4-bit text grid"
for M in $MODELS; do
  MODEL=$(echo $M|cut -d: -f1); SAN=$(echo $M|cut -d: -f2); SKIP=$(echo $M|cut -d: -f3)
  for T in $DST; do
    KEY="4b|$SAN|$T"; grep -qxF "$KEY" $LAUNCHED && continue
    N=$($RSSH behemoth "find $B/evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer/$SAN -path '*tgt=${T}_seed*' -path '*qat=bits=4*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
    if [ "${N:-0}" -lt 11 ]; then
      echo "### [$(date +%T)] 4b text INCOMPLETE $SAN/$T (${N:-0}/11) -> re-running"
      echo "$KEY" >> $LAUNCHED
      $RSSH behemoth "bash ~/runners/t_qv.sh 4 '$MODEL' '$T' '$ALLT' '$SKIP' 0 '.venv/bin/python' '$B'" >> $LOG/text4b.log 2>&1
    fi
  done
done

# ------------------------------------------------------ 2-bit vision checkpoints
echo "### [$(date +%T)] PASS 2: 2-bit vision checkpoints"
for D in $DSV; do
  KEY="2bvq|$D"; grep -qxF "$KEY" $LAUNCHED && continue
  N=$($RSSH behemoth "find $B/storage/checkpoints/vision/ilharco_timm_supervised/qat/vit_base_patch16_224_orig_in21k/$D -path '*bits=2*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
  if [ "${N:-0}" -lt 1 ]; then
    echo "### [$(date +%T)] 2b vision MISSING ckpt $D -> re-running"; echo "$KEY" >> $LAUNCHED
    $RSSH behemoth "bash ~/runners/v_qat.sh 2 '$D' 0 '.venv/bin/python' '$B'" >> $LOG/vis2b.log 2>&1
  fi
done

# -------------------------------------------------------- 2-bit text checkpoints
echo "### [$(date +%T)] PASS 3: 2-bit text checkpoints (BERTs only; gemma/Qwen pre-exist)"
for M in "google-bert/bert-base-uncased:google_bert_bert_base_uncased:classifier" "google-bert/bert-large-uncased:google_bert_bert_large_uncased:classifier"; do
  MODEL=$(echo $M|cut -d: -f1); SAN=$(echo $M|cut -d: -f2); SKIP=$(echo $M|cut -d: -f3)
  for D in $DST; do
    KEY="2btq|$SAN|$D"; grep -qxF "$KEY" $LAUNCHED && continue
    N=$($RSSH behemoth "find $B/storage/checkpoints/text/ilharco_automodelforsequenceclassification/qat/$SAN/$D -path '*bits=2*' -name 'backbone_epoch_*.pt' 2>/dev/null | wc -l")
    if [ "${N:-0}" -lt 1 ]; then
      echo "### [$(date +%T)] 2b text MISSING ckpt $SAN/$D -> re-running"; echo "$KEY" >> $LAUNCHED
      $RSSH behemoth "bash ~/runners/t_qat.sh 2 '$MODEL' '$D' '$SKIP' 0 '.venv/bin/python' '$B'" >> $LOG/text2b.log 2>&1
    fi
  done
done

# ------------------------------------------------------------- 2-bit grid cells
echo "### [$(date +%T)] PASS 4: 2-bit grid cells"
for T in $DSV; do
  KEY="2bvg|$T"; grep -qxF "$KEY" $LAUNCHED && continue
  N=$($RSSH behemoth "find $B/evaluations/vision/ilharco_timm_supervised/001_qat_transfer -path '*tgt=${T}_seed*' -path '*qat=bits=2*' -path '*ptq=bits=2*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
  if [ "${N:-0}" -lt 22 ]; then
    echo "### [$(date +%T)] 2b vision grid INCOMPLETE $T (${N:-0}/22) -> re-running"; echo "$KEY" >> $LAUNCHED
    $RSSH behemoth "bash ~/runners/v_qv.sh 2 '$T' '$ALLV' 0 '.venv/bin/python' '$B'" >> $LOG/vis2bgrid.log 2>&1
  fi
done
# Qwen's matched 2-bit grid pre-exists, so it is excluded here as in the pipeline
for M in "google/embeddinggemma-300m:google_embeddinggemma_300m:score" "google-bert/bert-base-uncased:google_bert_bert_base_uncased:classifier" "google-bert/bert-large-uncased:google_bert_bert_large_uncased:classifier"; do
  MODEL=$(echo $M|cut -d: -f1); SAN=$(echo $M|cut -d: -f2); SKIP=$(echo $M|cut -d: -f3)
  for T in $DST; do
    KEY="2btg|$SAN|$T"; grep -qxF "$KEY" $LAUNCHED && continue
    N=$($RSSH behemoth "find $B/evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer/$SAN -path '*tgt=${T}_seed*' -path '*qat=bits=2*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
    if [ "${N:-0}" -lt 11 ]; then
      echo "### [$(date +%T)] 2b text grid INCOMPLETE $SAN/$T (${N:-0}/11) -> re-running"; echo "$KEY" >> $LAUNCHED
      $RSSH behemoth "bash ~/runners/t_qv.sh 2 '$MODEL' '$T' '$ALLT' '$SKIP' 0 '.venv/bin/python' '$B'" >> $LOG/text2bgrid.log 2>&1
    fi
  done
done

# ------------------------------------------------------------------ final gather
echo "### [$(date +%T)] PASS 5: final gather"
L=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer/evaluations
rsync -a --partial "behemoth:$B/evaluations/" "$L/"
rsync -a --partial "rig-3090-ti:/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer/evaluations/" "$L/"
for BITS in 4 2; do
  V=$(find $L/vision/ilharco_timm_supervised/001_qat_transfer -path "*qat=bits=${BITS}*" -path "*ptq=bits=${BITS}*" -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l)
  T=$(find $L/text/ilharco_automodelforsequenceclassification/001_qat_transfer -path "*qat=bits=${BITS}*" -path "*ptq=bits=${BITS}*" -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l)
  echo "### [$(date +%T)] ${BITS}-bit grids -> vision $V/484   text $T/484"
done
echo "### [$(date +%T)] FINAL REPAIR COMPLETE"
