#!/bin/bash
# Reconcile pass: verify EXPECTED ARTIFACTS rather than trusting exit codes, and
# re-run whatever is missing. Runs strictly serially (one ssh at a time) so it
# cannot itself trip the MaxSessions cap that caused the original damage.
#
# Why this exists: a dropped ssh made dispatchers record FAILED for runs that had
# actually completed (Flowers102, OxfordIIITPet: 22/22 pairs written) and OK-looking
# state for runs that were truncated (Cars 20/22, PCAM 6/22). Exit status is not a
# reliable completion signal across a flaky transport; artifact count is.
set -u
SP=/tmp/claude-1000/-mnt-KS-2TB-PARA-Projects-quantization-qat-transfer/ab33a2a3-75dc-4425-941b-52636164991d/scratchpad
RSSH="bash $SP/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
LOCAL=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer
DS_V="Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN CIFAR10 CIFAR100 STL10 Food101 Flowers102 FER2013 PCAM OxfordIIITPet RenderedSST2 EMNIST FashionMNIST KMNIST TinyImageNet ImageNet"
ALL_V=$(echo $DS_V | tr ' ' ',')
DS_T="Emotion IMDB Banking77 AmazonReviewsClassification AmazonCounterfactual MassiveIntent MassiveScenario MTOPDomain MTOPIntent ToxicConversations TweetSentimentExtraction"

echo "### [$(date +%T)] reconcile: waiting for text Stage 1 queue to drain"
while [ -s $SP/q_text.txt ]; do sleep 60; done
echo "### [$(date +%T)] text queue drained"

############ TEXT: re-run any (model,dataset) with no 4-bit checkpoint ############
echo "### [$(date +%T)] TEXT checkpoint reconcile"
for M in "Qwen/Qwen3-Embedding-0.6B:Qwen_Qwen3_Embedding_0.6B:score" \
         "google-bert/bert-large-uncased:google_bert_bert_large_uncased:classifier" \
         "google/embeddinggemma-300m:google_embeddinggemma_300m:score" \
         "google-bert/bert-base-uncased:google_bert_bert_base_uncased:classifier"; do
  MODEL=$(echo $M | cut -d: -f1); SAN=$(echo $M | cut -d: -f2); SKIP=$(echo $M | cut -d: -f3)
  for DS in $DS_T; do
    N=$($RSSH behemoth "find $B/storage/checkpoints/text/ilharco_automodelforsequenceclassification/qat/$SAN/$DS -path '*bits=4*' -name 'backbone_epoch_*.pt' 2>/dev/null | wc -l")
    if [ "${N:-0}" -lt 1 ]; then
      echo "### [$(date +%T)] MISSING text ckpt: $SAN/$DS -> re-running"
      $RSSH behemoth "bash ~/text_qat_one.sh '$MODEL' '$DS' '$SKIP' 0 '.venv/bin/python' '$B'" >> $SP/runlogs/reconcile_text.log 2>&1
      N2=$($RSSH behemoth "find $B/storage/checkpoints/text/ilharco_automodelforsequenceclassification/qat/$SAN/$DS -path '*bits=4*' -name 'backbone_epoch_*.pt' 2>/dev/null | wc -l")
      echo "### [$(date +%T)] $SAN/$DS -> ckpts now ${N2:-0}"
    fi
  done
done
echo "### [$(date +%T)] TEXT RECONCILE DONE"

############ VISION: re-run any target with < 22 grid cells ############
echo "### [$(date +%T)] waiting for vision grid to drain"
while [ -s $SP/q_heavy.txt ] || [ -s $SP/q_light.txt ]; do sleep 120; done
sleep 300   # let in-flight items land
echo "### [$(date +%T)] VISION grid reconcile"
QVDIR="evaluations/vision/ilharco_timm_supervised/001_qat_transfer"
for T in $DS_V; do
  # NOTE: the 3090Ti's EVALUATION_BASE_PATH is /mnt/KS_960GB/... which is a DIFFERENT
  # mount from its CHECKPOINT_BASE_PATH (/mnt/WD_4TB/...). Omitting it here made the
  # completeness check under-count and would have re-run targets it had already done.
  NB=$($RSSH behemoth "find $B/$QVDIR -path '*qat=bits=4*' -path '*ptq=bits=4*' -path '*tgt=${T}_seed*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
  NL=$(find $LOCAL/$QVDIR -path '*qat=bits=4*' -path '*ptq=bits=4*' -path "*tgt=${T}_seed*" -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l)
  NT=$($RSSH rig-3090-ti "find /mnt/KS_960GB/PARA/Projects/quantization/qat-transfer/$QVDIR -path '*qat=bits=4*' -path '*ptq=bits=4*' -path '*tgt=${T}_seed*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
  TOT=$(( ${NB:-0} + ${NL:-0} + ${NT:-0} ))
  if [ "$TOT" -lt 22 ]; then
    echo "### [$(date +%T)] INCOMPLETE tgt=$T ($TOT/22) -> re-running all 22 sources"
    $RSSH behemoth "bash ~/qv_one.sh '$T' '$ALL_V' 0 '.venv/bin/python' '$B'" >> $SP/runlogs/reconcile_vision.log 2>&1
    NB2=$($RSSH behemoth "find $B/$QVDIR -path '*qat=bits=4*' -path '*ptq=bits=4*' -path '*tgt=${T}_seed*' -path '*alpha=1.0*' -path '*split=test*' -name eval_results.json 2>/dev/null | wc -l")
    echo "### [$(date +%T)] tgt=$T -> now ${NB2:-0} on behemoth"
  else
    echo "### [$(date +%T)] tgt=$T OK ($TOT/22)"
  fi
done
echo "### [$(date +%T)] RECONCILE COMPLETE"
