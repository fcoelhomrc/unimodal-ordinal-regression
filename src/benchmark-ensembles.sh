#!/bin/bash

IMAGE_DATASETS="FOCUSPATH SMEAR2005 FGNET"
LOSSES="CrossEntropy BinomialUnimodal_CE"
LOSSES_LAMBDA="CO2"

MODELS="WassersteinEnsemble_LP FastWassersteinEnsemble_LP"


DATASETS=$IMAGE_DATASETS

for DATASET in $DATASETS; do

    LOSSES_LAMBDA_VALUES=""
    for LOSS in $LOSSES_LAMBDA; do
      LAMBDA=$(python3 src/test-best-lambda.py $DATASET $LOSS --datadir=datasets --modeldir=saved-models)
      if [ -z "$LOSSES_LAMBDA_VALUES" ]; then
        LOSSES_LAMBDA_VALUES="$LAMBDA"
      else
        LOSSES_LAMBDA_VALUES="$LOSSES_LAMBDA_VALUES $LAMBDA"
      fi
    done

    METRIC=0
    for MODEL in $MODELS; do
        time python3 src/test-ensembles.py "$DATASET" "$MODEL" "$LOSSES" "$LOSSES_LAMBDA" "$LOSSES_LAMBDA_VALUES" --reps 1 2 3 4 --only-metric $METRIC --datadir=predictions
    done
done
