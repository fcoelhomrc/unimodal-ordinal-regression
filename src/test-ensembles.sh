#!/bin/bash

TABULAR_DATASETS=""
IMAGE_DATASETS="FOCUSPATH SMEAR2005 FGNET"
LOSSES="CrossEntropy BinomialUnimodal_CE"
LOSSES_LAMBDA="CO2"

MODELS="RandomEnsemble AverageEnsemble MajorityVotingEnsemble MedianEnsemble WassersteinEnsemble_LP FastWassersteinEnsemble_LP"

for DATASETS_TYPE in "TABULAR" "IMAGE"; do
if [ "$DATASETS_TYPE" == "TABULAR" ]; then
DATASETS=$TABULAR_DATASETS
else
DATASETS=$IMAGE_DATASETS
fi
echo $DATASETS_TYPE
echo "\documentclass{standalone}"
echo "\usepackage{xcolor}"
echo "\begin{document}"
echo "\begin{tabular}{llllllllllllll}"
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
    # echo "losses lambda: $LOSSES_LAMBDA -> best values: $LOSSES_LAMBDA_VALUES"
    echo -n "\\bf $DATASET" # & \bf Accuracy & \bf MAE & \bf Times Unimodal \\\\"
    for MODEL in $MODELS; do echo -n " & $MODEL"; done
    echo " \\\\"

    for METRIC in $(seq 0 7); do
        if [ $METRIC -eq 0 ]; then echo -n "\%Accuracy"; fi
        if [ $METRIC -eq 1 ]; then echo -n "MAE"; fi
        if [ $METRIC -eq 2 ]; then echo -n "QWK"; fi
        if [ $METRIC -eq 3 ]; then echo -n "\%$\tau$"; fi
        if [ $METRIC -eq 4 ]; then echo -n "\%Unimodal"; fi
        if [ $METRIC -eq 5 ]; then echo -n "\%Wass"; fi
        if [ $METRIC -eq 6 ]; then echo -n "ZME"; fi
        if [ $METRIC -eq 7 ]; then echo -n "NLL"; fi
        for MODEL in $MODELS; do
            python3 src/test-ensembles.py "$DATASET" "$MODEL" "$LOSSES" "$LOSSES_LAMBDA" "$LOSSES_LAMBDA_VALUES" --reps 1 2 3 4 --only-metric $METRIC --datadir=predictions
        done
        echo " \\\\"
    done
    echo "\\hline"
done
echo "\\end{tabular}"
echo "\\end{document}"
done
