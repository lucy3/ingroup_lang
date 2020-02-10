#!/bin/bash

module load java

dims=(2 10 20 50 100 150)
lambs=(1000 5000 10000 15000)
for j in "${dims[@]}"
do
    for k in "${lambs[@]}"
    do
        rm fscores.temp
        rm vmeasures.temp 
        for i in {0..9}
        do
            fscore=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/fscore.jar ../logs/semeval2010/semeval2010_clusters${j}_${i}_${k} ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

            vmeasure=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/vmeasure.jar ../logs/semeval2010/semeval2010_clusters${j}_${i}_${k} ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

            IFS=':' read -ra fscore_parts <<< "$fscore"
            IFS=':' read -ra vmeasure_parts <<< "$vmeasure"
            echo ${fscore_parts[1]} >> "fscores.temp"
            echo ${vmeasure_parts[1]} >> "vmeasures.temp"
        done
        echo "Scores for dimension $j, lambda $k"
        fscore_mean=$(cat fscores.temp | awk '{sum+=$1}END{print sum/NR}')
        fscore_std=$(cat fscores.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
        echo "FScore:" $fscore_mean, $fscore_std
        vmeasure_mean=$(cat vmeasures.temp | awk '{sum+=$1}END{print sum/NR}')
        vmeasure_std=$(cat vmeasures.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
        echo "Vmeasure:" $vmeasure_mean, $vmeasure_std
    done
done

