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
        rm average.temp 
        for i in {0..4}
        do
            fscore=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/fscore.jar ../logs/semeval2010/semeval2010_clusters${j}_${i}_${k} ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

            vmeasure=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/vmeasure.jar ../logs/semeval2010/semeval2010_clusters${j}_${i}_${k} ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

            IFS=':' read -ra fscore_parts <<< "$fscore"
            IFS=':' read -ra vmeasure_parts <<< "$vmeasure"
            avg=$(awk "BEGIN {print (${fscore_parts[1]}+${vmeasure_parts[1]})/2; exit}")
            echo ${fscore_parts[1]} >> "fscores.temp"
            echo ${vmeasure_parts[1]} >> "vmeasures.temp"
            echo $avg >> "average.temp"
        done
        echo "Scores for dimension $j, lambda $k"
        fscore_mean=$(cat fscores.temp | awk '{sum+=$1}END{print sum/NR}')
        fscore_std=$(cat fscores.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
        echo "FScore:" $fscore_mean, $fscore_std
        vmeasure_mean=$(cat vmeasures.temp | awk '{sum+=$1}END{print sum/NR}')
        vmeasure_std=$(cat vmeasures.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
        echo "Vmeasure:" $vmeasure_mean, $vmeasure_std
        average_mean=$(cat average.temp | awk '{sum+=$1}END{print sum/NR}')
        average_std=$(cat average.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
        echo "Average:" $average_mean, $average_std
 
    done
done

