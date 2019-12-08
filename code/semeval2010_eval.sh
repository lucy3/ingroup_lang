#!/bin/bash

module load java

# for j in something do 
rm fscores.temp
rm vmeasures.temp 
for i in {0..9}
do
    fscore=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/fscore.jar ../logs/semeval2010/semeval2010_clusters100_$i ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

    vmeasure=$(java -jar ../semeval-2010-task-14/evaluation/unsup_eval/vmeasure.jar ../logs/semeval2010/semeval2010_clusters100_$i ../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key all | tail -1)

    IFS=':' read -ra fscore_parts <<< "$fscore"
    IFS=':' read -ra vmeasure_parts <<< "$vmeasure"
    echo ${fscore_parts[1]} >> "fscores.temp"
    echo ${vmeasure_parts[1]} >> "vmeasures.temp"
done
echo "Scores for dimension 100"
fscore_mean=$(cat fscores.temp | awk '{sum+=$1}END{print sum/NR}')
fscore_std=$(cat fscores.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
echo "FScore:" $fscore_mean, $fscore_std
vmeasure_mean=$(cat vmeasures.temp | awk '{sum+=$1}END{print sum/NR}')
vmeasure_std=$(cat vmeasures.temp | awk '{sum+=$1; sumsq+=$1*$1}END{print sqrt(sumsq/NR - (sum/NR)**2)}')
echo "Vmeasure:" $vmeasure_mean, $vmeasure_std
# done

