#!/bin/sh
declare -a longitud=(
	"llargs"
    "normals"
)

declare -a subsets=(
	"ni"
	"i"
)
declare -a langs=(
    "ca"
    "es"
)

for long in "${longitud[@]}";
do
	for subset in "${subsets[@]}";
    do
        for lang in "${langs[@]}";
        do
            python3 calculate_abstractivity_indicators.py --dataset-path ./resums_generats_bilingue/resums_${long}/resums_test_${subset}_original_${lang}.json --output-filename ./resultats_bilingue/abstractivitat/resums_${long}/resultats_test_${subset}_original_${lang}.json
        done
    done
done
echo correcte > abstractivitat_acabada.txt