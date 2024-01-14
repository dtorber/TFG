#!/bin/sh
# Declare a string array with type
declare -a StringArray0=(
	"i_font_correcta"
	"i_forçat_60002"
	"i_forçat_60003"
	"i_forçat_60004"
	"i_forçat_60005"
	"i_forçat_60006"
	"i_forçat_60007"
	"i_original"
)

for dataset in "${StringArray0[@]}";
do
	python3 calculate_abstractivity_indicators.py --dataset-path ./resums_generats_primer_model/resums_test_$dataset.json --output-filename ./resultats_abstractivitat_primer_model/abstractivitat_$dataset.json
done

declare -a StringArray1=(
	"ni_forçat_60002"
	"ni_forçat_60003"
	"ni_forçat_60004"
	"ni_forçat_60005"
	"ni_forçat_60006"
	"ni_forçat_60007"
	"ni_original"
)

# Read the array values with space
for dataset in "${StringArray1[@]}";
do
	python3 calculate_abstractivity_indicators.py --dataset-path ./resums_generats_primer_model/resums_test_$dataset.json --output-filename ./resultats_abstractivitat_primer_model/abstractivitat_$dataset.json
done

for dataset in "${StringArray0[@]}";
do
	python3 calculate_abstractivity_indicators.py --dataset-path ./resums_generats/resums_llargs/resums_test_$dataset.json --output-filename ./resultats_abstractivitat_primer_model/resums_llargs/abstractivitat_$dataset.json
done

# Read the array values with space
for dataset in "${StringArray1[@]}";
do
	python3 calculate_abstractivity_indicators.py --dataset-path ./resums_generats/resums_llargs/resums_test_$dataset.json --output-filename ./resultats_abstractivitat/resums_llargs/abstractivitat_$dataset.json
done