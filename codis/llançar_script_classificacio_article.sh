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


declare -a StringArray1=(
	"ni_forçat_60002"
	"ni_forçat_60003"
	"ni_forçat_60004"
	"ni_forçat_60005"
	"ni_forçat_60006"
	"ni_forçat_60007"
	"ni_original"
)

for dataset in "${StringArray0[@]}";
do
	python3 test_model.py ./models_classificacio/ca-article ./resums_generats/resums_test_$dataset.json --text-key summary --label-key source --output ./resultats_classificacio/article/ca-article_$dataset.out
done

# Read the array values with space
for dataset in "${StringArray1[@]}";
do
	python3 test_model.py ./models_classificacio/ca-article ./resums_generats/resums_test_$dataset.json --text-key summary --label-key source --output ./resultats_classificacio/article/ca-article_$dataset.out --with-excluded
done

# Read the array values with space
for dataset in "${StringArray0[@]}";
do
	python3 test_model.py ./models_classificacio/ca-article ./resums_generats/resums_llargs/resums_test_$dataset.json --text-key summary --label-key source --output ./resultats_classificacio/article/resums_llargs/ca-article_$dataset.out
done

for dataset in "${StringArray1[@]}";
do
	python3 test_model.py ./models_classificacio/ca-article ./resums_generats/resums_llargs/resums_test_$dataset.json --text-key summary --label-key source --output ./resultats_classificacio/article/resums_llargs/ca-article_$dataset.out --with-excluded
done	