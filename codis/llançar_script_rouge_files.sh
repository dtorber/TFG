#!/bin/sh
for font in $(seq 0 1 5)
do
	echo "script_test_rouge{$font}"
	screen -dmS "script_files_$font" python3 script_test_rouge_forçat_vs_sense_files.py "$font"
done
