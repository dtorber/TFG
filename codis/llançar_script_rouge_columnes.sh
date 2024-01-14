#!/bin/sh
for font in 60002 60003 60004 60005 60006 60007
do
 	screen -dmS "script_columnes_$font" python3 script_test_rouge_forçat_vs_sense_columnes.py "$font"
done
