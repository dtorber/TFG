#!/bin/sh

deepspeed --num_gpus 1 generar_resums_bilingue.py
screen -dmS "test_rouge" deepspeed --include localhost:0 --master_port 29999 script_test_sense_generar.py
screen -dmS "abstractivitat_bilingue" bash avaluar_abstractivitat_bilingue.sh
echo correcte > screens_llançats.txt