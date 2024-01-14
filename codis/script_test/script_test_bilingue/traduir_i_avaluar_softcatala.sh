#!/bin/bash

# Script per traduir i avaluar el corpus amb eina en BOSO
deepspeed traduccio_2_voltes.py --num_gpus 1
python script_test_correccio_traduccions.py "../corpus_proves_traduccio/prova_traduccio_softcatala"