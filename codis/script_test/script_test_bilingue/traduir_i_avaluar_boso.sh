#!/bin/bash

# Script per traduir i avaluar el corpus amb eina en BOSO
python traduccio_2_voltes.py
python script_test_correccio_traduccions.py "../corpus_proves_traduccio/prova_traduccio_boso"