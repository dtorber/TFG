#!/bin/bash

# Script per traduir i avaluar el corpus amb eina en BOSO
deepspeed traduccio_2_voltes_softcatala.py --num_gpus 1
python quantificar_diferents_longituds_softcatala.py 