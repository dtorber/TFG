#!/bin/sh

deepspeed --include localhost:1 --master_port 29999 generar_resums.py
deepspeed --include localhost:1 --master_port 29999 generar_resums_llargs.py
deepspeed --include localhost:1 --master_port 29999 generar_n_resums.py
