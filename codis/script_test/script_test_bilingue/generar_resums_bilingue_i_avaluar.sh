#!/bin/bash
deepspeed --include localhost:1 generar_resums_bilingue.py
deepspeed --include localhost:1 script_test_sense_generar.py