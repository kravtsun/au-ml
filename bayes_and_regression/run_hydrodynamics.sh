#!/usr/bin/env bash
{ echo alpha,poly,rscore; for a in $(LANG=en_US seq --format="%.2f" 0.00 1.00 100.0); do for p in `seq 1 2`; do echo "$a","$p",$(python regression.py --train ../data/hydrodynamics_train.csv --test ../data/hydrodynamics_test.csv --poly $p --alpha $a); done; done } > hydrodynamics_results.txt
