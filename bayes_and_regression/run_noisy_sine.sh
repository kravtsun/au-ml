#!/usr/bin/env bash
{ echo alpha,poly,rscore; for a in $(LANG=en_US seq --format="%f" 0.00 1.00 100.0); do for p in `seq 2 10`; do echo "$a","$p",$(python regression.py --train ../data/noisysine_train.csv --test ../data/noisysine_test.csv --poly $p --alpha $a); done; done } > noisy_sine_results.txt 

