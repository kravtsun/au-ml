#!/bin/bash
e=20; 
for v in 0 1 2; 
do 
    for act in sigmoid tanh relu; 
    do 
        echo var=$v act=$act; 
        time python keras_tf.py --train ../data/NOTMNIST/train --test ../data/NOTMNIST/test --epochs=$e --activate=$act 1> report_"$v"_"$act".txt 2>/dev/null; 
    done; 
done
