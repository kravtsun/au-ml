#!/bin/bash
f=$1

if [ -z $f ];
then
    f="$(mktemp)"
fi

for k in {1..10}; do 
    echo $k $(./knn.py -f misc/spambase.csv -k $k);
done > "$f"

./task2.py "$f"
echo "$f"




