#!/usr/bin/env bash

filename="$1"
n="$2"

if [ -z "$filename" -o -z "$n" ];
then
    >&2 echo "specify file and number of sample to test."
    exit 1
fi

f="${filename%.csv}"

tmp="$(mktemp)"
tail -n +2 "$filename" | shuf > "$tmp"
nlines=$(wc -l < "$tmp")
if [ "$n" -gt "$nlines" ]
then
    >&2 echo "$n > $nlines"
    exit 1
fi
train=$n
test="$(($nlines - $train))"

if [ -z "$train" ] || [ -z "$test" ]
then
    >&2 echo "ERROR: train=$train test=$test"
    exit 1
fi

trainfile="$f"_train.csv
testfile="$f"_test.csv

head -n 1 "$filename" > "$trainfile"
head -n "$train" "$tmp" >> "$trainfile"

head -n 1 "$filename" > "$testfile"
tail -n "$test" "$tmp" >> "$testfile"

