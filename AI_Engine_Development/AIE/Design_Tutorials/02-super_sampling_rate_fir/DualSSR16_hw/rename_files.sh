#!/bin/bash

for f in data/*.txt; do
   for col in "$@"; do
      cp "$f" "${f%.txt}_$col.txt"
   done
done

