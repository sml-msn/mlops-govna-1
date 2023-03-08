#!/bin/bash
wc -l data_urls.txt
lines="$(grep -c $ data_urls.txt)"

for ((i = 0; i < lines; i++))
  do
  echo "$i" > loopCounter
  python3 data_creation.py && \
  python3 model_preprocessing.py && \
  python3 model_preparation.py && \ 
  python3 model_testing.py
  done

