#!/bin/bash

if python3 data_creation.py
then echo "opening model_preprocessing.py"
  if python3 model_preprocessing.py
  then echo "opening model_preparation.py"
    if python3 model_preparation.py
    then echo "opening model_testing.py"
      if python3 model_testing.py
      then echo "Success!!!"
      fi
    fi
  fi
fi
