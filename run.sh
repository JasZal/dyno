#!/bin/bash

# run the log_Reg training, either including the biggest dataset Nhanes or skipping it 
# for skipping it please set "INCLUDE_NHANES" to false

# default include Nhanes
INCLUDE_NAHNES=true


echo "start benchmarking DyNo"
time go run ./benchmarking 
echo "finished benchmarking DyNo"


echo "start logistic regression training"

time go run ./log_reg --includeNahnes=$INCLUDE_NAHNES

echo "finished logistic regression training"


echo "build figures"
# todo add python script
echo "finished everything"


