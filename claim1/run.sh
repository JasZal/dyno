#!/bin/bash

# run the log_Reg training, either skipping the biggest dataset Nhanes or including it 
# for including it please set "INCLUDE_NHANES" to true

# default skip Nhanes (false)
INCLUDE_NAHNES=false

if [[ "$1" == "--includeNahnes" ]]; then
  INCLUDE_NAHNES=true
fi


echo "start benchmarking DyNo"
go run ./artifact/benchmarking 
echo "finished benchmarking DyNo"


echo "start logistic regression training"
go run ./artifact/log_reg --includeNahnes=$INCLUDE_NAHNES
echo "finished logistic regression training"


echo "build figures"
python3 ../claim1/results/build_plots.py
echo "finished everything"


mkdir -p ./claim1Results

# copy files to volume folder
FILES=(
  "log_reg_utility.pdf"
  "log_reg_utility.txt"
  "log_reg_runtime.txt"
  "benchmarking_runtime.txt"
  "log_reg_utility_nhanes.txt"
  "log_reg_utility_nhanes.pdf"
)

for file in "${FILES[@]}"; do
    src="./claim1/results/$file"
    if [ -f "$src" ]; then
        cp "$src" ../claim1Results/
    fi
done

echo "saved results"

