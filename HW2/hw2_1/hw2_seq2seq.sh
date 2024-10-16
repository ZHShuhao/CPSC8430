#!/bin/bash
var1="$1"
var2="$2"

if [ -z "$var1" ]; then
  var1="/scratch/shuhaoz/DLHW/HW2/MLDS_hw2_1_data/testing_data"
fi
if [ -z "$var2" ]; then
  var2="/scratch/shuhaoz/DLHW/HW2/output.txt"
fi

echo "Using data directory: $var1"
echo "Using output file: $var2"

# Call the Python script with the user inputs as arguments
python test.py --data_dir "$var1" --output "$var2"

python /scratch/shuhaoz/DLHW/HW2/MLDS_hw2_1_data/bleu_eval.py "$var2"