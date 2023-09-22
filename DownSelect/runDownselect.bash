#!/bin/bash

# Define an array of directory names
dirs=("Basic" "TOS10" "TDOS" "SDOS" "BTOS" "BT" "BS" "TOST" "TOSS" "TS")

# Create the directories
for dir in "${dirs[@]}"; do
    mkdir "$dir"
done

# Copy .py and .csv into each directory
for dir in "${dirs[@]}"; do
    cp SVR_PermuImp.py "$dir/"
    cp OligomerFeatures_PreValidation_TSO10.csv "$dir/"
done

# Define a 2D array of arguments
args=("0 21 0 0" "21 51 0 0" "51 84 0 0" "84 114 0 0" "0 21 21 51" "0 21 51 84" "0 21 84 114" "21 51 51 84" "21 51 84 114" "51 84 84 114")

# Loop through the directories and run .py with arguments
for i in "${!dirs[@]}"; do
    cd "${dirs[i]}"
    echo "Running in ${dirs[i]} with arguments: ${args[i]}"
    python SVR_PermuImp.py ${args[i]} > out.out
    cd ..
done
