#!/usr/bin/env python3

import csv

# Define your input file path
input_file = 'RNH_HF_Rates_2025.csv'

# Open the input file and all three output text files at the same time
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open('RNH_500.txt', mode='w', newline='', encoding='utf-8') as out1, \
     open('RNH_700.txt', mode='w', newline='', encoding='utf-8') as out2, \
     open('RNH_900.txt', mode='w', newline='', encoding='utf-8') as out3:

    # Create the reader for your comma-separated input file
    reader = csv.reader(infile)
    
    # Create space-delimited writers (delimiter=' ') for the text files
    writer1 = csv.writer(out1, delimiter=' ')
    writer2 = csv.writer(out2, delimiter=' ')
    writer3 = csv.writer(out3, delimiter=' ')

    # Loop through each row and distribute the columns
    for row in reader:
        # Check to make sure the row actually has data to avoid index errors
        if not row:
            continue
            
        # File 1: Extracts columns 1 and 2 (Index 0 and 1)
        writer1.writerow([row[1], row[2], row[4], row[6]])
        
        # File 2: Extracts columns 3 and 4 (Index 2 and 3)
        writer2.writerow([row[1], row[8], row[10], row[12]])
        
        # File 3: Extracts columns 5 and 6 (Index 4 and 5)
        writer3.writerow([row[1], row[14], row[16], row[18]])

