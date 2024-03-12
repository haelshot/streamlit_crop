import pandas as pd
import csv

def convert_to_txt(filepath):
    # Define input and output filenames
    input_file = filepath
    output_file = 'csv_preprocessing/bin/temp.txt'

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Save the DataFrame as a CSV file
    df.to_csv(output_file, sep=',', index=False)

    print("Conversion complete! Your CSV file is saved as:", output_file)


def remove(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    is_jan = False

    for line in lines:
        if 'JAN' in line:
            is_jan = True
            # Skip appending the current line and the last three lines
            if len(new_lines) >= 3:
                new_lines = new_lines[:-3]
        else:
            is_jan = False
            # Append the line to the new_lines list
            new_lines.append(line)

    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.writelines(new_lines)

    print("Rows with 'JAN' and 3 rows before each occurrence have been removed.")

def clean_up_data(filepath):
    convert_to_txt(filepath)
    remove('csv_preprocessing/bin/temp.txt')
    with open('csv_preprocessing/bin/temp.txt') as f:
        data = f.readlines()
    # Split data into lines and extract header and rows
    processed_data = []
    for line in data:
        # Split the line into columns
        columns = line.strip().split(',')
        
        # Remove the first column
        columns = columns[1:]
        
        # Ensure there are 12 columns
        if len(columns) < 12:
            # If there are fewer than 12 columns, add empty columns
            columns.extend([''] * (12 - len(columns)))
        elif len(columns) > 12:
            # If there are more than 12 columns, truncate the excess
            columns = columns[:12]
        
        # Join the columns back into a line
        processed_line = ','.join(columns)
        
        # Append the processed line to the result
        processed_data.append(processed_line)

    # Write the processed data to the output file
    with open('csv_preprocessing/bin/temp.txt', 'w') as output_file:
        output_file.writelines('\n'.join(processed_data))