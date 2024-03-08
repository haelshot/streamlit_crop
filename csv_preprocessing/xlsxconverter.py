import pandas as pd

def convert_to_csv():
    # Define input and output filenames
    input_file = "/home/hael/Desktop/Pest-and-Diseases---DL/PestAndDisease/Climate-data.xlsx"
    output_file = "climate_data.csv"

    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Save the DataFrame as a CSV file
    df.to_csv(output_file, index=False)

    print("Conversion complete! Your CSV file is saved as:", output_file)



def clean_up_data(data):
    # Split data into lines and extract header and rows
    lines = data.strip().split('\n')
    header = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:]]

    # Create a list to store the transformed data
    transformed_data = []

    # Process each row and create a dictionary for each entry
    for row in rows:
        entry = {
            'year': row[0].split('(')[1].replace(')', '') if row[0].startswith('VARIABLES') else '',
            'month': row[0].split('(')[0] if row[0].startswith('VARIABLES') else row[0],
        }

        # Add the remaining values to the dictionary
        entry.update(zip(header[1:], row[1:]))
        transformed_data.append(entry)

    # Create a DataFrame from the transformed data
    df = pd.DataFrame(transformed_data)

    # Save the DataFrame to a CSV file
    df.to_csv('output.csv', index=False)


with open('/home/hael/Desktop/Pest-and-Diseases---DL/PestAndDisease/test.txt') as f:
    file = f.read()

# clean_up_data(file)
    
import csv

# Input and output file paths
input_file_path = '/home/hael/Desktop/Pest-and-Diseases---DL/PestAndDisease/test.txt'
output_file_path = 'output_data.csv'

# Open the input file and create a CSV writer for the output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # Write header to the CSV file
    csv_writer.writerow(["Year", "Month", "RAINFALL (mm)", "MAX. TEMP. (°C)", "MIN. TEMP. (°C)", "MEAN TEMP. (°C)",
                         "REL. HUMIDITY morn (%)", "REL. HUMIDITY eve (%)", "WIND SPEED (km/day)", "SUNSHINE (hours)",
                         "EVAPORATION (mm)", "SOIL TEMP. (°C) 50cm", "SOIL TEMP. (°C) 30cm", "SOIL TEMP. (°C) 20cm",
                         "SOIL TEMP. (°C) 10cm", "SOIL TEMP. (°C) 5cm"])

    # Initialize variables
    year = 2008
    month_index = 0

    # Read data from the input file
    for line in input_file:
        values = line.strip().split(',')

        # Increment month and reset to January if it exceeds December
        month_index += 1
        if month_index > 12:
            month_index = 1
            year += 1

        # Write a new row to the CSV file
        csv_writer.writerow([year, month_index] + values)

print("CSV file has been created successfully.")
