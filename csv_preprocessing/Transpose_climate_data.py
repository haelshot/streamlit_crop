import csv
from csv_preprocessing.xlsxconverter import clean_up_data

column_headers = ["Year", "Month", "rainfall_mm", "max_temp", "min_temp", "mean_temp",
                         "rel_humidity", "rel_humidity.1", "wind_speed", "sunshine_hours",
                         "evaporation_mm", "soil_temp_50cm", "soil_temp_30cm", "soil_temp_20cm",
                         "soil_temp_10cm", "soil_temp_5cm"]
    

def _transpose_list(read_txt_file) -> list:
    input_data = read_txt_file

    transposed_data = list(map(list, zip(*input_data)))

    return transposed_data

def _convert_to_dict(data):
    running = True
    start = 0
    end = 15
    year = 2008

    return_data = {}

    while running:
        month = 1
        transposed_data = _transpose_list(data[start:end])
        for line in transposed_data:
            line.insert(0, month)
            month += 1    

        return_data[year] = transposed_data
        year += 1
        if end >= 224:
            running = False
            break
        start += 15
        end += 15
    return return_data


def convert_to_csv(filepath):
    output_file_path = 'csv_preprocessing/bin/temp.csv'
    clean_up_data(filepath)
    with open('csv_preprocessing/bin/temp.txt', 'r') as file:
        data = [line.strip().split(',') for line in file]
    data_dict = _convert_to_dict(data)
    # Write data to CSV file
    with open(output_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write headers
        csv_writer.writerow(column_headers)
        
        # Write data rows
        for year, data_rows in data_dict.items():
            for data_row in data_rows:
                # Update the Year column
                data_row.insert(0, year)
                
                # Check if the row length matches the number of columns
                if len(data_row) < len(column_headers):
                    # Append empty strings to make the row length match the number of columns
                    data_row += [''] * (len(column_headers) - len(data_row))
                # Write the row to the CSV file
                csv_writer.writerow(data_row)