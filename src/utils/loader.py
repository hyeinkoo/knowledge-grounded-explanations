import csv
import json


def load_file(file_path):
    file_type = file_path.split('.')[-1]
    if file_type == 'csv':
        file = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                file.append(line)
    elif file_type == 'txt':
        file = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                file.append(line.lower().split())
    elif file_type == 'json':
        with open(file_path, 'r') as f:
            file = json.load(f)
    return file
