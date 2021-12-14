'''
Functions for data cleaning and data preprocessing and save file as csv
'''
import os
import sys
import csv
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'src'))
# Our code import
from utils.text_preprocessing_utils import normalize, noise_removal, text_preprocessing
from utils.parameters import process_parameters_yaml


def read_data(file_path, header=False):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        if header:
            next(reader, None)  # Skip header
        for line in reader:
            class_idx, title, description = line
            text = description
            data.append([text, int(class_idx) - 1])
    return data


# Data cleaning code
def data_cleaning(dataset, normalise=True):
    processed = []
    for text, label in tqdm(dataset):
        text = text.replace("\n", " ")
        text = noise_removal(text)
        word_list = text.split()
        if normalise:
            try:
                word_list = normalize(word_list)
            except:
                pass
        text = ' '.join(word_list)
        processed.append([text, label])
    return processed


# Save data to CSV
def data_to_csv(dataset, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Date Preprocessing of AGnews dataset')
    # parser.add_argument('--export_dir', help='export directory')
    parser.add_argument('--dataset', default='agnews', help='Dataset for model')
    parser.add_argument('--datacleaning', default='True', help='Set datacleaning option True or Flase')

    args = parser.parse_args()
    print("Argument values:", args)

    # Set file Path
    params = process_parameters_yaml()

    if args.dataset == 'agnews':
        dataset_path = params['agnews_dataset_path']
        train_file = params['agnews_train_file']
        test_file = params['agnews_test_file']
        pp_train_file = params['agnews_processed_train_file']
        pp_test_file = params['agnews_processed_test_file']
        header = True

    TRAIN_FILE_PATH = os.path.join(dataset_path, train_file)
    TEST_FILE_PATH = os.path.join(dataset_path, test_file)
    TRAIN_PROCESSED_PATH = os.path.join(dataset_path, pp_train_file)
    TEST_PROCESSED_PATH = os.path.join(dataset_path, pp_test_file)

    train_data = read_data(TRAIN_FILE_PATH, header)
    test_data = read_data(TEST_FILE_PATH, header)

    if args.datacleaning == 'True':
        # Run data cleaning function
        print("Starting to preprocess the data.")
        pp_train_data = data_cleaning(train_data)
        pp_test_data = data_cleaning(test_data)
        print("Preprocessing Finished.")

    # Save data as csv file in data folder
    data_to_csv(pp_train_data, TRAIN_PROCESSED_PATH)
    data_to_csv(pp_test_data, TEST_PROCESSED_PATH)
    print("Data saved as csv file!")

