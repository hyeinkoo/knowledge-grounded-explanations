"""
Save KG with format [subject, predicate, object]
"""
import os
import sys
import csv
import json
from argparse import ArgumentParser
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.parameters import process_parameters_yaml


def process_relation(rel, join='_'):
    res = reversed([idx for idx in range(len(rel)) if rel[idx].isupper() and idx !=0])
    for idx in res:
        rel = rel[:idx] + join + rel[idx:]
    return rel.lower()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--kg', default="conceptnet", help='KG')
    parser.add_argument('--language', default="en")
    parser.add_argument('--multiwords', default=False, action='store_true')
    args = parser.parse_args()

    params = process_parameters_yaml()
    if args.kg == 'conceptnet':
        kg_path = params['conceptnet_path']
        kg_file = params['conceptnet_file']
        kg_triples = params['conceptnet_triples_file']
        kg_multi_words = params['conceptnet_multi_words_file']
    language = f'/c/{args.language}/'

    kg_list = []
    multi_words = []
    with open(os.path.join(kg_path, kg_file), "r") as f:
        reader = csv.reader(f, dialect='excel-tab')
        for line in tqdm(reader):
            subj = line[2]
            obj = line[3]
            if subj.startswith(language) and obj.startswith(language):
                subj = subj.split('/')[3]
                obj = obj.split('/')[3]
                rel = process_relation(line[1].split('/')[-1])
                kg_list.append([subj, rel, obj])

                if args.multiwords:
                    for word in [subj, obj]:
                        if '_' in word:
                            multi_words.append(tuple(word.split('_')))
    multi_words = list(set(multi_words))

    with open(os.path.join(kg_path, kg_triples), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(kg_list)
    print(f'Triples file saved in {os.path.join(kg_path, kg_triples)}')

    if args.multiwords:
        with open(os.path.join(kg_path, kg_multi_words), 'w') as f:
            json.dump(multi_words, f)
        print(f'Multi-words file saved in {os.path.join(kg_path, kg_multi_words)}')