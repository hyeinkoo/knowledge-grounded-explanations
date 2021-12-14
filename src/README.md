# knowledge-grounded-explanations 

## Setup
Install requirements
```
pip install -r requirements.txt
```
Download nltk data manually
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Data
- Download [AG News Classification Dataset](https://www.kaggle.com/amananandrai/ag-news-classification-dataset/download)
 and save it in `data/ag_news/`
- Download [ConceptNet](https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz)
 and save it in `data/conceptnet/`
- Download [multi-word expressions resource](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/learn-multi-word-paraphrasing.html) and save it in `data/conceptnet/`

[comment]: <> (* [WordNet]&#40;http://wordnetcode.princeton.edu/wn3.1.dict.tar.gz&#41;)

[comment]: <> (* [Freebase Easy]&#40;https://freebase-easy.cs.uni-freiburg.de/dump/freebase-easy-latest.zip&#41;)

[comment]: <> (### Extract Subgraph)

[comment]: <> (```)

[comment]: <> (python src/data/extract_subgraph.py --export_dir data/20NewsGroupExport2hops --knowledge_graph data/conceptnet-assertions-5.7.0.csv --data_dir data --data_set 20NewsGroup --n_hops 2)

[comment]: <> (```)

### Preprocessing
Preprocess conceptnet
```
python src/data/preprocess_kg.py
```

Preprocess agnews dataset
```
python src/data/data_preprocessing.py  --dataset agnews --datacleaning True
```

## Train
```
python src/train.py --dataset agnews --model kbert --batch_size 16 --max_epochs 10 --gpus 1
```

## Evaluation
```
python src/evaluation.py --dataset agnews --model_path kbert/model_best.ckpt --batch_size 16
```

## Explainability
Check out `notebooks/captum.ipynb`
