import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results
from fire import Fire
import ir_datasets as irds
import os
from . import Scorer
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def extract_attrs(name):
    parts = name.split('.')
    if 'sal' in name:
        return {
            'generator' : parts[0],
            'position' : parts[2],
            'position_type' : 'relative',
            'item' : parts[5]
        }
    else:
        return {
            'generator' : parts[0],
            'position' : parts[1],
            'position_type' : 'absolute',
            'item' : parts[4]
        }


def read_injection(file : str):
    name = os.path.basename(file).replace('.tsv.gz', '')
    attrs = extract_attrs(name)
    df = pd.read_csv(file, sep='\t', dtype={'qid' : str, 'docno' : str, 'text' : str, 'item' : str})
    df['generator'] = attrs['generator']
    df['position'] = attrs['position']
    df['item'] = attrs['item']
    df['position_type'] = attrs['position_type']

    return df


def classify(model_id : str,
             baseline_dir : str, 
             injection_dir : str, 
             out_dir : str, 
             ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged',
             window_size : int = 0,
             batch_size : int = 128):
    
    ### set up scorer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    scorer = Scorer(model, tokenizer, window_size, batch_size, classifier=True)

    ### set up dataset

    ds = irds.load(ir_dataset)
    docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()

    ### set up baselines
    baseline_files = os.listdir(baseline_dir)
    baseline_examples = pd.concat([read_results(os.path.join(baseline_dir, file)) for file in baseline_files])
    baseline_examples.drop_duplicates(subset=['qid', 'docno'], inplace=True)
    baseline_examples['text'] = baseline_examples['docno'].apply(lambda x : docs[x])
    baseline_examples['label'] = 0

    TOTAL_BASELINE = len(baseline_examples)

    ### set up injections

    injection_dirs = [f for f in os.listdir(injection_dir) if f != 'qrel']
    injection_files = [os.path.join(injection_dir, dir, file) for dir in injection_dirs for file in os.listdir(os.path.join(injection_dir, dir))]
    injection_examples = pd.concat([read_injection(file) for file in injection_files])
    injection_examples.drop_duplicates(subset=['qid', 'docno', 'generator', 'position', 'item'], inplace=True)
    injection_examples['label'] = 1

    for generator in injection_examples.generator.unique():
        generator_examples = injection_examples[injection_examples.generator == generator]
        for position_type in injection_examples.position_type.unique():
            type_examples = generator_examples[generator_examples.position_type == position_type]
            for position in type_examples.position.unique():
                position_examples = type_examples[type_examples.position == position]
                position_examples = position_examples[['text', 'label']].sample(n=TOTAL_BASELINE, replace=True)

                current = pd.concat([baseline_examples[['text', 'label']], position_examples])
                current = current.sample(frac=1).reset_index(drop=True)

                current['pred'] = scorer(current['text'].tolist())

                print(current.head())

                # get accuracy f1, precision, recall for label and pred

                acc = accuracy_score(current['label'].to_list(), current['pred'].to_list())
                f1 = f1_score(current['label'].to_list(), current['pred'].to_list(), average='binary')
                precision = precision_score(current['label'].to_list(), current['pred'].to_list(), average='binary')
                recall = recall_score(current['label'].to_list(), current['pred'].to_list(), average='binary')

                out = {
                    'generator' : generator,
                    'position' : position,
                    'position_type' : position_type,
                    'window_size' : window_size,
                    'accuracy' : acc,
                    'f1' : f1,
                    'precision' : precision,
                    'recall' : recall
                }

                json_file = os.path.join(out_dir, f'{generator}.{position_type}.{position}.json')
                json.dump(out, open(json_file, 'w'))
                

    return "Done!"

if __name__ == '__main__':
    Fire(classify)