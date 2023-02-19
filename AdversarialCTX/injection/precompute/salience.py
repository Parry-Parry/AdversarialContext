import argparse
import bz2
from collections import defaultdict
import logging
import multiprocessing as mp
import pickle
import numpy as np
import os
import pandas as pd
import ir_datasets
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import re 
from pandarallel import pandarallel

### Sentence Regex from: https://stackoverflow.com/a/31505798

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

parser = argparse.ArgumentParser()

parser.add_argument('-split', type=str)
parser.add_argument('-dataset', type=str)
parser.add_argument('-qrels', type=str)
parser.add_argument('-sink', type=str)

def main(args):
    with bz2.BZ2File(args.split, 'rb') as f:
        split_docs = pickle.load(f)
    ds = ir_datasets.load("msmarco-passage")
    text = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
    logging.info('Training Lexer')
    lexer = LexRank(split_docs, stopwords=STOPWORDS['en'])
    logging.info('Done!')
    qrels = pd.DataFrame(ir_datasets.load(f"msmarco-passage/{args.qrels}").qrels_iter())
    docs = qrels['doc_id'].unique().tolist()

    make_record = lambda docno, salient, nonsalient : {'docno':docno, 'salient':salient, 'nonsalient':nonsalient}

    logging.info('Computing salient positions...')
    tmp_frame = []
    for id in docs:
        scores = lexer.rank_sentences(split_into_sentences(text[id]), fast_power_method=True)
        order = np.argsort(scores)
        tmp_frame.append(make_record(id, order[-1], order[0]))
    logging.info('Saving...')
    frame = pd.DataFrame.from_records(tmp_frame)
    frame.to_csv(args.sink)
 

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(args)