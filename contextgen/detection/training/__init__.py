import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, get_scheduler
from datasets import Dataset
import torch
import evaluate
import pathlib
import torch
from torch.utils import data

def to_categorical(row, n_class):
    row['label'] = np.eye(n_class)[row['label']]
    return row

def prepare_data(data, n_class, device, eval_size=0.1, test=False):
    from datasets import ClassLabel
    records = {'text':[], 'label':[]}
    X, Y = data
    for x, y in zip(X, Y):
        records['text'].append(x)
        records['label'].append(y)
    ds = Dataset.from_dict(records).shuffle(seed=42)
    features = ds.features.copy()
    features['label'] = ClassLabel(num_classes=n_class)
    ds = ds.cast(features)
    if test: 
        return ds.with_format("torch", device=device)
    else:
        splits = ds.train_test_split(test_size=eval_size, stratify_by_column="label")
        for k, d in splits.items():
            splits[k] = d.with_format("torch", device=device)
        return splits

def tokenize(set, tokenizer):
    return tokenizer(set['text'], truncation=True)  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])

    return metrics.compute(predictions=predictions, references=labels)
    
def train_bert(data, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    name = kwargs.pop('model_id')
    epochs = kwargs.pop('epochs', 1)
    lr = kwargs.pop('lr', 1e-5)
    n_class = kwargs.pop('n_class', 2)
    out = kwargs.pop('out', '/')

    ds = prepare_data(data, n_class, device)
    train = ds['train']
    eval = ds['test']
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_class).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    enc_train = train.map(lambda x : tokenize(x, tokenizer), batched=True)
    enc_eval = eval.map(lambda x : tokenize(x, tokenizer), batched=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_schedule = get_scheduler('linear', optimizer=optimizer, num_training_steps=len(train)*epochs, num_warmup_steps=int(len(train)//4))
    training_args = TrainingArguments(output_dir=kwargs.pop('out_dir'), 
                                      per_device_train_batch_size=kwargs.pop('batch_size', 8), 
                                      evaluation_strategy='epoch',
                                      save_strategy = 'epoch', 
                                      num_train_epochs=epochs,
                                      load_best_model_at_end=True)

    trainer = Trainer(model, 
                      args=training_args, 
                      train_dataset=enc_train, 
                      eval_dataset=enc_eval, 
                      tokenizer=tokenizer, 
                      optimizers=(optimizer, lr_schedule),
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out)

    return [model]

def test_bert(data, model, **kwargs):
    from evaluate import evaluator
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test = prepare_data(data, kwargs.pop('n_class', 2), device, test=True)
    tokenizer = AutoTokenizer.from_pretrained(kwargs.pop('model_id'))
    enc_test = test.map(lambda x : tokenize(x, tokenizer), batched=True)
    task_evaluator = evaluator("text-classification")
    return task_evaluator.compute(model, enc_test, metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]), tokenizer=tokenizer, label_mapping={"LABEL_0": 0, "LABEL_1": 1})

from pathlib import Path
import numpy as np

### LOAD

def read_data(directory):
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text())
        labels.append(parse_label(f.as_posix().replace('.txt', '.labels.tsv')))
    # labels can be empty 
    return ids, texts, labels

def parse_label(label_path):
    labels = []
    f= Path(label_path)
    
    if not f.exists():
        return labels

    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append([int(parts[2]), int(parts[3]), parts[1], 0, 0])
    labels = sorted(labels) 

    if labels:
        length = max([label[1] for label in labels]) 
        visit = np.zeros(length)
        res = []
        for label in labels:
            if sum(visit[label[0]:label[1]]):
                label[3] = 1
            else:
               visit[label[0]:label[1]] = 1
            res.append(label)
        return res 
    else:
        return labels

def clean_text(articles, ids):
    texts = []
    for article, id in zip(articles, ids):
        sentences = article.split('\n')
        start = 0
        end = -1
        res = []
        for sentence in sentences:
           start = end + 1
           end = start + len(sentence)  # length of sequence 
           if sentence != "": # if not empty line
               res.append([id, sentence, start, end])
        texts.append(res)
    return texts

def make_dataset(directory):
    ids, texts, labels = read_data(directory)
    texts = clean_text(texts, ids)
    res = []
    for text, label in zip(texts, labels):
        # making positive examples
        tmp = [] 
        pos_ind = [0] * len(text)
        for l in label:
            for i, sen in enumerate(text):
                if l[0] >= sen[2] and l[0] < sen[3] and l[1] > sen[3]:
                    l[4] = 1
                    tmp.append(sen + [l[0], sen[3], l[2], l[3], l[4]])
                    pos_ind[i] = 1
                    l[0] = sen[3] + 1
                elif l[0] != l[1] and l[0] >= sen[2] and l[0] < sen[3] and l[1] <= sen[3]: 
                    tmp.append(sen + l)
                    pos_ind[i] = 1
        # making negative examples
        dummy = [0, 0, 'O', 0, 0]
        for k, sen in enumerate(text):
            if pos_ind[k] != 1:
                tmp.append(sen+dummy)
        res.append(tmp)     
    return res
        
def make_bert_testset(dataset):

    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        tmp_sen = article[0][1]
        tmp_i = article[0][0]
        label = ['O'] * len(tmp_sen.split(' '))
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                if tmp_sen != sentence[1]:
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                    label = ['O'] * len(token_len)
                start = sentence[4] - sentence[2] 
                end = sentence[5] - sentence[2]
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)): 
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end) 
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else: 
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_sen = sentence[1]
                tmp_i = sentence[0]
            else:
                tmp_doc.append(tokens)
                tmp_id.append(sentence[0])
        if len(sentence) == 9:
            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
        words.append(tmp_doc) 
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids


def make_bert_dataset(dataset):
    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        tmp_sen = article[0][1]
        tmp_i = article[0][0]
        label = ['O'] * len(tmp_sen.split(' '))
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                if tmp_sen != sentence[1] or sentence[7]:
                    tmp_label.append(label)
                    tmp_doc.append(tmp_sen.split(' '))
                    tmp_id.append(tmp_i)
                    if tmp_sen != sentence[1]:
                        label = ['O'] * len(token_len)
                start = sentence[4] - sentence[2] 
                end = sentence[5] - sentence[2]
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)): 
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)  
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else: 
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_sen = sentence[1]
                tmp_i = sentence[0]
            else:
                tmp_doc.append(tokens)
                tmp_id.append(sentence[0])
        if len(sentence) == 9:
            tmp_label.append(label)
            tmp_doc.append(tmp_sen.split(' '))
            tmp_id.append(tmp_i)
        words.append(tmp_doc) 
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids


def mda(dataset):
    words, tags, ids= [], [], []
    for article in dataset:
        tmp_doc, tmp_label, tmp_id = [], [], []
        for sentence in article:
            tokens = sentence[1].split(' ')
            token_len = [len(token) for token in tokens]
            if len(sentence) == 9: # label exists
                start = sentence[4] - sentence[2]
                end = sentence[5] - sentence[2]
                label = ['O'] * len(token_len)
                if sentence[6] != 'O':
                    for i in range(1, len(token_len)):
                        token_len[i] += token_len[i-1] + 1
                    token_len[-1] += 1
                    token_len = np.asarray(token_len)
                    s_ind = np.min(np.where(token_len > start))
                    tmp = np.where(token_len >= end)  
                    if len(tmp[0]) != 0:
                        e_ind = np.min(tmp)
                    else:
                        e_ind = s_ind
                    for i in range(s_ind, e_ind+1):
                        label[i] = sentence[6]
                tmp_label.append(label)
            tmp_doc.append(tokens)
            tmp_id.append(sentence[0])
        words.append(tmp_doc)
        tags.append(tmp_label)
        ids.append(tmp_id)
    return words, tags, ids

### FORMAT

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_task = 1

VOCAB, tag2idx, idx2tag = [], [], []

VOCAB = ("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt"
                , "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language"
                , "Reductio_ad_hitlerum", "Bandwagon"
                , "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy"
                , "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism")

tag2idx.append({tag:idx for idx, tag in enumerate(VOCAB)})
idx2tag.append({idx:tag for idx, tag in enumerate(VOCAB)})

class PropDataset(data.Dataset):
    def __init__(self, fpath, IsTest=False):

        directory = pathlib.Path(fpath)
        dataset = make_dataset(directory)
        if IsTest:
            words, tags, ids = make_bert_testset(dataset)
        else:
            words, tags, ids = make_bert_dataset(dataset)
        flat_words, flat_tags, flat_ids = [], [], []
        for article_w, article_t, article_id in zip(words, tags, ids):
            for sentence, tag, id in zip(article_w, article_t, article_id):
                flat_words.append(sentence)
                flat_tags.append(tag)
                flat_ids.append(id)

        sents, ids = [], [] 
        tags_li = [[] for _ in range(num_task)]
   
        for word, tag, id in zip(flat_words, flat_tags, flat_ids):
            words = word
            tags = tag

            ids.append([id])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tmp_tags = []

            for i in range(num_task):
                tmp_tags.append(['O']*len(tags))
                for j, tag in enumerate(tags):
                    if tag != 'O' and tag in VOCAB:
                        tmp_tags[i][j] = tag
                tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])


        self.sents, self.ids, self.tags_li = sents, ids, tags_li
        assert len(sents) == len(ids) == len(tags_li[0])

    def return_ds(self):
        return self.sents, self.ids, self.tags_li
    
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = self.sents[idx] # tokens, tags: string list
        ids = self.ids[idx] # tokens, tags: string list
        tags = list(list(zip(*self.tags_li))[idx])

        x, is_heads = [], [] # list of ids
        y = [[] for _ in range(num_task)] # list of lists of lists
        tt = [[] for _ in range(num_task)] # list of lists of lists
        if num_task != 2:
            for w, *t in zip(words, *tags):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
    
                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))
    
                t = [[t[i]] + [t[i]] * (len(tokens) - 1) for i in range(num_task)]

                for i in range(num_task):
                    y[i].extend([tag2idx[i][each] for each in t[i]])
                    tt[i].extend(t[i])

                x.extend(xx)
                is_heads.extend(is_head)

        seqlen = len(y[0])

        words = " ".join(ids + words)

        for i in range(num_task):
            tags[i]= " ".join(tags[i]) 

        att_mask = [1] * seqlen
        return words, x, is_heads, att_mask, tags, y, seqlen

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))

    att_mask = f(-4, maxlen)
    y = []
    tags = []

    if num_task !=2:
        for i in range(num_task):
            y.append(torch.LongTensor([sample[-2][i] + [0] * (maxlen-len(sample[-2][i])) for sample in batch]))
            tags.append([sample[-3][i] for sample in batch])
    else:
        y.append(torch.LongTensor([sample[-2][0] + [0] * (maxlen-len(sample[-2][0])) for sample in batch]))
        y.append(torch.LongTensor([sample[-2][1] for sample in batch]))
        for i in range(num_task):
            tags.append([sample[-3][i] for sample in batch])


    return words, x, is_heads, att_mask, tags, y, seqlen

def load_dataset(path : str):
    with open(path, 'r') as f:
        items = map(lambda x : x.strip().split('\t'), f.readlines())
    
    x, y = map(list, zip(*items))
    return x, np.array(y, dtype=np.int8)