from nltk.tokenize import sent_tokenize

def batch_iter(items, batch_size):
    i = 0
    while i < (len(items) // batch_size):
        i += 1
        yield items[i*batch_size:min((i+1)*batch_size, len(items))]

def parse_span(text):
    spans = sent_tokenize(text)
    if len(spans) == 1: return text
    elif len(spans) == 0: return text
    else: return spans[0]