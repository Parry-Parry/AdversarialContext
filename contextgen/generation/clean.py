from fire import Fire
import pandas as pd

from contextgen import parse_span

def clean(file : str, out_file : str, cutoff : int = None):

    def clean_func(text): 
        parsed = parse_span(text)
        if len(parsed) < 3: return text
        if cutoff: parsed = parsed[:cutoff]
        return parsed

    df = pd.read_csv(file, sep='\t', index_col=False)
    df['span'] = df['span'].apply(clean_func)
    df.to_csv(out_file, sep='\t', index=False)

    return "Done!"

if __name__ == '__main__':
    Fire(clean)