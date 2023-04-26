from fire import Fire 
import pandas as pd
import os

# collate each file into a single dataframe, with the name of the file with extension stripped as the column name

def read(dir : str, file : str, alpha : float = None) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(dir, file), sep='\t', index_col=False)
    components = file.split('.')
    df['target'] = components[0]
    df['detector'] = components[1]
    df['type'] = components[2]
    df['injection_type'] = components[3]
    df['alpha'] = '.'.join(components[4:6])
    if alpha is not None:
        df = df[df['alpha'].astype(float) == alpha].copy()
    return df

def main(dir : str, outpath : str, alpha : float = None) -> str:
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    dfs = [read(dir, f, alpha) for f in files]
    df = pd.concat(dfs)
    df.to_csv(outpath, sep='\t', index=False)

    return f'Successfully wrote {len(df)} rows from {dir} to {outpath}'

if __name__ == '__main__':
    Fire(main)