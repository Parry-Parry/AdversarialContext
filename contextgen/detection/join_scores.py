import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results, write_results
from pyterrier.model import add_ranks
from fire import Fire

def interpolated_scores(score_file : str, rel_file : str, out_file : str, alpha : float = 1.0):

    # Load
    rel = read_results(rel_file)
    score = read_results(score_file)

    # Min max normalise rel score over each qid
    rel['score'] = rel.groupby('qid')['score'].transform(lambda x : (x - x.min()) / (x.max() - x.min()))

    # Transform by alpha interpolation 
    rel['score'] = rel['score'] * alpha
    score['score'] = score['score'] * (1 - alpha)
    score['promo'] = score['score'] 
    score.drop(columns=['score'], inplace=True)

    rel = rel.merge(score, on=['qid', 'docno'], how='left')
    rel['score'] = rel['score'] + rel['promo']

    write_results(add_ranks(rel), out_file)

    return "Done!"

if __name__ == '__main__':
    Fire(interpolated_scores)