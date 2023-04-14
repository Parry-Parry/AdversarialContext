
def interp_fusion(rel : float, prop : float, alpha : float = 0.9, pi : int = 60, linear : bool = False) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    if linear: return alpha * rel + (1 - alpha) * prop # Perform linear fusion
    else: return 1 / (pi + alpha * rel) + 1 / (pi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def rrfusion(rel : float, prop : float, alpha : float = 0.9, relpi : int = 4, proppi : int = 10) -> float:
    rel = (rel + 1) / 2  # normalise cosine similarity with theoretical max & min
    return 1 / (relpi + alpha * rel) + 1 / (proppi + (1 - alpha) * prop) # Perform reciprocal rank fusion

def alphafusion(rel : float, prop : float, alpha : float = 0.9):
    return (rel + 1) / 2 + alpha * prop

def priorityfusion(rel : float, prop : float, alpha : float = 0.9):
    pass 