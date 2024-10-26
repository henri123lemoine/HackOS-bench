# from betamark import bicycle
from src import bicycle

def placeholder(x):
    """
    Params:
    -------
    x: string representing a genomic sequence

    Returns:
    --------
    y_pred: int where 0 is negative (not an OCR) or 1 (is an OCR)
    """

    return 0


if __name__ == "__main__":
    score = bicycle.run_eval(user_func=placeholder)
    print(f"{score = }")
