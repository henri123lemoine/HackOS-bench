from betamark import ocr


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


ocr.run_eval(user_func=placeholder)
