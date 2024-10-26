from src.dataset import bicycle


def placeholder(x):
    """
    Params:
    -------
    x: NumPy array representation of an image (dimensions are non-fixed)

    Returns:
    --------
    y_pred: int where 0 is negative (no bicycle) or 1 (there is a bicycle)
    """
    return 0


if __name__ == "__main__":
    score = bicycle.run_eval(user_func=placeholder)
    print(f"{score = }")
