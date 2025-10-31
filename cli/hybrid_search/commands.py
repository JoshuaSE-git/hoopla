from .helpers import normalize_scores


def handler_normalize(scores: list[float]):
    n_scores = normalize_scores(scores)
    for score in n_scores:
        print(f"* {score:.4f}")
