def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_val = min_val = scores[0]
    for score in scores:
        if score < min_val:
            min_val = score
        elif score > max_val:
            max_val = score

    if max_val == min_val:
        return [1.0] * len(scores)

    n_scores = list(map(lambda x: (x - min_val) / (max_val - min_val), scores))

    return n_scores
