from collections.abc import Mapping, Sequence


def ranking_parity(
    baseline: Sequence[str],
    candidate: Sequence[str],
    *,
    top_values: Sequence[int] = (1, 3, 10),
) -> dict[str, float]:
    if len(baseline) != len(candidate) or set(baseline) != set(candidate):
        raise ValueError("rankings must contain the same unique document IDs")
    if len(set(baseline)) != len(baseline):
        raise ValueError("rankings must not contain duplicate document IDs")
    count = len(baseline)
    if count < 2:
        spearman = kendall = 1.0
    else:
        baseline_rank = {document_id: index for index, document_id in enumerate(baseline)}
        candidate_rank = {document_id: index for index, document_id in enumerate(candidate)}
        squared_difference = sum(
            (baseline_rank[document_id] - candidate_rank[document_id]) ** 2
            for document_id in baseline
        )
        spearman = 1 - (6 * squared_difference) / (count * (count**2 - 1))
        concordant = discordant = 0
        for left in range(count):
            for right in range(left + 1, count):
                ordered = baseline[left], baseline[right]
                if candidate_rank[ordered[0]] < candidate_rank[ordered[1]]:
                    concordant += 1
                else:
                    discordant += 1
        kendall = (concordant - discordant) / (concordant + discordant)
    result = {
        "spearman": float(spearman),
        "kendall_tau": float(kendall),
        "top_1_agreement": float(bool(baseline) and baseline[0] == candidate[0]),
    }
    for requested_top in top_values:
        top = min(requested_top, count)
        overlap = len(set(baseline[:top]) & set(candidate[:top])) / top if top else 1.0
        result[f"top_{requested_top}_overlap"] = overlap
    return result


def score_deviation(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, float]:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("score maps must contain the same non-empty document IDs")
    differences = [abs(baseline[key] - candidate[key]) for key in baseline]
    return {
        "mean_absolute_deviation": sum(differences) / len(differences),
        "max_absolute_deviation": max(differences),
    }
