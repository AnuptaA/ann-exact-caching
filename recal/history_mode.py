"""Common history split: the first part of a stream is history, the rest is scored."""


def history_length(n_requests: int, frac: float) -> int:
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"history fraction {frac} is not in [0, 1)")
    return int(frac * n_requests)


def stamp(result: dict, n_requests: int, history_len: int) -> dict:
    result.update({
        "history_frac_used": history_len / n_requests if n_requests else 0.0,
        "history_len": history_len,
        "first_scored": history_len,
        "n_scored": n_requests - history_len,
    })
    return result
