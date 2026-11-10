from workloads.q1 import cost


def test_more_threads_are_faster_and_larger_problems_take_longer():
    one = cost.ground_truth_seconds(1000, 10 ** 6, 768, 1)
    assert cost.ground_truth_seconds(1000, 10 ** 6, 768, 8) < one
    assert cost.ground_truth_seconds(2000, 10 ** 6, 768, 1) > one
    assert cost.encode_seconds("esci", 10 ** 6, 8) < cost.encode_seconds("esci", 10 ** 6, 1)


def test_the_rate_is_interpolated_between_measured_dimensions():
    low = cost.ground_truth_seconds(1000, 10 ** 6, 128, 1) / (128 * 1.0)
    high = cost.ground_truth_seconds(1000, 10 ** 6, 384, 1) / (384 * 1.0)
    mid = cost.ground_truth_seconds(1000, 10 ** 6, 256, 1) / (256 * 1.0)
    assert high < mid < low                     # seconds per dimension fall as the matmul dominates


def test_released_vector_datasets_have_no_embedding_time():
    assert cost.encode_seconds("spacev", 10 ** 6, 8) == 0.0
