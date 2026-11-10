from workloads.q1.progress import Progress, format_duration


def test_progress_logs_at_most_once_per_interval():
    now = [0.0]
    lines = []
    p = Progress("encode", 100, "texts", interval=30.0, log=lines.append, clock=lambda: now[0])
    now[0] = 10.0
    assert p.update(10) is None
    now[0] = 31.0
    text = p.update(31)
    assert text == lines[0] and "31/100 texts" in text and "1.0 texts/s" in text and "ETA 69 s" in text
    now[0] = 40.0
    assert p.update(40) is None and len(lines) == 1
    assert p.update(40, force=True) is not None


def test_rate_counts_only_work_since_the_start():
    now = [0.0]
    p = Progress("x", 1000, "q", done=900, interval=0.0, log=lambda s: None, clock=lambda: now[0])
    now[0] = 10.0
    text = p.update(1000)
    assert "10.0 q/s" in text and "(100%)" in text


def test_format_duration():
    assert format_duration(30) == "30 s"
    assert format_duration(600) == "10.0 min"
    assert format_duration(7200) == "2.0 h"
