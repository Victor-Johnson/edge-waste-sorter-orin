from datetime import datetime, timedelta, timezone

from scripts.jetson_power_probe import parse_sample, within


def test_tegrastats_sample_and_utc_alignment():
    line = ("11-05-2026 01:28:31 RAM 2897/7620MB GR3D_FREQ 42% "
            "gpu@47.968C VDD_IN 4454mW/4397mW")
    sample = parse_sample(line, timezone(timedelta(hours=8)))
    assert sample == {"timestamp_utc": datetime(2026, 11, 4, 17, 28, 31,
                                                tzinfo=timezone.utc),
                      "vdd_in_w": 4.454, "gpu_temp_c": 47.968,
                      "gpu_util_pct": 42}
    start = datetime(2026, 11, 4, 17, 28, 29, 300000, tzinfo=timezone.utc)
    end = datetime(2026, 11, 4, 17, 28, 32, 400000, tzinfo=timezone.utc)
    assert within([sample], start, end) == [sample]
