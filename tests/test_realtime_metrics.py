from src.eval.realtime import RealtimeSample, summarize_realtime


def test_realtime_summary():
    samples = [
        RealtimeSample(frame_latency_ms=30, fps=25, cpu_percent=45, gpu_percent=30),
        RealtimeSample(frame_latency_ms=35, fps=24, cpu_percent=47, gpu_percent=31),
        RealtimeSample(frame_latency_ms=28, fps=27, cpu_percent=44, gpu_percent=32),
    ]
    out = summarize_realtime(samples)
    assert out["avg_latency_ms"] > 0
    assert out["p95_latency_ms"] >= out["avg_latency_ms"]
    assert out["avg_fps"] > 0
