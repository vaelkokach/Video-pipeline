from pathlib import Path

from src.attention.taxonomy import AttentionLevel
from src.data.adapters import DAiSEEAdapter, split_samples


def test_daisee_adapter_and_split(tmp_path: Path):
    csv_path = tmp_path / "daisee.csv"
    csv_path.write_text(
        "clip_path,engagement_label,split,student_id\n"
        "a.mp4,high,train,1\n"
        "b.mp4,low,val,2\n"
        "c.mp4,very_low,test,3\n",
        encoding="utf-8",
    )

    adapter = DAiSEEAdapter(csv_path)
    samples = adapter.to_attention_samples()
    assert len(samples) == 3
    assert samples[0].level == AttentionLevel.ENGAGED
    assert samples[1].level == AttentionLevel.DISTRACTED
    assert samples[2].level == AttentionLevel.ATTENTION_LOSS

    train, val, test = split_samples(samples)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
