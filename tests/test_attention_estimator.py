from src.attention.estimator import AttentionEstimator, AttentionObservation


def test_attention_estimator_bounds():
    estimator = AttentionEstimator()
    obs = AttentionObservation(
        student_id=1,
        bbox=[10, 10, 80, 160],
        frame_width=640,
        frame_height=480,
        action_label="reading",
        emotion_label="curiosity",
    )
    out = estimator.predict(obs)
    assert 0.0 <= out.attention_score <= 1.0
    assert 0.0 <= out.loss_probability <= 1.0


def test_attention_estimator_distraction_lower_score():
    estimator = AttentionEstimator()
    engaged = estimator.predict(
        AttentionObservation(
            student_id=1,
            bbox=[200, 120, 280, 320],
            frame_width=640,
            frame_height=480,
            action_label="reading",
            emotion_label="curiosity",
        )
    )
    distracted = estimator.predict(
        AttentionObservation(
            student_id=1,
            bbox=[5, 20, 75, 110],
            frame_width=640,
            frame_height=480,
            action_label="using phone",
            emotion_label="boredom",
        )
    )
    assert distracted.attention_score < engaged.attention_score
