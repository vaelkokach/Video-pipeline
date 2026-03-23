import torch

from src.attention.losses import CATLConfig, CustomAttentionTransitionLoss


def test_catl_returns_positive_scalar():
    logits = torch.randn(2, 6, 4, requires_grad=True)
    targets = torch.randint(0, 4, (2, 6))
    loss_fn = CustomAttentionTransitionLoss(CATLConfig())
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_catl_transition_penalty_increases_with_bad_transition():
    cfg = CATLConfig(transition_weight=1.0, temporal_weight=0.0, transition_margin=0.3, attention_loss_index=3)
    loss_fn = CustomAttentionTransitionLoss(cfg)
    targets = torch.tensor([[1, 1, 3, 3]], dtype=torch.long)

    good_logits = torch.tensor(
        [[[2.0, 0.2, 0.1, -1.0], [1.7, 0.1, 0.1, -1.0], [0.1, 0.2, 0.3, 2.5], [0.0, 0.2, 0.1, 2.2]]],
        dtype=torch.float32,
    )
    bad_logits = torch.tensor(
        [[[2.0, 0.2, 0.1, -1.0], [1.7, 0.1, 0.1, -1.0], [0.3, 0.2, 0.2, 0.1], [0.3, 0.2, 0.2, 0.1]]],
        dtype=torch.float32,
    )
    good_loss = loss_fn(good_logits, targets)
    bad_loss = loss_fn(bad_logits, targets)
    assert float(bad_loss.item()) > float(good_loss.item())
