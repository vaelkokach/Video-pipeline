# Theoretical Contribution: CATL

This repository introduces **Custom Attention Transition Loss (CATL)** for attention-loss detection.

## Motivation

Attention-loss events are sparse and temporally structured. Standard cross-entropy often under-weights rare transition moments and over-reacts to short-term noise. CATL targets the cause with three coordinated terms:

1. **Class-balanced focal term** for rare event emphasis.
2. **Temporal consistency term** to reduce prediction jitter.
3. **Transition margin term** to force clearer confidence separation at pre-loss to loss boundaries.

## Formulation

For logits `z_{b,t,c}` and targets `y_{b,t}`:

- Focal component:
  - `L_focal = alpha * (1 - p_t)^gamma * CE(z, y)`
- Temporal component:
  - `L_temp = mean(||softmax(z_{t}) - softmax(z_{t-1})||^2)`
- Transition component:
  - For transitions `y_{t-1} != lossClass` and `y_t == lossClass`:
  - `L_trans = mean(max(0, m - (p_loss,t - p_loss,t-1)))`

Final loss:

`L_CATL = L_focal + lambda_temp * L_temp + lambda_trans * L_trans`

## Expected Benefit

Compared with CE/focal baselines, CATL is designed to improve:

- early detection quality,
- temporal stability,
- robustness to short-lived pose/emotion noise,
- calibration around transition boundaries.

## Evidence Protocol

Use:

- `scripts/train_attention.py` for CATL vs non-CATL training.
- `scripts/run_ablation.py` for baseline ranking and significance.
- `src/eval/` metrics for macro F1, AUROC/AUPRC, ECE, and temporal flip rate.
