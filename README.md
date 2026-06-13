# Ternary Distill — Knowledge Distillation for Ternary Neural Networks

**Ternary Distill** compresses knowledge from a full-precision teacher network into ternary student weights {-1, 0, +1}. It implements temperature-scaled softmax distillation, blends soft teacher targets with hard labels, and converts the result to ternary votes via threshold-based quantization. The KL divergence between teacher and student distributions is tracked to measure distillation quality.

## Why It Matters

Knowledge distillation is the standard technique for deploying large models on resource-constrained hardware. In the ternary setting, distillation is especially critical: a ternary student network has only 3 possible weight values, so it needs high-quality guidance to match the teacher's behavior. The teacher's soft probability distribution contains "dark knowledge" — information about class similarity that hard labels discard. By blending soft targets (α weight) with hard labels (1-α weight), the student learns both the teacher's nuanced understanding and the ground truth. This is how ternary networks achieve accuracy within 1-2% of full-precision models while using 16× less memory.

## How It Works

### Soft Target Generation

The teacher's logits are converted to soft probabilities using temperature-scaled softmax:

```
pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

where T is the temperature. Higher T produces softer distributions that reveal more inter-class relationships. T = 1 gives standard softmax; T → ∞ gives uniform.

### Blending

Soft targets are blended with hard labels:

```
blended = α · soft_target + (1 - α) · hard_label
```

where α ∈ [0, 1] controls the teacher influence. Typical values: α = 0.7 for strong teacher guidance.

### Ternary Quantization

The blended distribution is converted to a ternary vote by finding the maximum probability index and mapping it:

```
vote = argmax_index < n/3 ? -1 : argmax_index ≥ 2n/3 ? +1 : 0
```

This partitions the class space into three zones: negative (first third), neutral (middle third), positive (last third).

### KL Divergence Tracking

The KL divergence D(p_teacher || p_student) measures how much information is lost in quantization. A well-distilled model has KL < 0.1 nats. The divergence is:

```
D_KL(p || q) = Σᵢ pᵢ · log(pᵢ / qᵢ)
```

Computed in O(n) for n classes. Minimizing this during training guides the student toward the teacher's distribution.

## Quick Start

```rust
use ternary_distill::{SoftTarget, distill_ternary};

// Teacher produces logits for 3 classes
let teacher_logits = vec![vec![2.0, 1.0, 0.1]];
let teacher_probs = vec![vec![0.7, 0.2, 0.1]];

// Distill to ternary votes
let ternary_votes = distill_ternary(
    &teacher_probs,
    &teacher_logits,
    2.0,   // temperature
    0.7,   // alpha (teacher weight)
);

// SoftTarget from logits
let soft = SoftTarget::from_logits(vec![2.0, 1.0, 0.1], 2.0);
let vote = soft.to_ternary_vote();
```

```bash
cargo add ternary-distill
```

## API

| Type / Function | Description |
|---|---|
| `SoftTarget` | Teacher distribution: `from_probs()`, `from_logits(temp)`, `to_ternary_vote()` |
| `distill_ternary(probs, logits, temp, alpha)` | Batch distillation → `Vec<Vec<Trit>>` |
| `Trit` | Type alias for `i8` |

## Architecture Notes

Distillation is the model compression pathway in **SuperInstance**: full-precision models are distilled to ternary weights for fleet deployment. The γ + η = C conservation manifests in the distillation trade-off: higher α (teacher weight) preserves more γ (information) but slows student convergence (higher η entropy in optimization). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Hinton, Geoffrey et al. "Distilling the Knowledge in a Neural Network," *NeurIPS Workshop*, 2015 — seminal distillation paper.
- Li, Feng et al. "Ternary Weight Networks," *arXiv:1605.04711*, 2016 — ternary quantization.
- Polino, Antonio et al. "Model Compression via Distillation and Quantization," *ICLR*, 2018.

## License

MIT
