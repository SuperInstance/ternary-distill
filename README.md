# ternary-distill

*Knowledge distillation for ternary networks. A float teacher produces soft targets; a ternary student learns from both the distribution and the hard labels.*

## Why This Exists

You can't just quantize a float network to ternary and expect the same accuracy. The information loss is too severe. Knowledge distillation is the bridge: a full-precision teacher network produces soft probability distributions that contain far more information than hard labels. The ternary student learns from these distributions, guided by both the teacher's knowledge and the ground truth.

This crate implements the distillation loop: soft target generation, loss computation, and a ternarization schedule that gradually converts float weights to ternary during training.

## Architecture

```
Float Teacher → soft targets [0.7, 0.2, 0.1]
                        ↓ distillation loss
Ternary Student → hard predictions [-1, 0, +1]
                        + hard labels
                        ↓ combined loss
                   Weight updates → ternarize() → next step
```

### Key Types

- **`SoftTarget`** — Teacher's probability distribution for a single example
- **`DistillationLoss`** — Combines KL divergence (teacher-student) with cross-entropy (student-labels). Temperature-controlled.
- **`TernarizationSchedule`** — Gradually convert float weights to ternary. Linear, cosine, or step schedule.
- **`DistillationTracker`** — Track teacher agreement, student accuracy, and loss components over training.

## Usage

```rust
use ternary_distill::*;

// Teacher produces soft targets
let soft = SoftTarget::from_logits(&[2.0, 0.5, -0.5], 2.0); // temperature=2

// Compute distillation loss
let loss = DistillationLoss::new(0.7, 3.0); // alpha=0.7, temperature=3.0
let total = loss.compute(&soft, &student_logits, &hard_label);

// Schedule ternarization
let schedule = TernarizationSchedule::cosine(100); // 100 epochs
let ratio = schedule.ratio_at_epoch(50); // 0.5 of weights are ternary at epoch 50
```

## The Deeper Idea

Distillation reveals something profound about ternary networks: the *information* is in the distribution, not the precision. A ternary network can capture the same decision boundaries as a float network — it just needs to be taught correctly. The soft targets are the curriculum.

This connects to `ternary-optimizer` (the training loop), `ternary-prune` (complementary compression), and the agent-cognition crates (distillation is a form of teaching/learning between agents).

## Related Crates

- `ternary-prune` — Complementary compression via sparsity
- `ternary-optimizer` — Training optimizers for ternary weights
- `ternary-accumulator` — Gradient accumulation during distillation
- `ternary-loss` — Loss functions including distillation losses
