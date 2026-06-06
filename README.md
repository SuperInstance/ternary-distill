# ternary-distill

*The teacher speaks in probabilities. The student answers in trits.*

---

Knowledge distillation for ternary networks. A full-precision teacher network produces soft probability distributions. A ternary student (weights in {-1, 0, +1}) learns from both the teacher's soft targets and the hard ground-truth labels.

The crate implements: softmax with temperature for soft targets, KL divergence between teacher/student distributions, combined distillation + hard-label loss (configurable alpha), gradual ternarization schedule (warmup → anneal), and a distillation tracker for monitoring progress.

The key insight: ternary students can match teacher accuracy at 16× compression because most of the information in float weights is noise. The signal fits in three values. The teacher's job is to show which three values matter most.

11 tests covering soft targets, KL divergence, distillation loss ordering, ternarization schedule, ternarize values, tracker statistics, compression ratio.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
