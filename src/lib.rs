//! # ternary-distill
//!
//! Knowledge distillation for ternary networks.
//! A float teacher produces soft targets. A ternary student learns from them,
//! guided by both the teacher's distribution and the hard labels.
//!
//! Connected to the [`ternary-types`](https://github.com/SuperInstance/ternary-types)
//! fleet — use `ternary_types::Ternary` for cross-crate interop.

pub type Trit = i8;

/// Soft target distribution from teacher.
#[derive(Debug, Clone)]
pub struct SoftTarget {
    /// Logits or probabilities for each class.
    pub probs: Vec<f64>,
}

impl SoftTarget {
    pub fn from_probs(probs: Vec<f64>) -> Self {
        Self { probs }
    }

    /// Create from logits using softmax with temperature.
    pub fn from_logits(logits: Vec<f64>, temperature: f64) -> Self {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter()
            .map(|l| ((l - max_logit) / temperature).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        Self { probs: exps.iter().map(|e| e / sum).collect() }
    }

    /// Ternary hard decision from soft target.
    pub fn to_ternary_vote(&self) -> Trit {
        let max_idx = self.probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let n = self.probs.len().max(1);
        if max_idx < n / 3 { -1 }
        else if max_idx >= 2 * n / 3 { 1 }
        else { 0 }
    }
}

/// Distill a teacher's soft targets into a student's ternary weights.
///
/// # Arguments
/// * `teacher_probs` — Probability distributions from teacher (classes × samples)
/// * `teacher_logits` — Raw logits from teacher (classes × samples)
/// * `temperature` — Softmax temperature (higher = softer targets)
/// * `alpha` — Weight given to teacher soft targets vs hard labels (0..1)
///
/// # Returns
/// The hard ternary votes from the most confident teacher predictions.
pub fn distill_ternary(
    teacher_probs: &[Vec<f64>],
    teacher_logits: &[Vec<f64>],
    temperature: f64,
    alpha: f64,
) -> Vec<Vec<Trit>> {
    assert_eq!(teacher_probs.len(), teacher_logits.len(),
        "teacher_probs and teacher_logits must have same number of samples");

    teacher_logits.iter().enumerate().map(|(i, logits)| {
        let soft_target = SoftTarget::from_logits(logits.clone(), temperature);

        // Blend soft target with hard label from probs
        let blended: Vec<f64> = soft_target.probs.iter().enumerate().map(|(j, &soft)| {
            let hard = teacher_probs[i][j];
            alpha * soft + (1.0 - alpha) * hard
        }).collect();

        // Convert blended distribution to ternary votes
        let blended_soft = SoftTarget { probs: blended };
        vec![blended_soft.to_ternary_vote()]
    }).collect()
}

/// KL divergence between two probability distributions.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi == 0.0 { 0.0 }
            else { pi * (pi / qi).ln() }
        })
        .sum()
}

/// Jensen-Shannon divergence: symmetric version of KL.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_target_from_logits() {
        let logits = vec![2.0, 1.0, 0.1];
        let st = SoftTarget::from_logits(logits, 1.0);
        assert!((st.probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(st.probs[0] > st.probs[1]);
        assert!(st.probs[1] > st.probs[2]);
    }

    #[test]
    fn test_to_ternary_vote_three_classes() {
        let st = SoftTarget { probs: vec![0.8, 0.1, 0.1] };
        assert_eq!(st.to_ternary_vote(), -1); // top third → -1

        let st = SoftTarget { probs: vec![0.1, 0.8, 0.1] };
        assert_eq!(st.to_ternary_vote(), 0); // middle third → 0

        let st = SoftTarget { probs: vec![0.1, 0.1, 0.8] };
        assert_eq!(st.to_ternary_vote(), 1); // bottom third → +1
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!((kl_divergence(&p, &q) - 0.0).abs() < 1e-6);

        let q2 = vec![0.9, 0.1];
        assert!(kl_divergence(&p, &q2) > 0.0);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = vec![0.8, 0.2];
        let q = vec![0.3, 0.7];
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert!((js_pq - js_qp).abs() < 1e-10);
    }
}
