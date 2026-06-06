//! # ternary-distill
//!
//! Knowledge distillation for ternary networks.
//! A float teacher produces soft targets. A ternary student learns from them,
//! guided by both the teacher's distribution and the hard labels.

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
    /// Maps top third → +1, middle third → 0, bottom third → -1.
    pub fn to_ternary_vote(&self) -> Trit {
        let max_idx = self.probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let n = self.probs.len().max(1);
        if max_idx < n / 3 { -1 }
        else if max_idx >= 2 * n / 3 { 1 }
        else { 0 }
    }

    /// KL divergence from this distribution to another.
    pub fn kl_divergence(&self, other: &SoftTarget) -> f64 {
        self.probs.iter().zip(other.probs.iter())
            .map(|(&p, &q)| {
                if p > 0.0 && q > 0.0 { p * (p / q).ln() }
                else { 0.0 }
            })
            .sum()
    }
}

/// Distillation loss combining teacher guidance and hard label supervision.
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Temperature for soft target sharpening.
    pub temperature: f64,
    /// Weight for distillation loss vs hard label loss.
    pub alpha: f64,
}

impl DistillationLoss {
    pub fn new(temperature: f64, alpha: f64) -> Self {
        Self { temperature, alpha }
    }

    /// Compute combined loss.
    /// student_logits: raw output from student network
    /// teacher_probs: soft targets from teacher
    /// hard_label: ground truth class index
    pub fn compute(&self, student_logits: &[f64], teacher_probs: &[f64], hard_label: usize) -> f64 {
        let n = student_logits.len().min(teacher_probs.len());
        let teacher_soft = SoftTarget::from_logits(student_logits.to_vec(), self.temperature);
        let teacher_target = SoftTarget::from_probs(teacher_probs.to_vec());

        // Distillation loss: KL divergence between student and teacher soft targets
        let distill_loss = teacher_soft.kl_divergence(&teacher_target) * (self.temperature * self.temperature);

        // Hard label loss: negative log likelihood
        let student_probs = SoftTarget::from_logits(student_logits.to_vec(), 1.0);
        let hard_loss = if hard_label < student_probs.probs.len() && student_probs.probs[hard_label] > 0.0 {
            -student_probs.probs[hard_label].ln()
        } else {
            10.0 // large loss for wrong
        };

        self.alpha * distill_loss + (1.0 - self.alpha) * hard_loss
    }
}

/// Ternarization schedule: gradually move from float to ternary during distillation.
#[derive(Debug, Clone)]
pub struct TernarizationSchedule {
    /// Step at which ternarization begins.
    pub warmup_steps: usize,
    /// Total steps over which ternarization happens.
    pub anneal_steps: usize,
}

impl TernarizationSchedule {
    pub fn new(warmup: usize, anneal: usize) -> Self {
        Self { warmup_steps: warmup, anneal_steps: anneal }
    }

    /// Get ternarization probability at a given step.
    /// 0.0 = all float, 1.0 = all ternary.
    pub fn ternary_prob(&self, step: usize) -> f64 {
        if step < self.warmup_steps { return 0.0; }
        let progress = (step - self.warmup_steps) as f64 / self.anneal_steps as f64;
        progress.min(1.0)
    }

    /// Ternarize a value based on current schedule step.
    pub fn ternarize(&self, value: f64, step: usize) -> Trit {
        if self.ternary_prob(step) >= 1.0 || (self.ternary_prob(step) > 0.0 && self.should_ternarize(step)) {
            if value > 0.3 { 1 }
            else if value < -0.3 { -1 }
            else { 0 }
        } else {
            // Keep as rounded trit-like value but don't force
            if value > 0.5 { 1 } else if value < -0.5 { -1 } else { 0 }
        }
    }

    fn should_ternarize(&self, step: usize) -> bool {
        // Deterministic based on step
        let hash = (step as u64).wrapping_mul(6364136223846793005);
        ((hash >> 32) as f64) / (u32::MAX as f64) < self.ternary_prob(step)
    }
}

/// Teacher-student pair for distillation tracking.
#[derive(Debug)]
pub struct DistillationTracker {
    pub student_name: String,
    pub teacher_name: String,
    pub steps: usize,
    pub total_distill_loss: f64,
    pub total_hard_loss: f64,
    pub best_accuracy: f64,
}

impl DistillationTracker {
    pub fn new(student: &str, teacher: &str) -> Self {
        Self {
            student_name: student.to_string(),
            teacher_name: teacher.to_string(),
            steps: 0,
            total_distill_loss: 0.0,
            total_hard_loss: 0.0,
            best_accuracy: 0.0,
        }
    }

    pub fn record_step(&mut self, distill_loss: f64, hard_loss: f64, accuracy: f64) {
        self.steps += 1;
        self.total_distill_loss += distill_loss;
        self.total_hard_loss += hard_loss;
        if accuracy > self.best_accuracy {
            self.best_accuracy = accuracy;
        }
    }

    pub fn avg_distill_loss(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.total_distill_loss / self.steps as f64 }
    }

    pub fn avg_hard_loss(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.total_hard_loss / self.steps as f64 }
    }

    /// Compression ratio: teacher params / student params.
    pub fn compression_ratio(&self, teacher_params: usize, student_params: usize) -> f64 {
        if student_params == 0 { 0.0 } else { teacher_params as f64 / student_params as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_target_from_logits() {
        let st = SoftTarget::from_logits(vec![1.0, 2.0, 3.0], 1.0);
        assert!(st.probs[2] > st.probs[1]);
        assert!(st.probs[1] > st.probs[0]);
        let sum: f64 = st.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_soft_target_ternary_vote() {
        let st = SoftTarget::from_probs(vec![0.1, 0.1, 0.8]);
        assert_eq!(st.to_ternary_vote(), 1); // top third with highest prob
    }

    #[test]
    fn test_kl_divergence_same() {
        let st = SoftTarget::from_probs(vec![0.5, 0.5]);
        assert!(st.kl_divergence(&st) < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let a = SoftTarget::from_probs(vec![0.9, 0.1]);
        let b = SoftTarget::from_probs(vec![0.1, 0.9]);
        assert!(a.kl_divergence(&b) > 0.5);
    }

    #[test]
    fn test_distillation_loss_higher_when_wrong() {
        let dl = DistillationLoss::new(2.0, 0.7);
        let teacher = vec![0.9, 0.05, 0.05];
        // Good student: logits match teacher
        let good = dl.compute(&[3.0, 0.0, 0.0], &teacher, 0);
        // Bad student: logits disagree
        let bad = dl.compute(&[0.0, 0.0, 3.0], &teacher, 0);
        assert!(bad > good);
    }

    #[test]
    fn test_ternarization_schedule() {
        let sched = TernarizationSchedule::new(100, 100);
        assert_eq!(sched.ternary_prob(0), 0.0);
        assert_eq!(sched.ternary_prob(50), 0.0); // still warmup
        assert!((sched.ternary_prob(150) - 0.5).abs() < 1e-10);
        assert_eq!(sched.ternary_prob(300), 1.0);
    }

    #[test]
    fn test_ternarize_high() {
        let sched = TernarizationSchedule::new(0, 0);
        assert_eq!(sched.ternarize(0.9, 100), 1);
    }

    #[test]
    fn test_ternarize_low() {
        let sched = TernarizationSchedule::new(0, 0);
        assert_eq!(sched.ternarize(-0.9, 100), -1);
    }

    #[test]
    fn test_ternarize_near_zero() {
        let sched = TernarizationSchedule::new(0, 0);
        assert_eq!(sched.ternarize(0.1, 100), 0);
    }

    #[test]
    fn test_tracker() {
        let mut t = DistillationTracker::new("student", "teacher");
        t.record_step(0.5, 0.3, 0.8);
        t.record_step(0.4, 0.2, 0.85);
        assert_eq!(t.steps, 2);
        assert!((t.best_accuracy - 0.85).abs() < 1e-10);
        assert!((t.avg_distill_loss() - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_compression_ratio() {
        let t = DistillationTracker::new("s", "t");
        assert_eq!(t.compression_ratio(1000, 100), 10.0);
    }
}
