//! # Skill Library & Trajectory Memory
//!
//! Stores successful (state $\to$ action $\to$ result) plans with cosine similarity retrieval.

/// A single step in a stored skill trajectory.
#[derive(Debug, Clone)]
pub struct SkillStep {
    pub state_embedding: Vec<f64>,
    pub action: Vec<f64>,
    pub reward: f64,
    pub description: Option<String>,
}

/// A composite skill composed of sequential transitions achieving a goal.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub goal_embedding: Vec<f64>,
    pub steps: Vec<SkillStep>,
    pub cumulative_reward: f64,
    pub success_count: usize,
}

/// Skill Library repository.
#[derive(Debug, Clone, Default)]
pub struct SkillLibrary {
    pub skills: Vec<Skill>,
}

impl SkillLibrary {
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Registers or updates a skill in the library.
    pub fn insert(&mut self, skill: Skill) {
        if let Some(existing) = self.skills.iter_mut().find(|s| s.name == skill.name) {
            existing.steps = skill.steps;
            existing.cumulative_reward = skill.cumulative_reward;
            existing.success_count += 1;
        } else {
            self.skills.push(skill);
        }
    }

    /// Finds the closest matching skill using cosine similarity against `query_state_or_goal`.
    pub fn search(&self, query_embedding: &[f64], min_similarity: f64) -> Option<&Skill> {
        let mut best_skill: Option<&Skill> = None;
        let mut best_sim = min_similarity;

        for skill in &self.skills {
            let sim = cosine_sim(query_embedding, &skill.goal_embedding);
            if sim > best_sim {
                best_sim = sim;
                best_skill = Some(skill);
            }
        }

        best_skill
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_library_search() {
        let mut lib = SkillLibrary::new();
        let skill = Skill {
            name: "open_notepad".into(),
            goal_embedding: vec![1.0, 0.0, 0.0],
            steps: vec![SkillStep {
                state_embedding: vec![1.0, 0.0, 0.0],
                action: vec![100.0, 200.0],
                reward: 1.0,
                description: Some("Click notepad icon".into()),
            }],
            cumulative_reward: 1.0,
            success_count: 1,
        };
        lib.insert(skill);

        let query = vec![0.9, 0.1, 0.0];
        let found = lib.search(&query, 0.7);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "open_notepad");
    }
}
