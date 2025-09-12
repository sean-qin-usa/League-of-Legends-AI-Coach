# League-of-Legends-AI-Coach

## Summary
The jungle coach treats gameplay as an MDP and unifies multiple learning paradigms into a single decision system: supervised models for calibrated predictions, unsupervised methods for feature structuring and playstyle priors, reinforcement and imitation learning for dynamic policy shaping, and meta-learning for adaptability. Final choices use ensemble arbitration (stacking / weighted voting / Borda count) with uncertainty-aware gating and League-specific guardrails (prio/smite/objective timing), balancing high-elo discipline with practical, safe decision-making for broader applicability and sustained improvement.

---

## Data Processing & Preparation
Raw data was sourced from the Riot Games API, combining match metadata and timeline events into structured decision states. Each state vector included:

- **Game metrics:** team gold and XP leads, objective timers, camp respawn times, tempo indices, etc.  
- **Player metrics:** champion HP/mana, cooldowns, summoner spell readiness, resource checks, etc.  
- **Map context:** lane priority, champion proximity, vision/ward events, jungler tracking, etc.  

Timelines were segmented into fixed-length windows, producing state–action training samples. Features were scaled and normalized with leakage guards, with engineered metrics such as gold per minute, jungle tempo indices, and vision-adjusted threat levels.  

To reduce dimensionality and improve generalization, **PCA / manifold embeddings** compressed high-dimensional vectors into compact state embeddings. Additionally, **k-means / density-based clustering** was applied to player and champion trajectories, producing playstyle archetypes (e.g., tempo ganker, power farmer) used as priors during action evaluation.

---

## Machine Learning Methods
The system integrated multiple ML paradigms:

- **Supervised Learning:** Random Forest and XGBoost classifiers estimated probabilities of gank or objective success; Linear/Elastic Net and XGBoost regressors predicted changes in win probability, gold, or XP, with probability calibration (e.g., Platt/isotonic).  
- **Unsupervised Learning:** PCA/manifold embeddings for feature reduction and k-means for playstyle profiling enhanced interpretability and modulated risk via priors.  
- **Reinforcement Learning:** Tabular Q-learning prototypes and actor–critic variants explored value-based and policy-gradient approaches; exploration used ε-greedy/UCB under safety constraints.  
- **Imitation Learning:** Behavioral cloning trained models directly on Challenger+ replays, replicating expert jungler decisions without heavy reward shaping.  
- **Meta-Learning (Lightweight):** Fast re-training pipelines on Masters+ subsets or single-champion datasets provided rapid adaptation across patches or role-specific contexts.  
- **Decision Integration:** Model opinions were aggregated via stacking / weighted voting / Borda count; uncertainty penalties, risk flags, cooldown timers, and hard guardrails (e.g., smite availability, lane priority, soul/elder/baron windows) gated actions to maximize risk-adjusted utility and prevent oscillation.  

---

## Rationale & Practicality in League Context
Several design choices were guided by League-specific realities:

- **Training on high-elo data for generalization:** Models were trained primarily on Masters+ and Challenger games, where strategies are disciplined and consistent. This yields cleaner decision patterns, even when applied in low-elo environments. The conservative bias (e.g., respecting lane prio before objectives) makes recommendations safe and generalizable.  
- **Risk-aware action scoring:** Embedding uncertainty penalties and risk flags avoids overfitting to risky “coin-flip” plays. In practice, low-risk consistency beats occasional high-risk success.  
- **Guardrails tied to competitive strategy:** Explicit rules around soul points, elder dragon, and baron windows mirror real-world priorities, ensuring the AI respects critical win conditions.  
- **Playstyle archetype clustering:** Unsupervised priors modulate utilities based on whether a situation favors farm-scaling or tempo aggression, making outputs more intuitive to human players.  
- **Meta-adaptation:** League is patch-driven. Lightweight retraining on filtered data (e.g., champion-specific updates) enables fast adaptability without rebuilding the whole pipeline.  

---

## Inspiration
After watching Rank 1 Streamer **Pentaless** climb from Iron to Challenger on Nunu, considered by the League Community to be one of the weakest champions in the game, I realized that precise, consistent decisions grounded in high-level concepts could carry even a hardstuck Iron player (bottom 10%) to meaningful improvement.

---

## Results
Exclusively using the League Coach AI in live games, I achieved a **70%+ win rate over 60 games**, climbing from **Iron I → Gold IV**, jumping from the bottom 10% of players in North America to roughly the top 30%.  

During testing by friends in various ELOs:

- **High elo:** Masters → Challenger (by Pentaless#NA) → **75%+ win rate**  
- **Mid elo:** Platinum → Diamond (by Pentaless#NA) → **90%+ win rate**  
- **Low elo:** Iron → Gold (by DormantDreams#LLJ) → **100% win rate**  

Training yielded **accuracy above 80%** versus high-elo decisions on 300+ games using supervised + offline RL models, with ablations across comps and rank brackets to mitigate overfitting.  

By contrast, **standalone RL baselines averaged ~50% accuracy** (mirroring symmetric match outcomes), making RL most useful as a blended signal inside the ensemble rather than a solo driver.

---

## Recommended Model Borda Weightings
To optimize Jungle Coach performance per elo, the following adjustments—toggleable within cells—are recommended (with the option to choose win condition directly).

### Low Elo (Iron–Gold) → Farm-Heavy, Safe Scaling
- **Supervised:** Down-weight aggressive gank classifiers; emphasize farm/objective regressors.  
- **Unsupervised:** Power-farming clusters shift priors to safe plays.  
- **Reinforcement Learning:** Q-values from failed ganks reinforce conservative bias.  
- **Imitation Learning:** Challenger replay aggression softened with safety heuristics.  

**Result:** Strong farm/objective bias; risky ganks discouraged unless confidence is very high.  

---

### Mid Elo (Platinum–Diamond) → Balanced Playstyle
- **Supervised:** Balanced weighting of classifiers vs. regressors.  
- **Unsupervised:** Mixed clusters enable situational flexibility.  
- **Reinforcement Learning:** Actor–critic loop allows opportunistic tempo plays.  
- **Imitation Learning:** High-elo patterns applied more directly.  

**Result:** Alternates between farm and gank depending on prio, vision, and jungler tracking.  

---

### High Elo (Masters+) → Tempo-Aggressive, Proactive
- **Supervised:** Classifier confidence trusted more; regressors emphasize tempo swings.  
- **Unsupervised:** Aggressive clusters amplify proactive plays.  
- **Reinforcement Learning:** Q-learning and actor–critic blended (η > 0).  
- **Imitation Learning:** Challenger+ replay cloning used directly.  

**Result:** Proactive tempo strategy; aggression disciplined by prio/smite guardrails.  

---

### Elo → Utility Adjustment → ML Enforcement

| Elo         | Utility Adjustment       | ML Enforcement Mechanism                                                                 |
|-------------|--------------------------|------------------------------------------------------------------------------------------|
| Iron–Gold   | Farm bias, higher γ      | - Down-weight gank classifiers <br> - Regressors favor farm/objectives <br> - Q-learning punishes failed aggression <br> - Playstyle clusters favor safe scaling |
| Plat–Diamond| Balanced weights         | - Equal weighting of classifiers/regressors <br> - Playstyle clusters vary situationally <br> - Actor–critic adds opportunistic plays <br> - Behavioral cloning applied more directly |
| Masters+    | Aggression bias, low γ   | - Classifier outputs trusted more heavily <br> - Regression models emphasize tempo gains <br> - RL blended into utilities (η > 0) <br> - Playstyle clusters favor tempo-aggressive <br> - Behavioral cloning aligns with Challenger replays |

---

### Recommended Model Key Takeaways
- The same **high-elo–trained models** are used across all elos.  
- Elo-specific strategies arise from **utility weighting, priors, and penalties**, not separate models.  
- This yields strategies that are **safe in low elo, adaptive in mid elo, and proactive in high elo**, while staying grounded in disciplined, high-elo decision quality.  

---

## Future Direction
Future improvements may focus on integrating **large language models (LLMs)** to reduce rigidity and improve interpretability. LLMs can infer win conditions from drafts, adapt strategies to patch notes and meta shifts, and process multimodal inputs such as VODs or minimaps without hand-crafted features. Most importantly, they can provide **elo-specific, natural language explanations**, turning the coach from a decision engine into an **interactive teaching assistant** that adapts strategy and communication to the player’s level.
