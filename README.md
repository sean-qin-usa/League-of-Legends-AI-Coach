# League-of-Legends-AI-Coach

## Summary
The jungle coach integrates multiple machine learning strategies into a unified decision system, combining supervised models for predictive accuracy, unsupervised methods for feature structuring and strategy priors, reinforcement and imitation learning for dynamic policy shaping, and meta-learning for adaptability. These components are grounded in League-specific strategy, balancing high-elo discipline with practical, safe decision-making for broader applicability.

**Additional detail:**  
The jungle coach treats gameplay as an **MDP** and unifies these paradigms with **ensemble arbitration** (stacking / weighted voting / **Borda count**) and **uncertainty-aware gating**. Decisions are bounded by League-specific **guardrails** (lane prio / smite / objective timing), emphasizing stability and sustained improvement.

---

## Data Processing & Preparation
Raw data was sourced from the Riot Games API, combining match metadata and timeline events into structured decision states. Each state vector included:

- **Game metrics:** team gold and XP leads, objective timers, camp respawn times, etc.  
- **Player metrics:** champion HP/mana, cooldowns, summoner spell readiness, etc.  
- **Map context:** lane priority, champion proximity, vision/ward events, etc.

Timelines were segmented into fixed-length windows, producing state–action training samples. Features were scaled and normalized, with engineered metrics such as gold per minute, jungle tempo indices, and threat levels based on enemy vision.

To reduce dimensionality and improve generalization, PCA compressed high-dimensional vectors into compact state embeddings. Additionally, k-means clustering was applied to player and champion trajectories, producing playstyle archetypes (e.g., tempo ganker, power farmer) used as priors during action evaluation.

**Additional detail:**  
- Features were scaled/normalized **with leakage guards** to prevent train/test contamination; engineered metrics included **tempo indices** and **vision-adjusted threat levels**.  
- Dimensionality reduction can use **PCA / manifold embeddings** (e.g., nonlinear projections) to form compact state embeddings.  
- Unsupervised profiling can incorporate **k-means / density-based clustering** to derive robust archetypes from trajectory data.

---

## Machine Learning Methods
The system integrated multiple ML paradigms:

- **Supervised Learning:** Random Forest and XGBoost classifiers estimated probabilities of gank or objective success; Linear Regression, Elastic Net, and XGBoost regressors predicted changes in win probability, gold, or XP.
- **Unsupervised Learning:** PCA for feature reduction and k-means for playstyle profiling enhanced interpretability and modulated risk.
- **Reinforcement Learning:** A tabular Q-learning prototype and asynchronous actor–critic loop explored value-based and policy-gradient approaches for state–action optimization.
- **Imitation Learning:** Behavioral cloning trained models directly on Challenger+ replays, replicating expert jungler decisions without explicit reward shaping.
- **Meta-Learning (Lightweight):** Fast re-training pipelines on Masters+ subsets or single-champion datasets provided rapid adaptation across patches or role-specific contexts.
- **Decision Integration:** Model predictions were combined with uncertainty penalties, risk flags, and hard guardrails (e.g., smite availability, lane priority, soul/elder timing). Final action selection maximized risk-adjusted utility, with cooldown timers preventing oscillation.

**Additional detail:**  
- Supervised probabilities can be **calibrated** (e.g., **Platt scaling / isotonic regression**) to improve thresholding.  
- RL exploration can use **ε-greedy / UCB** under the same safety constraints, and actor–critic variants complement tabular Q-learning.  
- Decision integration aggregates model opinions via **stacking / weighted voting / Borda count**, then applies uncertainty and risk penalties before guardrails.

---

## Rationale & Practicality in League Context
Several design choices were guided by League-specific realities:

- **Training on high-elo data for generalization:** Models were trained primarily on Masters+ and Challenger games, where strategies are more disciplined and consistent. This ensures cleaner decision patterns, even if the system is later applied in low-elo environments where opponents make more mistakes. The conservative bias (e.g., respecting lane prio before objectives) makes recommendations safe and generalizable.
- **Risk-aware action scoring:** By embedding uncertainty penalties and risk flags, the system avoids overfitting to risky “coin-flip” plays. This reflects practical jungling priorities — low-risk consistency is often better than occasional high-risk success.
- **Guardrails tied to competitive strategy:** Explicit rules around soul points, elder dragon, and baron windows mirror real-world competitive priorities, ensuring the AI respects critical win conditions.
- **Playstyle archetype clustering:** Using unsupervised clustering to modulate action utilities reflects how different junglers balance farming versus ganking. This allows the system to mimic recognizable strategies, making outputs more intuitive to human players.
- **Meta-adaptation:** League is patch-driven. Lightweight retraining on filtered data (e.g., champion-specific updates) provides fast adaptability without rebuilding the full pipeline.

**Additional detail:**  
Unsupervised priors help determine whether a state favors **farm-scaling** or **tempo aggression**, improving interpretability and aligning outputs with recognizable human strategies.

---

## Inspiration
After watching Rank 1 Streamer Pentaless climb from Iron to Challenger on Nunu, noted by the League of Legends Community as one of the weakest champions in the game, I realized that with precise and consistent decision-making based on high-level concepts, it would be possible for me, a hardstuck Iron player in the bottom 10% of players, to climb too.

---

## Results and Observations
Exclusively using the League Coach AI in live games I was able to achieve over a 70% win rate over 60 games, climbing from Iron I to Gold IV, over 9 divisions and jumping from the bottom 10% of players in North America to the top 30%.

During testing by friends in various ELOs:

- **High elo:** Masters → Challenger (by Pentaless#NA) → 75%+ win rate  
- **Mid elo:** Platinum → Diamond (by Pentaless#NA) → 90%+ win rate  
- **Low elo:** Iron → Gold (by DormantDreams#LLJ) → 100% win rate

Training yielded accuracy above 80% compared to high-elo decisions when trained on 300+ games using supervised learning and offline RL models. Although Challenger decisions represent the best players, there is still a major skill gap between the top 10 and bottom of Challenger. Overfitting risks were mitigated with larger samples across multiple comps and rank brackets.

By contrast, non-supervised reinforcement models averaged ~50% accuracy, reflecting the symmetry of decision-making within the same elo (win rate ~50% baseline). This limited standalone RL usefulness in high-elo testing.

**Additional detail:**  
Standalone RL’s ~50% accuracy indicates RL is most effective as a **blended signal** in the ensemble, rather than a solo driver.

---

## Recommended Model Borda Weightings
To optimize Jungle Coach performance per elo, the following adjustmentsadjustments, toggleable within cells, are recommended (with the option to choose win condition directly).

### Low Elo (Iron–Gold) → Farm-Heavy, Safe Scaling
- **Supervised:** Down-weight gank classifiers with higher γ; regressors for farm/objectives more influential.  
- **Unsupervised:** Power-farming clusters shift priors to safe plays.  
- **Reinforcement Learning:** Q-values from failed ganks reinforce conservative bias.  
- **Imitation Learning:** Challenger replay bias toward aggression softened with safe heuristics.  

**Result:** Strong farm/objective bias; risky ganks discouraged unless confidence is very high.

---

### Mid Elo (Platinum–Diamond) → Balanced Playstyle
- **Supervised:** Balanced weighting of classifiers vs regressors.  
- **Unsupervised:** Mixed clusters enable situational flexibility.  
- **Reinforcement Learning:** Actor–critic loop allows exploratory, opportunistic plays.  
- **Imitation Learning:** Directly applied; mid elo players can partially replicate high-elo strategies.  

**Result:** Alternates between farm and gank depending on prio, vision, and jungler tracking.

---

### High Elo (Masters+) → Tempo-Aggressive, Proactive
- **Supervised:** Classifier outputs trusted more heavily; regressors emphasize tempo.  
- **Unsupervised:** Aggressive clusters amplify proactive plays.  
- **Reinforcement Learning:** Q-learning and actor–critic blended (η > 0).  
- **Imitation Learning:** Challenger+ replay cloning used directly.  

**Result:** Proactive tempo strategy; aggression disciplined by prio/smite guardrails.

---

## Elo → Utility Adjustment → ML Enforcement
Elo	Utility Adjustment	ML Enforcement Mechanism  
Iron–Gold	Farm bias, higher γ	- Down-weight gank classifiers  
- Regressors favor farm/objectives  
- Q-learning punishes failed aggression  
- Playstyle clusters favor safe scaling  
Plat–Diamond	Balanced weights	- Equal weighting of classifiers/regressors  
- Playstyle clusters vary situationally  
- Actor–critic adds opportunistic plays  
- Behavioral cloning applied more directly  
Masters+	Aggression bias, low γ	- Classifier outputs trusted more heavily  
- Regression models emphasize tempo gains  
- RL blended into utilities (η > 0)  
- Playstyle clusters favor tempo-aggressive  
- Behavioral cloning aligns with Challenger replays

| Elo         | Utility Adjustment     | ML Enforcement Mechanism                                                                 |
|-------------|------------------------|------------------------------------------------------------------------------------------|
| Iron–Gold   | Farm bias, higher γ    | - Down-weight gank classifiers <br> - Regressors favor farm/objectives <br> - Q-learning punishes failed aggression <br> - Playstyle clusters favor safe scaling |
| Plat–Diamond| Balanced weights       | - Equal weighting of classifiers/regressors <br> - Playstyle clusters vary situationally <br> - Actor–critic adds opportunistic plays <br> - Behavioral cloning applied more directly |
| Masters+    | Aggression bias, low γ | - Classifier outputs trusted more heavily <br> - Regression models emphasize tempo gains <br> - RL blended into utilities (η > 0) <br> - Playstyle clusters favor tempo-aggressive <br> - Behavioral cloning aligns with Challenger replays |

---

## Recommended Model Key Takeaways
The same high-elo–trained models are used across all elos.  
Elo-specific strategies arise from utility weighting, priors, and penalties, not separate models.  
This ensures strategies are safe in low elo, adaptive in mid elo, and proactive in high elo, while staying grounded in high-elo decision quality.

---

## Future Direction
Future improvements may focus on integrating large language models (LLMs) to overcome current limitations in rigidity and interpretability. LLMs can infer win conditions from drafts, adapt strategies to patch notes and meta shifts, and process multimodal inputs such as VODs or minimaps without hand-crafted features. Most importantly, they can provide elo-specific, natural language explanations, turning the coach from a decision engine into an interactive teaching assistant that adapts strategies and communication to the player’s level.
