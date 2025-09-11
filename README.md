# League-of-Legends-AI-Coach

Summary:

The jungle coach integrates multiple machine learning strategies into a unified decision system, combining supervised models for predictive accuracy, unsupervised methods for feature structuring and strategy priors, reinforcement and imitation learning for dynamic policy shaping, and meta-learning for adaptability. These components are grounded in League-specific strategy, balancing high-elo discipline with practical, safe decision-making for broader applicability.

Data Processing & Preparation:

Raw data was sourced from the Riot Games API, combining match metadata and timeline events into structured decision states. Each state vector included:

Game metrics: team gold and XP leads, objective timers, camp respawn times, etc.

Player metrics: champion HP/mana, cooldowns, summoner spell readiness, etc.

Map context: lane priority, champion proximity, vision/ward events, etc.

Timelines were segmented into fixed-length windows, producing state–action training samples. Features were scaled and normalized, with engineered metrics such as gold per minute, jungle tempo indices, and threat levels based on enemy vision.

To reduce dimensionality and improve generalization, PCA compressed high-dimensional vectors into compact state embeddings. Additionally, k-means clustering was applied to player and champion trajectories, producing playstyle archetypes (e.g., tempo ganker, power farmer) used as priors during action evaluation.

Machine Learning Methods:

The system integrated multiple ML paradigms:

Supervised Learning: Random Forest and XGBoost classifiers estimated probabilities of gank or objective success; Linear Regression, Elastic Net, and XGBoost regressors predicted changes in win probability, gold, or XP.

Unsupervised Learning: PCA for feature reduction and k-means for playstyle profiling enhanced interpretability and modulated risk.

Reinforcement Learning: A tabular Q-learning prototype and asynchronous actor–critic loop explored value-based and policy-gradient approaches for state–action optimization.

Imitation Learning: Behavioral cloning trained models directly on Challenger+ replays, replicating expert jungler decisions without explicit reward shaping.

Meta-Learning (Lightweight): Fast re-training pipelines on Masters+ subsets or single-champion datasets provided rapid adaptation across patches or role-specific contexts.

Decision Integration: Model predictions were combined with uncertainty penalties, risk flags, and hard guardrails (e.g., smite availability, lane priority, soul/elder timing). Final action selection maximized risk-adjusted utility, with cooldown timers preventing oscillation.

Rationale & Practicality in League Context:

Several design choices were guided by League-specific realities:

Training on high-elo data for generalization: Models were trained primarily on Masters+ and Challenger games, where strategies are more disciplined and consistent. This ensures cleaner decision patterns, even if the system is later applied in low-elo environments where opponents make more mistakes. The conservative bias (e.g., respecting lane prio before objectives) makes recommendations safe and generalizable.

Risk-aware action scoring: By embedding uncertainty penalties and risk flags, the system avoids overfitting to risky “coin-flip” plays. This reflects practical jungling priorities — low-risk consistency is often better than occasional high-risk success.

Guardrails tied to competitive strategy: Explicit rules around soul points, elder dragon, and baron windows mirror real-world competitive priorities, ensuring the AI respects critical win conditions.

Playstyle archetype clustering: Using unsupervised clustering to modulate action utilities reflects how different junglers balance farming versus ganking. This allows the system to mimic recognizable strategies, making outputs more intuitive to human players.

Meta-adaptation: League is patch-driven. Lightweight retraining on filtered data (e.g., champion-specific updates) provides fast adaptability without rebuilding the full pipeline.


Inspiration:

After watching Rank 1 Streamer Pentaless' climb from Iron to Challenger on Nunu, noted by the League of Legends Community as one of the weakest champions in the game, I realized that with precise and consistent decision-making based on high-level concepts, it would be possible for me, a hardstuck Iron player in the bottom 10% of players, to climb too.


Results:

Exclusively using the League Coach AI in live games I was able to achieve over a 70% over 60 games, climbing from Iron I to Gold IV, over 9 divisions and jumping from the bottom 10% of players in North America to the top 30%. During testing by friends in various ELOs (High: Masters to Challenger, Mid: Platinum to Diamond by Rank 1 Pentaless#NA, Low: Iron to Gold by DormantDreams#LLJ), win rates were 90%+ (Pentaless) and 100% (DormantDreams).

Training yielded accuracy above 80% compared to high ELO decisions when trained in over 300 games when using supervised learning and offline reinforcement learning models. This is a very good result as although Challenger decisions (being the best players in the world) should be a great resource, there is still a huge difference in accuracy/efficiency from players such as those in the top 10 and the bottom of Masters ("The difference between Rank 1 and bottom of Challenger is greater than the difference from bottom of Challenger to Iron" - Pentaless). There is thus potential for higher accuracy and win rate prediction power if the model is trained only on the top 10 players, but there is a serious chance of overfitting and seemingly illogical/inefficient (e.g. enemy-disruptive based) playstyles from that. Overfitting was generally mitigated with larger sample sizes across multiple team compositions and sub-ELO rank brackets. 

The 50% accuracy when using non-supervised reinforcement models was likely do to the fact that players with similar decision-making skills (hence their similar ELO bracket) would average out to 50% accuracy (when win-rate based) in practice, effectively making the AI useless. This, however, is likely only an issue when looking at high-ELO. Below are Model selection recommendations depending on common win conditions for different ELOs.


Recommended Model Borda Weightings:

In order to optimize AI Jungle Coach performance for your ELO, there are the following recommendations (also with the option to directly choose win-con independent of ELO)

Low Elo (Iron–Gold) → Farm-Heavy, Safe Scaling

Supervised models (RF/XGBoost classifiers, regressors):

Classifier predictions of gank success are often down-weighted with a higher risk penalty (γ).

Regressors estimating Δ win-probability for farming/objective plays are given more influence.

Unsupervised (k-means clusters):

Archetypes biased toward “power farming” clusters shift priors toward safe plays.

Reinforcement learning (Q-learning prototype):

Q-values from failed risky ganks reinforce conservative strategies.

Imitation learning (behavioral cloning):

Challenger replays normally favor proactive plays, but the system blends them with farm-safe heuristics for low elo.

Result: The action utility function heavily favors farming and uncontested objectives, while risky ganks are penalized unless the classifier confidence is extremely high.

Mid Elo (Platinum–Diamond) → Balanced Playstyle

Supervised models:

Classifier predictions for gank success and regressor estimates of farm value are balanced equally.

Neither farm nor aggression is dominant; utilities are more sensitive to lane prio and vision features.

Unsupervised (k-means):

Mixed clusters allow playstyle priors to vary per state, encouraging situational adaptation.

Reinforcement learning:

Actor–critic loop contributes exploratory policies, allowing the agent to occasionally attempt riskier plays.

Imitation learning:

Behavioral cloning is more directly applied, since mid elo players can partially replicate high-elo strategies.

Result: The coach alternates between farm and gank, depending on state features like lane prio, enemy jungler position, and vision control.

High Elo (Masters+) → Tempo-Aggressive, Proactive

Supervised models:

Classifier outputs for gank/objective success are trusted more heavily, with reduced risk penalties (γ).

Regression models reward tempo swings (early objectives, invades) more strongly.

Unsupervised (k-means):

Archetypes biased toward “tempo/aggressive” clusters amplify proactive plays.

Reinforcement learning:

Q-learning and actor–critic signals blend into utilities (via η), encouraging early aggression where rewards outweigh risks.

Imitation learning:

Challenger+ replays align naturally with this elo, so behavioral cloning outputs are used with minimal adjustment.

Result: The system actively selects proactive, tempo-based actions, but guardrails (prio + smite requirements, objective timers) keep aggression disciplined.

+-------------+----------------------+--------------------------------------------+
|    Elo      | Utility Adjustment   | ML Enforcement Mechanism                   |
+-------------+----------------------+--------------------------------------------+
| Iron–Gold   | Farm bias, higher γ  | - Down-weight gank classifiers             |
|             |                      | - Regressors favor farm/objectives         |
|             |                      | - Q-learning punishes failed aggression    |
|             |                      | - Playstyle clusters favor safe scaling    |
+-------------+----------------------+--------------------------------------------+
| Plat–Diamond| Balanced weights     | - Equal weighting of classifiers/regressors|
|             |                      | - Playstyle clusters vary situationally    |
|             |                      | - Actor–critic adds opportunistic plays    |
|             |                      | - Behavioral cloning applied more directly |
+-------------+----------------------+--------------------------------------------+
| Masters+    | Aggression bias, low γ| - Classifier outputs trusted more heavily  |
|             |                      | - Regression models emphasize tempo gains  |
|             |                      | - RL blended into utilities (η>0)          |
|             |                      | - Playstyle clusters favor tempo-aggressive|
|             |                      | - Cloning from Challenger replays aligns   |
+-------------+----------------------+--------------------------------------------+

Recommended Model Key Takeaways:

The same high-elo-trained models are used across all elos.

Utility weighting, cluster priors, and risk penalties are adjusted by elo context.

This ensures strategies are safe in low elo, adaptive in mid elo, and proactive in high elo, while still grounded in high-elo decision quality.


Future Direction:

LLMs may be used to ____.
