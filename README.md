# League-of-Legends-AI-Coach

Summary:

The jungle coach integrates supervised learning, unsupervised learning, reinforcement learning, imitation learning, and lightweight meta-learning into a unified decision system.

Supervised learning is central to action evaluation. Random Forest and XGBoost classifiers estimate gank or objective success probabilities, while Linear Regression, Elastic Net, and XGBoost regressors predict expected value shifts (e.g., changes in win probability, gold or XP advantages).

Unsupervised learning supports feature engineering and strategic priors. PCA is used for dimensionality reduction of timeline features, while k-means clustering identifies playstyle archetypes that modulate action preferences.

Reinforcement learning prototypes extend the system beyond static supervision. A tabular Q-learning head updates state–action values from short-horizon rewards, and an asynchronous actor–critic loop explores policy optimization. These signals are blended with supervised utilities during training.

Imitation learning is achieved through behavioral cloning, directly training state→action mappings from Challenger+ replays to replicate expert jungler decisions.

Meta-learning is approximated through rapid retraining on filtered subsets (e.g., Masters+ only or single-champion datasets), enabling fast adaptation across patches or champions without redesigning the pipeline.

The decision process begins by constructing a feature vector for the current state, pruning infeasible actions, and then scoring each candidate action with the supervised models. Expected values are adjusted for opportunity costs, uncertainty penalties, and explicit risk flags. Hard guardrails enforce constraints such as lane priority requirements, smite availability, and objective-specific timing (soul, elder, baron). The final action is chosen as the utility-maximizing option, subject to cooldowns and tempo-based tie-breakers.

This architecture ensures that decisions are model-guided, risk-aware, and rule-safe, combining statistical predictions with reinforcement-style learning and expert imitation.



Inspiration:



Results:



Future Direction:

