# League-of-Legends-AI-Coach

A real-time jungle coach that treats the game as an MDP and combines several learning paradigms in one decision system: supervised models for calibrated predictions, unsupervised methods for feature structure and playstyle priors, offline reinforcement and imitation learning for policy shaping, and lightweight meta-learning for patch adaptation. Final actions come from ensemble arbitration (stacking, weighted voting, or Borda count) gated by uncertainty penalties and League-specific guardrails such as lane priority, smite availability, and objective timing.

## Data

Raw data comes from the Riot Games API, with match metadata and timeline events combined into structured decision states. Each state vector includes game metrics (team gold and XP leads, objective timers, camp respawns, tempo indices), player metrics (HP/mana, cooldowns, summoner spells, resources), map context (lane priority, champion proximity, vision and ward events, jungler tracking), and composition cues (champion tag counts such as split-push, poke, engage).

Timelines are segmented into fixed-length windows to produce state-action training samples. Features are scaled and normalized with leakage guards, and engineered metrics include gold per minute, jungle tempo indices, and vision-adjusted threat levels. PCA and manifold embeddings compress the high-dimensional vectors into compact state embeddings, and k-means and density-based clustering over player and champion trajectories produce playstyle archetypes (tempo ganker, power farmer, and so on) that serve as priors during action evaluation.

## Models

- Supervised: XGBoost classifiers estimate gank and objective success probabilities; linear/elastic-net and XGBoost regressors predict changes in win probability, gold, and XP, with Platt or isotonic calibration and cross-validation for thresholding.
- Unsupervised: the PCA/manifold embeddings and clustering above, used both for dimensionality reduction and for risk-modulating playstyle priors.
- Reinforcement learning (offline/batch): tabular Q-learning prototypes and actor-critic variants, with ε-greedy/UCB exploration under safety constraints.
- Imitation learning: behavioral cloning on Challenger+ replays, replicating expert jungler decisions without heavy reward shaping.
- Meta-learning: fast retraining pipelines on Masters+ subsets or single-champion datasets for adaptation across patches and roles.
- Ensemble integration: model opinions are aggregated by stacking, weighted voting, or Borda count; uncertainty penalties, risk flags, cooldown timers, and hard guardrails (smite availability, lane priority, soul/elder/baron windows) gate the final action and prevent oscillation.

## Strategy priors

Priors are pre-model weights over action types (gank, farm, invade, objective, ward/clear) that encode which plans a composition and context should prefer before exact outcomes are evaluated. They come from the playstyle archetypes, from champion composition tags mapped to win conditions (teamfight, split, pick, poke/siege, snowball, objective/soul), from an elo profile that tilts safe → balanced → proactive, and optionally from fixed manual profiles.

In scoring, priors modulate utilities and tie-breaks without bypassing the safety rules:

`U(a) = (EV(a) × α_prior(a)) − λ·uncertainty − γ·risk`

Plan-consistent actions are boosted and low-fit actions lightly down-weighted. A split-push comp steers toward Herald, plates, and side vision; a teamfight comp toward grouping and objectives; a pick comp toward vision traps, roams, and siege angles.

## Win-condition inference

The system infers a high-level win condition per game and phase: teamfight scaling, pick/skirmish, 1-3-1 split, siege/poke, early snowball, or objective stack. Signals include draft and composition tags, role synergies, lane priority and wave states, objective timers, item spikes, vision control, and the archetype priors. The inferred condition scales utilities (`U_win(a) = U(a) × α_win(a | comp, state)`), boosts or filters candidate actions (side-lane pressure for 1-3-1, early dragons for objective stack, vision traps for pick), adapts guardrails (stricter prio/smite checks for teamfight comps, side-vision requirements for split, lower tolerance for coin-flip invades on scaling comps), and breaks ties toward actions that advance the plan.

## Design rationale

Models are trained primarily on Masters+ and Challenger games, where strategy is disciplined and consistent; the resulting conservative bias (respecting lane priority before objectives, for example) keeps recommendations safe even in low-elo games. Uncertainty penalties and risk flags in the action scoring keep the coach away from coin-flip plays, since low-risk consistency beats occasional high-risk success. Hard rules around soul point, elder, and baron windows mirror how the game is actually decided. Playstyle clustering makes outputs more intuitive by matching the recommendation to whether the situation favors farm-scaling or tempo aggression. And because League is patch-driven, the lightweight retraining path (champion-specific or Masters+ subsets) adapts the system without rebuilding the pipeline.

## Inspiration

After watching a Rank 1 Challenger streamer climb from Iron to Challenger on Nunu, long considered a weak champion, I realized that precise, consistent decisions grounded in high-level concepts could carry even a hardstuck Iron player to real improvement.

> "The difference between Rank 1 and bottom of Challenger is greater than the difference from bottom of Challenger to Iron." — a Rank 1 Challenger streamer

## Results

Using the coach exclusively in my own live games, I went 70%+ win rate over 60 games, climbing Iron I → Gold IV (9 divisions, roughly bottom 10% to top 30% of NA players). In testing by friends: a Challenger-level player took an account Masters → Challenger at a 75%+ win rate and another account Platinum → Diamond at 90%+, and an Iron-level player went Iron → Gold at 100%.

Supervised and offline-RL models trained on 300+ games reached over 80% accuracy against high-elo decisions, with ablations across comps and rank brackets to check for overfitting. Non-supervised reinforcement models averaged about 50% accuracy (mirroring symmetric match outcomes), which is why RL enters the ensemble as a blended signal rather than a solo driver.

## Elo-specific weightings

The same high-elo-trained models are used at every elo; what changes is the utility weighting, priors, and penalties. The presets (toggleable per cell, with the option to pick a win condition directly):

| Elo | Utility adjustment | Enforcement |
|---|---|---|
| Iron–Gold | farm bias, higher γ | gank classifiers down-weighted; regressors favor farm/objectives; Q-learning punishes failed aggression; clusters favor safe scaling |
| Plat–Diamond | balanced weights | equal classifier/regressor weighting; clusters vary situationally; actor-critic adds opportunistic plays; cloning applied more directly |
| Masters+ | aggression bias, lower γ | classifier outputs trusted more; regressors emphasize tempo gains; RL blended into utilities (η > 0); clusters favor tempo aggression; cloning aligns with Challenger replays |

The net effect is a coach that plays safe scaling in low elo, alternates between farm and gank in mid elo depending on prio, vision, and jungler tracking, and plays proactive tempo in high elo with aggression still disciplined by the prio/smite guardrails.

## Future direction

The main planned improvement is integrating LLMs to reduce rigidity and improve interpretability: inferring win conditions from drafts, adapting to patch notes and meta shifts, processing VODs or minimap frames without hand-crafted features, and above all giving elo-appropriate natural-language explanations, turning the coach from a decision engine into an interactive teaching assistant.
