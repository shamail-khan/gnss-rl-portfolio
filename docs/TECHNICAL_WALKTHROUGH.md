# Technical walkthrough

## Research framing

GNSS interference localization is treated as active information gathering. The policy chooses navigation actions using RF measurements rather than directly observing the unknown source position. Reward and success calculations use position information available to the research environment.

## Environments

- [jammer_env_A.py](../src/classes/jammer_env_A.py): dataset-backed 3D environment, nearest-neighbor lookup, 24 discrete actions, and configurable spatial success tolerances. Its spectrogram observation shape is (4, 32, 32).
- [jammer_env_B.py](../src/classes/jammer_env_B.py): structured ring navigation with radial, angular, and vertical actions, walls, and doorways.
- [register_env_A.py](../src/classes/register_env_A.py) and [register_env_B.py](../src/classes/register_env_B.py): archived integration with the original dataset loader and package layout. These require restoration before execution.

## Agent implementations

| Files in src/agents/ | Role |
| --- | --- |
| dqn_cnn_spec.py, dqn_cnn_iq.py | DQN variants for different RF representations |
| drqn_iq.py | Recurrent Q-learning |
| ppo_ccn_spec.py, ppo_iq.py | Feedforward PPO variants; the original ccn filename is preserved |
| ppo_lstm_iq.py | Recurrent PPO |
| ppo_atari_lstm_walls.py | Recurrent PPO wall experiments; retained CleanRL naming does not imply an Atari application |
| ppo_lstm_fomaml.py | Recurrent PPO with heads-only inner adaptation and first-order meta-updates |

The meta-learning implementation separates inner adaptation from query rollouts and accumulates first-order gradients for the outer update. The reported study uses wall configurations 0-2 for training and holds out configurations 3-4.

## Technical contribution represented

The artifacts document formulation of a measurement-backed navigation task, integration of deep RL agents with custom environments, recurrent observation processing, a first-order adaptation experiment, and analysis of failure under domain shift. The task-specific investigation and integration build on established methods and credited upstream code.

Research code is preserved for inspection. Publication preparation did not fix algorithmic or runtime issues, recreate missing evaluation scripts, or invent experiments.
