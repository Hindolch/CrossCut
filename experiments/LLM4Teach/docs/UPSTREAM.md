# Official LLM4Teach origin

Source: https://github.com/ZJLAB-AMMI/LLM4Teach

The checkout already present in this workspace was cloned from that official
repository. The pristine tracked source from commit
`d8b537d0ea1e5d5437c9dce799d84a5f2239e90c` is retained in `../upstream/`.
It is reference source, not an installed dependency or runnable Crafter condition.
Its original author headers and README are preserved. No upstream license file
was present in this snapshot; this adaptation does not assign it a new license.

`../ppo.py` adapts `upstream/algos/ppo.py`, retaining its clipped policy surrogate,
clipped value objective, entropy regularization, and soft teacher supervision.
The adaptation replaces iteration-based kickstarting with the specification's
per-environment-step coefficient and temperature-scaled forward KL; makes the
recurrent rollout and episode masks explicit; and removes MiniGrid worker/option
interfaces. KL differs from soft cross-entropy by a frozen-teacher constant.

The pixel CNN–GRU (`student.py`), Crafter collection/evaluation (`train.py`,
`evaluate.py`), and Gemma candidate scorer (`teacher.py`) replace the original
MiniGrid model, skill controllers, symbolic mediator and teacher API. The
original algorithms are available for a direct source comparison.

The entire previous working checkout, including modified `planner.py`, untracked
Jarvis notes, and its `.git` directory, was preserved locally at
`experiments/LLM4Teach-legacy-local/`. That backup is ignored by the parent
repository and is not transferred to GitHub. The active experiment is tracked
as ordinary files in the parent repository, not as an embedded Git repository.

Paper: Zhou et al., *Large Language Model as a Policy Teacher for Training
Reinforcement Learning Agents*, IJCAI 2024, https://arxiv.org/abs/2311.13373.
