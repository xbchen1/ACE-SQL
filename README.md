# ACE-SQL Anonymous Repository

This anonymous release contains the training and evaluation code needed to reproduce the ACE-SQL supervised fine-tuning, reinforcement-learning, and benchmark evaluation stages.

## Layout

- `SFT/`: supervised fine-tuning cold start with LLaMA-Factory
- `RL/`: joint GRPO training with empirical retrieval targets and PCGrad
- `EVAL/`: BIRD and Spider evaluation framework with retrieval, SQL generation, and execution-evaluation pipelines

Model weights, benchmark databases, generated predictions, and local cluster files are not included. Put local models under each stage's `models/` directory and databases under `RL/external/` or `EVAL/data/`, or override the paths with environment variables described in the stage READMEs.

## Quick Start

```bash
cd SFT
bash scripts/train_sft.sh
```

```bash
cd RL
bash scripts/train_rl_pcgrad.sh
```

```bash
cd EVAL
bash eval_bird_maj.sh
```

Use `cd RL && make syntax` and `cd RL && make check-paths` for lightweight repository checks before running training.
