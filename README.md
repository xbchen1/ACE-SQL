# ACE-SQL

[![arXiv](https://img.shields.io/badge/arXiv-2606.05906-b31b1b.svg)](https://arxiv.org/abs/2606.05906)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/xiaobing11/ACE-SQL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Adaptive Co-Optimization via Empirical Credit Assignment for Text-to-SQL**

ACE-SQL trains schema retrieval and SQL generation as two roles of one shared language model. Execution-correct SQL rollouts supply adaptive column-set targets for the retriever, while the retriever's selections define the generator's schema context.

This repository follows the **October 9, 2026 manuscript revision**. The current training entrypoint uses **slime**, SGLang, and Megatron with direct joint GRPO. See [paper-to-code alignment](docs/paper_alignment.md) for the exact source fingerprints and validation scope. The linked arXiv preprint and dataset release may precede this revision.

![ACE-SQL overview](assets/ace_sql_overview.png)

## Results reported in the manuscript

Greedy execution accuracy (EX, %); RL rows are means over four seeds.

| Model | BIRD Dev | Spider Dev | Spider Test |
| --- | ---: | ---: | ---: |
| ACE-SQL / Qwen3-4B | 64.7 | 82.6 | 85.9 |
| ACE-SQL / Qwen3-8B | 66.3 | 84.0 | 87.5 |

The 8B model uses **0.87k output tokens per query**, summed across both roles. Training uses **14,184 SFT samples** (7,092 per role) and **2,913 RL questions**. These are manuscript-reported measurements; this repository update did not rerun those experiments. Model weights and per-seed prediction files are not bundled.

## Method

1. **SFT cold start.** Mix model-generated retrieval and execution-filtered SQL samples in one full-parameter fine-tuning run.
2. **Two-pass rollout.** Sample eight retriever outputs per question, take their **union**, and sample eight SQL outputs against that shared pruned schema.
3. **Online empirical target.** Count column sets from execution-matched SQL, update `pool = 0.5 * historical_pool + current_counts`, then read the most frequent set. All tied maximum-frequency sets are accepted.
4. **Sparse rewards.** Give the generator execution-match reward and the retriever exact-target reward. Only matched outputs receive the length penalty: zero up to 512 tokens, increasing linearly to 0.5 at 2,048 tokens.
5. **Joint update.** Normalize advantages separately for each question and role. Combine the two role losses as `(1 - lambda/2) * L_ret + (lambda/2) * L_gen`, with `lambda` rising from 0 to 1 over the first 25% of RL steps. Apply one optimizer update to the shared policy. No gradient projection is used.

Evaluation uses one greedy retrieval followed by one greedy SQL generation. Evaluation does not update the pool.

## Repository guide

| Directory | Contents |
| --- | --- |
| [SFT](SFT/README.md) | LLaMA-Factory configurations, launch script, and both SFT datasets |
| [RL](RL/README.md) | Current slime training adapter, launchers, rewards, tests, and RL data |
| [EVAL](EVAL/README.md) | BIRD, Spider, and Spider robustness evaluation |
| [docs](docs/paper_alignment.md) | Method mapping, data manifest, and release limitations |
| [audits](audits/README.md) | Historical overlap audit, rerunnable code, and input fingerprints |

The older `RL/trainer/`, `RL/third_party/verl/`, and `train_rl_pcgrad.sh` are retained for legacy compatibility. They are not the current paper recipe and are not invoked by `train_rl.sh`.

## Run

Use separate environments for SFT, RL, and evaluation. SQLite databases and checkpoints must be supplied separately; [data documentation](docs/data.md) describes the included files.

SFT (default: eight GPUs, one epoch):

```bash
cd SFT
bash scripts/train_sft.sh
```

RL configuration preview (no GPU or model load):

```bash
bash RL/scripts/train_rl.sh --dry-run
```

For training, follow the [slime setup and checkpoint conversion guide](RL/README.md) and the [pinned H20/CUDA environment](RL/configs/slime-h20-compatibility.md). The paper recipe uses eight H20 GPUs, five epochs, a learning rate of `1e-6`, group size eight per role, and KL coefficient `0.001`.

BIRD greedy evaluation of an exported shared-policy checkpoint:

```bash
cd EVAL
MODEL_PATH=/path/to/checkpoint SQL_MODEL=/path/to/checkpoint \
BIRD_DEV_DATA=/path/to/bird/dev.json \
BIRD_TABLES=/path/to/bird/dev_tables.json \
BIRD_DB_PATH=/path/to/bird/dev_databases \
bash eval_bird_maj.sh
```

See [EVAL](EVAL/README.md) for Spider and robustness commands. BIRD uses its benchmark set-based execution comparison; training uses ordered row/column comparison. Training SQL timeout is 30 seconds; BIRD evaluation defaults to 180 seconds.

## Verification

```bash
python -m pip install -r RL/configs/requirements-test.txt
make -C RL syntax
make -C RL test
python scripts/verify_release.py
```

CPU tests exercise actual SQLite execution, pool updates, role-separated advantages, weighted PPO/KL loss, validation isolation, and checkpoint recovery with mocked model responses. Optional integration tests inspect the pinned slime source. **GPU training, distributed weight synchronization, and the reported benchmark scores have not been revalidated by this release update.**

## License and dependencies

Project code is under the [MIT License](LICENSE); bundled third-party code retains its own license. Dataset and base-model use is subject to the original sources' terms.

Built with [Qwen3](https://github.com/QwenLM/Qwen3), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [slime](https://github.com/THUDM/slime), [SGLang](https://github.com/sgl-project/sglang), and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Evaluation uses [vLLM](https://github.com/vllm-project/vllm) and the [BIRD](https://bird-bench.github.io/) and [Spider](https://yale-lily.github.io/spider) benchmarks.
