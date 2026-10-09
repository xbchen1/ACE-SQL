# H20 / CUDA 12.9 compatibility contract

The deployment uses the official **slimerl/slime:v0.2.3** image for Linux amd64,
pinned to `sha256:6e7748eb401c9bfbf99ca0e1d81a2f0772dd4400d8b860f71c09eaf0ee3f5120`.
Its OCI index digest is
`sha256:ce5fbe63b0e5186535f7e58074eb81290a729138798160e42553667b9ccb895f`.
The registry manifest and image configuration were retrieved on 2026-10-04;
this is an identified upstream image, not a locally executed H20 test.

The image config records `CUDA_VERSION=12.9.1` and cudart `12.9.79`.
Its build selects the official PyTorch `cu129` index and FlashInfer 0.6.3
cu129 JIT cache. The actual official index contains **torch 2.9.1+cu129**
Linux x86_64 wheels, even though the short previous-versions webpage omits
that variant. The launch preflight checks the actual installed packages and
`torch.version.cuda`, not just the Docker tag or `nvidia-smi` banner.

| Component | Selected upstream contract |
| --- | --- |
| Container toolkit | CUDA 12.9.1 |
| PyTorch | 2.9.1+cu129; runtime 12.9 |
| SGLang | 0.5.9, source `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` |
| sgl-kernel | 0.3.21 |
| FlashInfer | 0.6.3, including cu129 JIT cache |
| Transformers | 4.57.1 |
| Megatron-LM | `3714d81d418c9f1bca4594fc35f9e8289f652862` plus upstream image patches |
| slime extension API | v0.2.3, `6195417d45f272b72d04f619ea334613135d1c1f` |
| GPU | NVIDIA H20, compute capability 9.0 |

The published image was built with `SLIME_COMMIT=main` before the release tag.
The wrapper therefore checks out the exact v0.2.3 commit in the disposable
container and installs that package with `--no-deps --no-build-isolation`.
The SGLang/Megatron/native dependency stack stays in the pinned image. ACE-SQL
CPU dependencies are installed separately with `--no-deps`; unrecognized
missing or incompatible dependencies fail preflight.

The upstream image deliberately replaces torch's cuDNN and NCCL pins with
`nvidia-cudnn-cu12==9.16.0.29` and `nvidia-nccl-cu12==2.28.3`. Preflight records
and accepts only those exact two known torch dependency overrides; all other
`pip check` failures remain fatal. The overrides are recorded in the retrieved
OCI Docker build history and the upstream Dockerfile.

## Container entry points

From the repository root on the H20 Linux host, with NVIDIA Container Toolkit:

```bash
bash scripts/container.sh --dry-run train
bash scripts/container.sh preflight
bash scripts/container.sh prepare
bash scripts/container.sh train
```

The default mode is preflight. `prepare` invokes `ace_sql.runtime.prepare`;
`train` invokes preparation followed by `ace_sql.training.launch`. Both use the
same `ACE_SQL_CONFIG` TOML and environment overrides.
`--dry-run` prints the resolved Docker command without pulling or executing it.
The wrapper forwards `ACE_SQL_*` settings, mounts the repository and configured
external asset directories at the same absolute paths, and leaves the host's
Python/CUDA installation unchanged. `shell` opens the pinned container for
manual inspection. Use `ACE_SQL_ASSETS_DIR` to mount a common directory for absolute
paths written only in TOML; otherwise set the corresponding `ACE_SQL_*` path
overrides so the wrapper can mount them. Config files should live under the
mounted repository or asset directory. The native image is approximately 25.5 GB compressed.

A direct environment check is also available:

```bash
ace-sql check-env \
  --slime-dir /root/slime --expected-cuda 12.9 --expected-gpus 8 \
  --gpu-smoke --report /path/to/run/slime-environment.json
```

The default requires runtime and toolkit 12.9, and driver API capability
**at least** 12.9; newer drivers are accepted. Driver capability, toolkit,
and torch build are separate recorded values. Older drivers may require
NVIDIA compatibility libraries, but that is a separate validation path,
not an assumed property of this recipe.

An accepted training validation still requires model load, a full retrieval
and generation rollout, SQL reward computation, reference/old-policy log
probabilities, backward/optimizer step, weight synchronization, and a second
rollout using the new weights. Neither registry metadata, CPU contract tests,
nor a BF16 matrix multiplication establish those results. No GPU training
validation or container execution was performed while preparing this package.

## Original strict CUDA 12.4 option

The user accepted CUDA 12.9, so strict 12.4 is no longer the default deployment.
`--expected-cuda 12.4` remains an explicit check for a separately validated
custom build. Official torch 2.6.0+cu124 exists, but replacing torch 2.9.1
with it does not satisfy the pinned SGLang API stack. No validated strict-cu124
slime v0.2.3 environment is claimed here.

## Pinned extension interfaces

The v0.2.3 source is authoritative when it differs from current online docs:

- `--rollout-function-path`: synchronous
  `generate_rollout(args, rollout_id, data_source, evaluation=False)` returns
  `RolloutFnTrainOutput(samples=list[list[Sample]], metrics=dict)` for training
  and `RolloutFnEvalOutput(data=dict[str, dict], metrics=dict)` for evaluation.
- `--custom-reward-post-process-path`: `(args, samples)` returns
  `(raw_rewards, rewards)`, with one scalar per flattened sample. Use the
  explicit role/group identity for normalization; automatic positional GRPO
  normalization is not suitable for mixed roles.
- `--loss-type custom_loss --custom-loss-function-path ...`: the function
  `(args, batch, logits, sum_of_sample_mean)` returns `(loss, metrics)`;
  `loss` is a scalar tensor and metrics are detached scalar tensors. The
  Megatron wrapper applies microbatch/data-parallel scaling afterward.
- The stock conversion creates `train_data["metadata"]` from
  `Sample.train_metadata`, but the stock DP splitter drops it and the training
  batch reader does not request it. The adapter needs an explicit, reviewed
  metadata-forwarding patch at both boundaries before a custom loss can use
  `batch["metadata"]`. Ordinary `Sample.metadata` is not training metadata.
- A `Sample` contains prompt-plus-response `tokens`, `response_length`,
  response-only `loss_mask`, `reward`, `index`, `group_index`, and completion
  status. Both roles can use the same policy and SGLang serving pool.
  v0.2.3 does **not** have `Sample.rollout_id`; current documentation's sibling
  aggregation semantics must not be assumed. Role weighting must account for
  samples explicitly.
- Do not depend on `--rollout-data-postprocess-path` receiving a data batch:
  v0.2.3 actually invokes this callback as `hook(args)`.

## Primary sources

- [Official slime Docker tag metadata](https://hub.docker.com/v2/repositories/slimerl/slime/tags/v0.2.3)
- [Official torch cu129 wheel index](https://download.pytorch.org/whl/cu129/torch/)
- [slime v0.2.3 Dockerfile](https://github.com/THUDM/slime/blob/6195417d45f272b72d04f619ea334613135d1c1f/docker/Dockerfile)
- [slime v0.2.3 image matrix](https://github.com/THUDM/slime/blob/6195417d45f272b72d04f619ea334613135d1c1f/docker/README.md)
- [SGLang v0.5.9 dependency contract](https://github.com/sgl-project/sglang/blob/v0.5.9/python/pyproject.toml)
- [Pinned rollout and reward contracts](https://github.com/THUDM/slime/blob/6195417d45f272b72d04f619ea334613135d1c1f/slime/ray/rollout.py)
- [Pinned loss contract](https://github.com/THUDM/slime/blob/6195417d45f272b72d04f619ea334613135d1c1f/slime/backends/megatron_utils/loss.py)
- [NVIDIA CUDA compatibility limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
