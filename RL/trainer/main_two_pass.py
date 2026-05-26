"""
Entry point for Two-Pass GRPO joint training on VERL 0.5.5.
"""

import os
import socket
import sys
import time
import json

import hydra
import ray
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@hydra.main(config_path="config", config_name="two_pass_grpo", version_base=None)
def main(config):
    run_two_pass_grpo(config)


def run_two_pass_grpo(config):
    """Initialize Ray and run two-pass GRPO training."""
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        spill_dirs = config.ray_init.get("object_spilling_dirs", None)
        if spill_dirs:
            spill_dirs = list(OmegaConf.to_container(spill_dirs, resolve=True))
            for spill_dir in spill_dirs:
                os.makedirs(spill_dir, exist_ok=True)
            os.environ["RAY_object_spilling_config"] = json.dumps(
                {
                    "type": "filesystem",
                    "params": {
                        "directory_path": spill_dirs,
                    },
                }
            )
        else:
            os.environ.pop("RAY_object_spilling_config", None)

        ray_init_kwargs = {
            "runtime_env": get_ppo_ray_runtime_env(),
            "num_cpus": config.ray_init.num_cpus,
        }
        temp_dir = config.ray_init.get("temp_dir", None)
        if temp_dir:
            ray_init_kwargs["_temp_dir"] = temp_dir
        object_store_memory = config.ray_init.get("object_store_memory", None)
        if object_store_memory:
            ray_init_kwargs["object_store_memory"] = int(object_store_memory)
        ray.init(**ray_init_kwargs)

    task_runner_cls = ray.remote(num_cpus=1)(TwoPassTaskRunner)
    runner = task_runner_cls.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TwoPassTaskRunner:
    """Ray remote task runner for two-pass GRPO training."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def run(self, config):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        _t0 = time.time()

        def _log(msg):
            elapsed = time.time() - _t0
            print(f"[TwoPass {elapsed:7.1f}s] {msg}", flush=True)

        _log(f"hostname: {socket.gethostname()}")
        _log("resolving config...")
        OmegaConf.resolve(config)
        _log(
            f"use_kl_loss={config.actor_rollout_ref.actor.use_kl_loss}, "
            f"use_kl_in_reward={config.algorithm.use_kl_in_reward}"
        )

        _log("setting up actor rollout worker class...")
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ref_policy_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ref_policy_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(
                f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"

        use_ref = config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
        _log(f"need_reference_policy={use_ref}")
        if use_ref:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

        _log(f"loading tokenizer from {config.actor_rollout_ref.model.path}...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.actor_rollout_ref.model.get(
            "trust_remote_code",
            config.data.get("trust_remote_code", False),
        )
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        _log("tokenizer loaded")

        _log("creating resource pool...")
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

        _log(f"loading train dataset: {config.data.train_files}")
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
        )
        _log(f"train dataset: {len(train_dataset)} samples")
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
        )
        _log(f"val dataset: {len(val_dataset)} samples")
        train_sampler = create_rl_sampler(config.data, train_dataset)

        _log("creating TwoPassGRPOTrainer...")
        from trainer.two_pass_ray_trainer import TwoPassGRPOTrainer

        trainer = TwoPassGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        _log("calling trainer.init_workers()...")
        trainer.init_workers()
        _log("init_workers() done, starting trainer.fit()...")
        trainer.fit()
        _log("fit() completed")


if __name__ == "__main__":
    main()
