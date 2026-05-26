# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from collections import OrderedDict

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import (
    fsdp_version,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.model import check_exclude_modules, check_target_modules, convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import (
    TensorLoRARequest,
    VLLMHijack,
    is_version_ge,
    lora_tensors_fingerprint,
    patch_vllm_moe_model_weight_loader,
)

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _trace_lora_path(message: str) -> None:
    if os.getenv("ACE_SQL_LORA_TRACE", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    trace_file = os.getenv("ACE_SQL_LORA_TRACE_FILE", ".run/lora_trace.txt")
    try:
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} {message}\n")
    except Exception as exc:
        logger.warning("Failed to write LoRA trace to %s: %s", trace_file, exc)


def _dtype_from_rollout_config(dtype_name, model_config=None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    dtype_name = str(dtype_name).lower()
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name[len("torch."):]
    if dtype_name in {"auto", "none"}:
        inferred_dtype = getattr(model_config, "torch_dtype", None)
        if isinstance(inferred_dtype, torch.dtype):
            return inferred_dtype
        if inferred_dtype is not None:
            return _dtype_from_rollout_config(inferred_dtype, None)
        return None
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"fp32", "float32", "float"}:
        return torch.float32
    return None


def _require_vllm_lora_execution_dtype(dtype: torch.dtype | None, dtype_source) -> torch.dtype:
    if dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "Dual LoRA vLLM sync requires fp16/bf16 LoRA execution dtype because "
            "vLLM Punica asserts that hidden states and LoRA weights have the same "
            f"fp16/bf16 dtype. Got {dtype} from rollout dtype {dtype_source!r}."
        )
    return dtype


def _cast_lora_tensors_for_vllm(lora_params: OrderedDict, target_dtype: torch.dtype | None) -> OrderedDict:
    if target_dtype is None:
        return lora_params
    return OrderedDict(
        (name, tensor.to(dtype=target_dtype) if torch.is_floating_point(tensor) else tensor)
        for name, tensor in lora_params.items()
    )


class FSDPVLLMShardingManager(BaseShardingManager):
    """Sharding manager for FSDP models with vLLM inference engine integration.

    Manages parameter synchronization between FSDP training models and vLLM
    inference engines, handling both full parameters and LoRA adapters with
    efficient memory management and device placement.
    """

    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = "dummy_hf",
        layered_summon: bool = True,
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if
        # inference_engine else None

        self.model_runner = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            if self.inference_engine
            else None
        )

        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        self._is_dual_lora = (
            hasattr(peft_model, "peft_config")
            and "retriever" in getattr(peft_model, "peft_config", {})
        )
        self._ret_lora_id = 1
        self._gen_lora_id = 2
        self._single_lora_id = 1
        self._dirty_lora_adapters = {"retriever", "generator"} if self._is_dual_lora else set()
        self._loaded_lora_adapters = set()
        self._is_single_lora = (
            hasattr(peft_model, "peft_config")
            and not self._is_dual_lora
            and "default" in getattr(peft_model, "peft_config", {})
        )
        self._single_lora_loaded = False
        self._dirty_single_lora = self._is_single_lora

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        def __collect_lora_params(adapter_name=None) -> OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm.
            When adapter_name is given and base is synced, only that adapter's params are returned.
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError(
                            "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                            "rollout.load_format=safetensors"
                        )
                    all_lora = layered_summon_lora_params(self.module)
                    if adapter_name and self._is_dual_lora:
                        lora_params = OrderedDict(
                            (k, v) for k, v in all_lora.items() if adapter_name in k
                        )
                    else:
                        lora_params = all_lora
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            if adapter_name:
                                lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
                            else:
                                lora_params = get_peft_model_state_dict(peft_model)
                            lora_params = {
                                name: param.full_tensor().detach().cpu()
                                if hasattr(param, "full_tensor")
                                else param.detach().cpu()
                                for name, param in lora_params.items()
                            }
                        else:
                            model = peft_model.base_model.model
                            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                            model = model.to("cpu")
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ["_flat_param", "lora_"]):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                                lora_params[name] = (
                                    param.full_tensor().detach().cpu()
                                    if hasattr(param, "full_tensor")
                                    else param.detach().cpu()
                                )
                            model = model.to(orig_dev)
                    get_torch_device().empty_cache()
            else:
                if self.base_sync_done:
                    if adapter_name:
                        lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
                    else:
                        lora_params = get_peft_model_state_dict(peft_model)
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)
            return lora_params

        def __sync_dual_lora_adapters(adapter_names=None) -> None:
            if peft_config is None:
                raise RuntimeError("Dual LoRA sync requires a PEFT config, but none was found")
            if adapter_names is None:
                adapter_names = {"retriever", "generator"}
            adapter_names = set(adapter_names)
            valid_adapter_names = {"retriever", "generator"}
            bad_adapter_names = adapter_names - valid_adapter_names
            if bad_adapter_names:
                raise RuntimeError(
                    f"Dual LoRA sync received invalid adapter names: {sorted(bad_adapter_names)}"
                )
            if not adapter_names:
                _trace_lora_path("[FSDPVLLMShardingManager.sync_dual_lora] skip=no_dirty_adapters")
                return

            def __lora_runtime():
                if hasattr(self.inference_engine, "llm_engine"):
                    return self.inference_engine.llm_engine
                if hasattr(self.inference_engine, "worker") and hasattr(self.inference_engine.worker, "add_lora"):
                    return self.inference_engine.worker
                if self.model_runner is not None and hasattr(self.model_runner, "add_lora"):
                    return self.model_runner
                raise RuntimeError("Cannot find add_lora method on inference_engine or model_runner")

            def __list_loras(runtime):
                if not hasattr(runtime, "list_loras"):
                    return "unavailable"
                try:
                    loras = runtime.list_loras()
                    if isinstance(loras, dict):
                        loras = loras.keys()
                    return sorted(int(lora_id) for lora_id in loras)
                except Exception as exc:
                    return f"error:{type(exc).__name__}:{exc}"

            def __remove_lora(runtime, lora_id: int):
                remove_fn = getattr(runtime, "remove_lora", None)
                if remove_fn is None:
                    remove_fn = getattr(runtime, "remove_adapter", None)
                if remove_fn is None:
                    raise RuntimeError("Cannot reload dual LoRA adapter because runtime has no remove_lora method")
                return remove_fn(lora_id)

            required_lora_ids = {self._ret_lora_id, self._gen_lora_id}
            runtime = __lora_runtime()
            for adapter_name, lora_id in [("retriever", self._ret_lora_id), ("generator", self._gen_lora_id)]:
                if adapter_name not in adapter_names:
                    continue
                adapter_peft_config = peft_model.peft_config.get(adapter_name, None)
                if adapter_peft_config is None:
                    raise RuntimeError(f"Dual LoRA sync missing PEFT config for adapter '{adapter_name}'")
                adapter_params = __collect_lora_params(adapter_name=adapter_name)
                adapter_params = convert_weight_keys(
                    adapter_params,
                    getattr(self.module, "_fsdp_wrapped_module", self.module),
                )
                if not adapter_params:
                    raise RuntimeError(f"Dual LoRA sync collected no parameters for adapter '{adapter_name}'")
                target_dtype = _require_vllm_lora_execution_dtype(
                    _dtype_from_rollout_config(self.rollout_config.get("dtype", None), self.model_config),
                    self.rollout_config.get("dtype", None),
                )
                adapter_params = _cast_lora_tensors_for_vllm(
                    adapter_params,
                    target_dtype,
                )
                lora_request = TensorLoRARequest(
                    lora_name=f"{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(adapter_peft_config),
                    lora_tensors=adapter_params,
                )
                tensor_fp = lora_tensors_fingerprint(adapter_params)
                loras_before = __list_loras(runtime)
                already_loaded = lora_id in loras_before if isinstance(loras_before, list) else None
                remove_result = None
                loras_after_remove = None
                if already_loaded:
                    remove_result = __remove_lora(runtime, lora_id)
                    loras_after_remove = __list_loras(runtime)
                    if isinstance(loras_after_remove, list) and lora_id in loras_after_remove:
                        raise RuntimeError(
                            f"Failed to remove stale dual LoRA adapter '{adapter_name}' (id={lora_id}) before reload"
                        )
                add_result = runtime.add_lora(lora_request)
                loras_after = __list_loras(runtime)
                if not isinstance(loras_after, list):
                    raise RuntimeError(
                        "Dual LoRA sync cannot verify loaded LoRA ids because runtime.list_loras() "
                        f"returned {loras_after!r}"
                    )
                if lora_id not in loras_after:
                    raise RuntimeError(
                        f"Dual LoRA sync failed to load adapter '{adapter_name}' as id={lora_id}; "
                        f"runtime LoRA ids after add_lora: {loras_after}"
                    )
                logger.info(
                    "Dual LoRA: add_lora adapter '%s' (id=%s), params=%s, "
                    "already_loaded=%s, remove_result=%s, add_result=%s, "
                    "loras_before=%s, loras_after_remove=%s, loras_after=%s, tensor_fp=%s",
                    adapter_name,
                    lora_id,
                    len(adapter_params),
                    already_loaded,
                    remove_result,
                    add_result,
                    loras_before,
                    loras_after_remove,
                    loras_after,
                    tensor_fp,
                )
                _trace_lora_path(
                    "[FSDPVLLMShardingManager.sync_dual_lora] "
                    f"adapter_name={adapter_name} lora_int_id={lora_id} "
                    f"params={len(adapter_params)} base_sync_done={self.base_sync_done} "
                    f"already_loaded={already_loaded} remove_result={remove_result} add_result={add_result} "
                    f"loras_before={loras_before} loras_after_remove={loras_after_remove} loras_after={loras_after} "
                    f"tensor_fp={tensor_fp}"
                )
                self._loaded_lora_adapters.add(adapter_name)
                self._dirty_lora_adapters.discard(adapter_name)
            loras_final = __list_loras(runtime)
            synced_lora_ids = {
                self._ret_lora_id if adapter_name == "retriever" else self._gen_lora_id
                for adapter_name in adapter_names
            }
            if not isinstance(loras_final, list) or not synced_lora_ids.issubset(set(loras_final)):
                raise RuntimeError(
                    f"Dual LoRA sync requires loaded LoRA ids {sorted(synced_lora_ids)}, "
                    f"got {loras_final}"
                )
            if required_lora_ids.issubset(set(loras_final)):
                self._loaded_lora_adapters.update({"retriever", "generator"})

        self.timing = {}
        with simple_timer("reshard", self.timing):
            get_torch_device().empty_cache()

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)

            peft_config = None
            params = None
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            if hasattr(peft_model, "peft_config"):
                if self._is_dual_lora:
                    peft_config = peft_model.peft_config.get("retriever", None)
                else:
                    peft_config = peft_model.peft_config.get("default", None)

                if self._is_dual_lora and self.base_sync_done:
                    params = None
                elif (
                    self._is_single_lora
                    and self.base_sync_done
                    and self._single_lora_loaded
                    and not self._dirty_single_lora
                ):
                    _trace_lora_path(
                        "[FSDPVLLMShardingManager.sync_single_lora] skip=clean_loaded_adapter "
                        f"lora_int_id={self._single_lora_id}"
                    )
                    params = None
                else:
                    params = __collect_lora_params()
            else:
                params = self.module.state_dict()

            if params is not None:
                params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

            if self.rollout_config.free_cache_engine:
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()

            if self._is_dual_lora and self.base_sync_done:
                adapters_to_sync = set(self._dirty_lora_adapters)
                if not {"retriever", "generator"}.issubset(self._loaded_lora_adapters):
                    adapters_to_sync.update({"retriever", "generator"})
                __sync_dual_lora_adapters(adapter_names=adapters_to_sync)
            elif params is not None:
                self.update_params(params, peft_config=peft_config)
                if self._is_dual_lora:
                    __sync_dual_lora_adapters(adapter_names={"retriever", "generator"})
                elif self._is_single_lora and not self._single_lora_loaded:
                    # The first dummy-load sync only transfers frozen base weights.
                    # vLLM still needs the default adapter mounted as fixed id=1
                    # before the first rollout request can safely use LoRA.
                    single_lora_params = __collect_lora_params()
                    single_lora_params = convert_weight_keys(
                        single_lora_params,
                        getattr(self.module, "_fsdp_wrapped_module", self.module),
                    )
                    if not single_lora_params:
                        raise RuntimeError("Single LoRA sync collected no adapter parameters after base sync.")
                    self.update_params(single_lora_params, peft_config=peft_config)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if (
                self.rollout_config.free_cache_engine
                and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
            ):
                self.inference_engine.wake_up(tags=["kv_cache"])

            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.rollout_config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        """Update model parameters in the vLLM inference engine.

        Synchronizes parameters from the FSDP training model to the vLLM inference
        engine, handling both full model parameters and LoRA adapters with proper
        device placement and memory management.

        Args:
            updated_params (dict): Dictionary of parameter names to tensor values.
            peft_config (optional): PEFT configuration for LoRA adapters.
        """
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                if self._is_dual_lora:
                    raise RuntimeError(
                        "Dual LoRA must be synced through fixed adapter ids "
                        "1(retriever) and 2(generator); refusing generic random-id LoRA update_params path."
                    )

                def __lora_runtime():
                    engine = self.inference_engine
                    if hasattr(engine, "llm_engine"):
                        return engine.llm_engine
                    if hasattr(engine, "worker") and hasattr(engine.worker, "add_lora"):
                        return engine.worker
                    if self.model_runner is not None and hasattr(self.model_runner, "add_lora"):
                        return self.model_runner
                    raise RuntimeError("Cannot find add_lora method on inference_engine or model_runner")

                def __list_loras(runtime):
                    if not hasattr(runtime, "list_loras"):
                        raise RuntimeError("Single LoRA sync requires runtime.list_loras() for fail-fast verification.")
                    loras = runtime.list_loras()
                    if isinstance(loras, dict):
                        loras = loras.keys()
                    return sorted(int(lora_id) for lora_id in loras)

                def __remove_lora(runtime, lora_id: int):
                    remove_fn = getattr(runtime, "remove_lora", None)
                    if remove_fn is None:
                        remove_fn = getattr(runtime, "remove_adapter", None)
                    if remove_fn is None:
                        raise RuntimeError("Cannot reload single LoRA adapter because runtime has no remove_lora method")
                    return remove_fn(lora_id)

                target_dtype = _require_vllm_lora_execution_dtype(
                    _dtype_from_rollout_config(self.rollout_config.get("dtype", None), self.model_config),
                    self.rollout_config.get("dtype", None),
                )
                updated_params = _cast_lora_tensors_for_vllm(updated_params, target_dtype)
                lora_int_id = self._single_lora_id
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                runtime = __lora_runtime()
                loras_before = __list_loras(runtime)
                already_loaded = lora_int_id in loras_before
                remove_result = None
                loras_after_remove = None
                if already_loaded:
                    remove_result = __remove_lora(runtime, lora_int_id)
                    loras_after_remove = __list_loras(runtime)
                    if lora_int_id in loras_after_remove:
                        raise RuntimeError(
                            f"Failed to remove stale single LoRA adapter id={lora_int_id} before reload"
                        )
                add_result = runtime.add_lora(lora_reqest)
                loras_after = __list_loras(runtime)
                if lora_int_id not in loras_after:
                    raise RuntimeError(
                        f"Single LoRA sync failed to load adapter id={lora_int_id}; "
                        f"runtime LoRA ids after add_lora: {loras_after}"
                    )
                tensor_fp = lora_tensors_fingerprint(updated_params)
                logger.info(
                    "Single LoRA: add_lora id=%s, params=%s, already_loaded=%s, "
                    "remove_result=%s, add_result=%s, loras_before=%s, "
                    "loras_after_remove=%s, loras_after=%s, tensor_fp=%s",
                    lora_int_id,
                    len(updated_params),
                    already_loaded,
                    remove_result,
                    add_result,
                    loras_before,
                    loras_after_remove,
                    loras_after,
                    tensor_fp,
                )
                _trace_lora_path(
                    "[FSDPVLLMShardingManager.sync_single_lora] "
                    f"lora_int_id={lora_int_id} params={len(updated_params)} "
                    f"base_sync_done={self.base_sync_done} already_loaded={already_loaded} "
                    f"remove_result={remove_result} add_result={add_result} "
                    f"loras_before={loras_before} loras_after_remove={loras_after_remove} "
                    f"loras_after={loras_after} tensor_fp={tensor_fp}"
                )
                self._single_lora_loaded = True
                self._dirty_single_lora = False
                return
            else:

                def replace_lora_wrapper(k):
                    """Replace LoRA parameter keys with base layer equivalents.

                    Transforms LoRA parameter names to their corresponding base layer
                    names for proper weight loading in vLLM when base model sync is not done.

                    Args:
                        k (str): Original parameter key name.

                    Returns:
                        str: Transformed parameter key for base layer.
                    """
                    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    if k.endswith(".weight"):
                        module_k = k[: -len(".weight")]
                        if check_exclude_modules(peft_config, module_k):
                            return k
                        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                            peft_config, module_k
                        ):
                            return f"{module_k}.base_layer.weight"
                    if k.endswith(".bias"):
                        module_k = k[: -len(".bias")]
                        if check_exclude_modules(peft_config, module_k):
                            return k
                        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                            peft_config, module_k
                        ):
                            return f"{module_k}.base_layer.bias"
                    return k

                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(
            (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in updated_params.items()
            )
        )

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
