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


import hashlib
import os

import torch
from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.third_party.vllm import get_version

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []


def lora_trace_enabled() -> bool:
    return os.getenv("ACE_SQL_LORA_TRACE", "0").lower() in {"1", "true", "yes", "on"}


def trace_lora_path(message: str) -> None:
    if not lora_trace_enabled():
        return
    trace_file = os.getenv("ACE_SQL_LORA_TRACE_FILE", ".run/lora_trace.txt")
    try:
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} {message}\n")
    except Exception:
        pass


def lora_tensors_fingerprint(tensors: dict | None) -> str:
    """Small diagnostic fingerprint for LoRA tensors.

    This intentionally computes a few full-tensor reductions only when LoRA
    tracing is enabled. It is for diagnosis, not for cryptographic identity.
    """
    if not lora_trace_enabled():
        return "disabled"
    if tensors is None:
        return "none"

    digest = hashlib.sha256()
    num_tensors = 0
    total_numel = 0
    total_sum = 0.0
    total_abs_sum = 0.0
    total_sq_sum = 0.0
    max_abs = 0.0
    sample_values = 0
    for name, tensor in sorted(tensors.items()):
        if tensor is None or not hasattr(tensor, "detach"):
            continue
        data = tensor.detach().float().cpu().reshape(-1)
        numel = int(data.numel())
        if numel == 0:
            continue
        num_tensors += 1
        total_numel += numel
        tensor_sum = float(data.sum().item())
        tensor_abs_sum = float(data.abs().sum().item())
        tensor_sq_sum = float(data.square().sum().item())
        tensor_max_abs = float(data.abs().max().item())
        total_sum += tensor_sum
        total_abs_sum += tensor_abs_sum
        total_sq_sum += tensor_sq_sum
        max_abs = max(max_abs, tensor_max_abs)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            f"{tensor_sum:.16e}:{tensor_abs_sum:.16e}:{tensor_sq_sum:.16e}:{tensor_max_abs:.16e}".encode("utf-8")
        )
        if numel <= 2048:
            sample = data.contiguous()
        else:
            sample_indices = (torch.linspace(0, numel - 1, steps=2048).round().long())
            sample = data[sample_indices].contiguous()
        sample_values += int(sample.numel())
        digest.update(sample.numpy().tobytes())

    return (
        f"digest={digest.hexdigest()[:16]},tensors={num_tensors},numel={total_numel},"
        f"samples={sample_values},sum={total_sum:.12e},abs={total_abs_sum:.12e},"
        f"sq={total_sq_sum:.12e},max_abs={max_abs:.12e}"
    )

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {
        MixtralForCausalLM: "block_sparse_moe",
    }
    DEFAULT_MLP_ATTR = "mlp"

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)):
        return

    model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    for layer in model.layers:
        mlp_attr = MLP_ATTR_MAPPING.get(type(model), DEFAULT_MLP_ATTR)
        mlp = getattr(layer, mlp_attr)

        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                    trace_lora_path(
                        "[VLLMHijack._load_adapter] "
                        f"lora_int_id={lora_request.lora_int_id} "
                        f"lora_name={lora_request.lora_name} "
                        f"tensor_fp={lora_tensors_fingerprint(lora_tensors)}"
                    )
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        embeddings=None,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
            except Exception as e:
                raise e

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        if not hasattr(LRUCacheWorkerLoRAManager, "_ace_sql_original_add_adapter"):
            LRUCacheWorkerLoRAManager._ace_sql_original_add_adapter = LRUCacheWorkerLoRAManager.add_adapter
        original_add_adapter = LRUCacheWorkerLoRAManager._ace_sql_original_add_adapter

        def hijack_add_adapter(self, lora_request: LoRARequest) -> bool:
            lora_id = int(lora_request.lora_int_id)
            loras_before = sorted(int(adapter_id) for adapter_id in self.list_adapters())
            already_loaded = lora_id in loras_before
            request_fp = (
                lora_tensors_fingerprint(lora_request.lora_tensors)
                if isinstance(lora_request, TensorLoRARequest)
                else "not_tensor_request"
            )
            trace_lora_path(
                "[VLLMHijack.add_adapter.before] "
                f"lora_int_id={lora_id} already_loaded={already_loaded} "
                f"will_call_load_adapter={not already_loaded} "
                f"loras_before={loras_before} request_tensor_fp={request_fp}"
            )
            result = original_add_adapter(self, lora_request)
            loras_after = sorted(int(adapter_id) for adapter_id in self.list_adapters())
            trace_lora_path(
                "[VLLMHijack.add_adapter.after] "
                f"lora_int_id={lora_id} already_loaded={already_loaded} "
                f"called_load_adapter={not already_loaded} add_result={result} "
                f"loras_after={loras_after} request_tensor_fp={request_fp}"
            )
            return result

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)
        do_hijack(LRUCacheWorkerLoRAManager, "add_adapter", hijack_add_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
