#!/usr/bin/env python3
"""
Maj-Voting 模式的单个检查点评估脚本

流程:
1. 合并 LoRA 权重
2. 准备评估数据
3. 检索器推理 (n=1, temperature=0.0)
4. 合并检索结果 (取并集或majority voting)
5. 生成 SQL 提示词
6. SQL 生成 (n=1, temperature=0.0)
7. Majority Voting 评估

用法:
    python single_eval_maj.py --checkpoint_path /path/to/checkpoint --output_dir /path/to/output
"""

import argparse
import glob
import os
import sys
import subprocess
import json
import re
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    BASE_MODEL_PATH, SQL_MODEL_PATH,
    BIRD_DEV_DATA, BIRD_TABLES, BIRD_DB_PATH,
    RETRIEVER_PROMPT_PATH, VALUE_LIMIT_NUM, DEFAULT_OUTPUT_DIR,
    VLLM_CONFIG, VERL_ROOT,
)

# Maj-voting 配置
MAJ_NUM_SAMPLES = 1
MAJ_TEMPERATURE = 0.0

# 推理最大长度配置
RETRIEVER_MAX_TOKENS = 4096  # 原来是 512，增大 8 倍
SQL_MAX_TOKENS = 8192        # 原来是 2048，增大 4 倍


def run_command(cmd, cwd=None, capture_output=False):
    """运行命令并返回结果"""
    print(f"Running: {' '.join(cmd)}")
    if capture_output:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, cwd=cwd, capture_output=False)
        return result.returncode == 0, None, None


def validate_generated_file(path: str, step_name: str) -> bool:
    """校验步骤产物是否真实生成，避免子脚本空跑时被误判成功。"""
    if not os.path.exists(path):
        print(f"Error: {step_name}未生成输出文件: {path}")
        return False
    if os.path.getsize(path) == 0:
        print(f"Error: {step_name}生成了空文件: {path}")
        return False
    return True


def append_vllm_runtime_args(
    cmd,
    dtype: str = None,
    tokenizer_path: str = None,
    lora_adapter_path: str = None,
    lora_name: str = None,
    lora_int_id: int = None,
    lora_dtype: str = "auto",
    max_lora_rank: int = 64,
):
    """给 retriever/sql 推理脚本追加可选的 vLLM 运行时参数。"""
    if dtype:
        cmd.extend(["--dtype", dtype])
    if tokenizer_path:
        cmd.extend(["--tokenizer_path", tokenizer_path])
    if lora_adapter_path:
        cmd.extend([
            "--lora_adapter_path", lora_adapter_path,
            "--lora_name", lora_name or "default",
            "--lora_int_id", str(lora_int_id or 1),
            "--lora_dtype", lora_dtype,
            "--max_lora_rank", str(max_lora_rank),
        ])
    return cmd


def _has_model_weights(directory: str) -> bool:
    """检查目录中是否有模型权重文件（.safetensors 或 .bin）"""
    for ext in ("*.safetensors", "*.bin"):
        if glob.glob(os.path.join(directory, ext)):
            return True
    return False


def _find_fsdp_actor_dir(checkpoint_path: str):
    """
    从 checkpoint_path 向上查找 FSDP actor 目录。

    支持两种传入方式：
      - .../actor/huggingface  -> actor_dir = .../actor
      - .../actor              -> actor_dir = .../actor

    Returns:
        actor_dir (str) 如果找到合法的 FSDP actor 目录，否则 None
    """
    candidates = [checkpoint_path]
    parent = os.path.dirname(checkpoint_path.rstrip("/"))
    if parent != checkpoint_path:
        candidates.append(parent)

    for d in candidates:
        fsdp_config = os.path.join(d, "fsdp_config.json")
        if os.path.isfile(fsdp_config) and glob.glob(os.path.join(d, "model_world_size_*_rank_*.pt")):
            return d
    return None


def merge_fsdp_checkpoint(actor_dir: str, target_dir: str) -> bool:
    """
    调用 verl.model_merger 将 FSDP 分片合并为 HuggingFace 格式。

    Args:
        actor_dir:  包含 fsdp_config.json 和 model_world_size_*_rank_*.pt 的目录
        target_dir: 合并后 HuggingFace 模型的保存目录（通常就是 actor_dir/huggingface）

    Returns:
        是否成功
    """
    env = os.environ.copy()
    if VERL_ROOT:
        python_path = env.get("PYTHONPATH", "")
        if VERL_ROOT not in python_path:
            env["PYTHONPATH"] = f"{VERL_ROOT}:{python_path}"

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", actor_dir,
        "--target_dir", target_dir,
    ]
    print(f"Running FSDP merge: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def detect_checkpoint_type(checkpoint_path: str):
    """
    检测检查点类型并返回实际的LoRA路径和是否需要跳过merge

    支持四种情况：
      1. 已合并模型且权重文件完整 -> 跳过merge
      2. FSDP分片检查点（有 fsdp_config.json + model_world_size_*_rank_*.pt，
         但 huggingface/ 下无权重文件）-> 需要先合并FSDP分片
      3. 标准LoRA检查点（根目录有adapter_config.json）-> 直接使用
      4. veRL检查点（lora_adapter/子目录有adapter_config.json）-> 使用子目录路径

    Returns:
        (actual_lora_path, skip_merge, checkpoint_type)
    """
    config_json = os.path.join(checkpoint_path, "config.json")
    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    lora_adapter_dir = os.path.join(checkpoint_path, "lora_adapter")
    lora_adapter_config = os.path.join(lora_adapter_dir, "adapter_config.json")

    if os.path.exists(config_json) and not os.path.exists(adapter_config) and not os.path.isdir(lora_adapter_dir):
        # 有 config.json，看起来像已合并模型，但需要验证权重文件是否存在
        if _has_model_weights(checkpoint_path):
            # 情况1: 真正完整的已合并模型
            return checkpoint_path, True, "merged_model"
        else:
            # 权重文件缺失，检查是否为 FSDP 分片检查点
            fsdp_actor_dir = _find_fsdp_actor_dir(checkpoint_path)
            if fsdp_actor_dir:
                # 情况2: FSDP 分片检查点，需要合并
                return checkpoint_path, False, "fsdp_checkpoint"
            else:
                # config.json 存在但无权重也无 FSDP 分片，仍当作 merged 尝试
                print(f"警告: {checkpoint_path} 中未找到权重文件，也未检测到 FSDP 分片")
                return checkpoint_path, True, "merged_model"
    elif os.path.exists(adapter_config):
        # 情况3: 标准LoRA检查点
        return checkpoint_path, False, "standard_lora"
    elif os.path.exists(lora_adapter_config):
        # 情况4: veRL检查点结构
        return lora_adapter_dir, False, "verl_checkpoint"
    else:
        # 无法识别，尝试作为LoRA检查点处理
        return checkpoint_path, False, "unknown"


def evaluate_single_checkpoint_maj(
    checkpoint_path: str,
    output_dir: str,
    base_model: str = None,
    sql_model: str = None,
    skip_merge: bool = False,
    skip_if_exists: bool = True,
    num_samples: int = MAJ_NUM_SAMPLES,
    temperature: float = MAJ_TEMPERATURE,
    sql_prompts_file: str = None,
    retriever_merge_mode: str = "union",
    add_primary_keys: bool = True,
    retriever_prompt_format: str = "list",
    no_think: bool = False,
    merge_torch_dtype: str = "float16",
    retriever_lora_adapter: str = None,
    generator_lora_adapter: str = None,
    tokenizer_path: str = None,
    retriever_tokenizer_path: str = None,
    generator_tokenizer_path: str = None,
    vllm_dtype: str = None,
    lora_dtype: str = "auto",
    max_lora_rank: int = 64,
):
    """
    使用 Maj-Voting 模式评估单个检查点

    Args:
        checkpoint_path: 检查点路径
        output_dir: 输出目录
        base_model: 基础模型路径
        sql_model: SQL生成模型路径
        skip_merge: 是否跳过模型合并（如果已经是完整模型）
        skip_if_exists: 如果结果已存在是否跳过
        num_samples: 每个样本生成的数量
        temperature: 采样温度
        sql_prompts_file: 直接使用指定的SQL提示词文件，跳过Step 0-4
        retriever_merge_mode: 检索结果合并模式，"union" (取并集) 或 "maj" (majority voting)
        add_primary_keys: 是否添加主键
        retriever_prompt_format: 检索器提示词格式，"list" (列表格式) 或 "json" (JSON格式)
        no_think: 是否在推理时追加 /no_think 标志
        merge_torch_dtype: LoRA merge时加载基础模型的dtype
        retriever_lora_adapter: 可选，vLLM运行时挂载的检索器LoRA
        generator_lora_adapter: 可选，vLLM运行时挂载的生成器LoRA
        tokenizer_path: 可选，统一tokenizer路径
        retriever_tokenizer_path: 可选，检索器tokenizer路径
        generator_tokenizer_path: 可选，生成器tokenizer路径
        vllm_dtype: 可选，vLLM加载模型dtype
        lora_dtype: 可选，vLLM LoRA权重dtype
        max_lora_rank: vLLM LoRA最大rank

    Returns:
        accuracy: 准确率，失败返回None
    """
    base_model = base_model or BASE_MODEL_PATH
    sql_model = sql_model or SQL_MODEL_PATH
    vllm_dtype = vllm_dtype or VLLM_CONFIG["dtype"]
    retriever_tokenizer_path = retriever_tokenizer_path or tokenizer_path
    generator_tokenizer_path = generator_tokenizer_path or tokenizer_path

    # 如果提供了sql_prompts_file，跳过检查点检测
    if retriever_lora_adapter:
        print(f"使用vLLM挂载检索器LoRA: {retriever_lora_adapter}")
        print("将跳过检索器LoRA merge，checkpoint_path作为检索器基础模型加载")
        skip_merge = True
    elif sql_prompts_file:
        print(f"使用外部SQL提示词文件: {sql_prompts_file}")
        print("将跳过 Step 0-4（合并LoRA、准备数据、检索器推理、合并检索结果、生成SQL提示词）")
    else:
        # 自动检测检查点类型（如果没有显式指定skip_merge）
        actual_lora_path, auto_skip_merge, checkpoint_type = detect_checkpoint_type(checkpoint_path)
        print(f"检测到检查点类型: {checkpoint_type}")
        print(f"实际LoRA路径: {actual_lora_path}")

        # 处理 FSDP 分片检查点：自动合并为 HuggingFace 格式
        if checkpoint_type == "fsdp_checkpoint":
            fsdp_actor_dir = _find_fsdp_actor_dir(checkpoint_path)
            target_hf_dir = checkpoint_path  # 合并后的权重保存到当前路径
            if _has_model_weights(target_hf_dir):
                print(f"FSDP 合并结果已存在: {target_hf_dir}，跳过合并")
            else:
                print(f"\n{'=' * 60}")
                print("[FSDP Merge] 合并 FSDP 分片为 HuggingFace 格式...")
                print(f"  FSDP 分片目录: {fsdp_actor_dir}")
                print(f"  目标目录: {target_hf_dir}")
                print(f"{'=' * 60}")
                if not merge_fsdp_checkpoint(fsdp_actor_dir, target_hf_dir):
                    print("Error: FSDP 分片合并失败")
                    return None
                print("FSDP 分片合并完成")
            # 合并完成后，当作已合并模型处理
            skip_merge = True
        # 如果自动检测到是merged模型，覆盖skip_merge参数
        elif auto_skip_merge and not skip_merge:
            print("自动检测到已是merged模型，将跳过LoRA合并步骤")
            skip_merge = True

    os.makedirs(output_dir, exist_ok=True)

    # 定义输出文件路径
    merged_model_path = os.path.join(output_dir, "merged_model")
    retriever_prompts = os.path.join(output_dir, "retriever_prompts.json")
    retriever_outputs = os.path.join(output_dir, "retriever_outputs_maj.json")
    merged_retriever_outputs = os.path.join(output_dir, "retriever_outputs_merged.json")
    sql_prompts = os.path.join(output_dir, "sql_prompts_maj.json")
    sql_outputs = os.path.join(output_dir, "sql_outputs_maj.json")
    eval_log = os.path.join(output_dir, "eval_maj.log")
    result_file = os.path.join(output_dir, "result_maj.json")

    # 检查是否已有结果
    if skip_if_exists and os.path.exists(result_file):
        print(f"结果已存在: {result_file}")
        with open(result_file, 'r') as f:
            result = json.load(f)
        return result.get("accuracy")

    eval_dir = str(project_root)

    # 检查是否已有预测文件，如果有则跳过推理直接评估
    skip_inference = os.path.exists(sql_outputs)
    if skip_inference:
        print("\n" + "=" * 60)
        print(f"发现已有预测文件: {sql_outputs}")
        print("跳过推理步骤 (Step 0-6)，直接进行评估...")
        print("=" * 60)
    elif sql_prompts_file:
        # 使用外部提供的SQL提示词文件，跳过Step 0-4
        print("\n" + "=" * 60)
        print("[跳过 Step 0-4] 使用外部SQL提示词文件...")
        print("=" * 60)
        print(f"  SQL提示词文件: {sql_prompts_file}")

        # 将外部文件路径赋值给sql_prompts
        sql_prompts = sql_prompts_file

        # Step 5: SQL生成 (多样本)
        print("\n" + "=" * 60)
        print(f"[Step 5/7] SQL生成 (n={num_samples}, temperature={temperature})...")
        print("=" * 60)

        if not os.path.exists(sql_outputs):
            sql_cmd = [
                "python3", os.path.join(eval_dir, "sql_infer.py"),
                "--model_path", sql_model,
                "--input_file", sql_prompts,
                "--output_file", sql_outputs,
                "--tensor_parallel_size", "1",
                "--temperature", str(temperature),
                "--max_tokens", str(SQL_MAX_TOKENS),
                "--num_samples", str(num_samples)
            ]
            append_vllm_runtime_args(
                sql_cmd,
                dtype=vllm_dtype,
                tokenizer_path=generator_tokenizer_path,
                lora_adapter_path=generator_lora_adapter,
                lora_name="generator",
                lora_int_id=2,
                lora_dtype=lora_dtype,
                max_lora_rank=max_lora_rank,
            )
            if no_think:
                sql_cmd.append("--no_think")
            success, _, _ = run_command(sql_cmd)
            if not success:
                print("Error: SQL生成失败")
                return None
        else:
            print(f"  SQL输出已存在: {sql_outputs}，跳过")
    else:
        # Step 0: 合并 LoRA 权重
        print("\n" + "=" * 60)
        print("[Step 0/7] 合并LoRA权重...")
        print("=" * 60)

        if skip_merge:
            # 如果跳过合并，直接使用checkpoint作为模型路径
            merged_model_path = checkpoint_path
            if retriever_lora_adapter:
                print(f"  跳过合并，使用基础模型并在vLLM中挂载检索器LoRA: {merged_model_path}")
            else:
                print(f"  跳过合并，使用完整模型: {merged_model_path}")
        elif os.path.exists(merged_model_path):
            print(f"  合并模型已存在: {merged_model_path}，跳过")
        else:
            success, _, _ = run_command([
                "python3", os.path.join(eval_dir, "merge_lora.py"),
                "--base_model_path", base_model,
                "--lora_adapter_path", actual_lora_path,
                "--output_path", merged_model_path,
                "--torch_dtype", merge_torch_dtype
            ])
            if not success:
                print("Error: LoRA合并失败")
                return None

        # Step 1: 准备评估数据
        print("\n" + "=" * 60)
        print("[Step 1/7] 准备评估数据...")
        print("=" * 60)

        # 根据提示词格式选择对应的提示词文件
        if retriever_prompt_format == "json":
            retriever_prompt_path = str(project_root / "prompts" / "retriever_prompt_json.txt")
            print(f"  使用 JSON 格式提示词: {retriever_prompt_path}")
        else:
            retriever_prompt_path = RETRIEVER_PROMPT_PATH
            print(f"  使用列表格式提示词: {retriever_prompt_path}")

        if not os.path.exists(retriever_prompts):
            success, _, _ = run_command([
                "python3", os.path.join(eval_dir, "prepare_eval_data.py"),
                "--dev_data_path", BIRD_DEV_DATA,
                "--tables_path", BIRD_TABLES,
                "--retriever_prompt_path", retriever_prompt_path,
                "--output_path", retriever_prompts
            ])
            if not success:
                print("Error: 数据准备失败")
                return None
        else:
            print(f"  检索器提示词已存在: {retriever_prompts}，跳过")

        # Step 2: 检索器推理 (多样本)
        print("\n" + "=" * 60)
        print(f"[Step 2/7] 检索器推理 (n={num_samples}, temperature={temperature})...")
        print("=" * 60)

        if not os.path.exists(retriever_outputs):
            retriever_cmd = [
                "python3", os.path.join(eval_dir, "retriever_infer.py"),
                "--model_path", merged_model_path,
                "--input_file", retriever_prompts,
                "--output_file", retriever_outputs,
                "--tensor_parallel_size", "1",
                "--temperature", str(temperature),
                "--max_tokens", str(RETRIEVER_MAX_TOKENS),
                "--num_samples", str(num_samples)
            ]
            append_vllm_runtime_args(
                retriever_cmd,
                dtype=vllm_dtype,
                tokenizer_path=retriever_tokenizer_path,
                lora_adapter_path=retriever_lora_adapter,
                lora_name="retriever",
                lora_int_id=1,
                lora_dtype=lora_dtype,
                max_lora_rank=max_lora_rank,
            )
            if no_think:
                retriever_cmd.append("--no_think")
            success, _, _ = run_command(retriever_cmd)
            if not success:
                print("Error: 检索器推理失败")
                return None
        else:
            print(f"  检索器输出已存在: {retriever_outputs}，跳过")

        # Step 3: 合并检索结果
        merge_mode_desc = "取并集" if retriever_merge_mode == "union" else "majority voting"
        print("\n" + "=" * 60)
        print(f"[Step 3/7] 合并检索结果 ({merge_mode_desc})...")
        print("=" * 60)

        if not os.path.exists(merged_retriever_outputs):
            cmd = [
                "python3", os.path.join(eval_dir, "merge_retriever_outputs.py"),
                "--input_file", retriever_outputs,
                "--output_file", merged_retriever_outputs,
                "--mode", retriever_merge_mode
            ]
            if not add_primary_keys:
                cmd.append("--no_add_primary_keys")
            success, _, _ = run_command(cmd)
            if not success:
                print("Error: 合并检索结果失败")
                return None
        else:
            print(f"  合并结果已存在: {merged_retriever_outputs}，跳过")

        # Step 4: 生成SQL提示词
        print("\n" + "=" * 60)
        print("[Step 4/7] 生成SQL提示词...")
        print("=" * 60)

        if not os.path.exists(sql_prompts):
            success, _, _ = run_command([
                "python3", os.path.join(eval_dir, "generate_sql_prompts.py"),
                "--retriever_output_path", merged_retriever_outputs,
                "--db_path", BIRD_DB_PATH,
                "--output_path", sql_prompts,
                "--value_limit_num", str(VALUE_LIMIT_NUM)
            ])
            if not success:
                print("Error: SQL提示词生成失败")
                return None
            if not validate_generated_file(sql_prompts, "SQL提示词生成"):
                return None
        else:
            print(f"  SQL提示词已存在: {sql_prompts}，跳过")

        # Step 5: SQL生成 (多样本)
        print("\n" + "=" * 60)
        print(f"[Step 5/7] SQL生成 (n={num_samples}, temperature={temperature})...")
        print("=" * 60)

        if not os.path.exists(sql_outputs):
            sql_cmd2 = [
                "python3", os.path.join(eval_dir, "sql_infer.py"),
                "--model_path", sql_model,
                "--input_file", sql_prompts,
                "--output_file", sql_outputs,
                "--tensor_parallel_size", "1",
                "--temperature", str(temperature),
                "--max_tokens", str(SQL_MAX_TOKENS),
                "--num_samples", str(num_samples)
            ]
            append_vllm_runtime_args(
                sql_cmd2,
                dtype=vllm_dtype,
                tokenizer_path=generator_tokenizer_path,
                lora_adapter_path=generator_lora_adapter,
                lora_name="generator",
                lora_int_id=2,
                lora_dtype=lora_dtype,
                max_lora_rank=max_lora_rank,
            )
            if no_think:
                sql_cmd2.append("--no_think")
            success, _, _ = run_command(sql_cmd2)
            if not success:
                print("Error: SQL生成失败")
                return None
            if not validate_generated_file(sql_outputs, "SQL生成"):
                return None
        else:
            print(f"  SQL输出已存在: {sql_outputs}，跳过")

    eval_mode = "greedy_search" if num_samples == 1 else "major_voting"
    eval_label = "Greedy Search" if eval_mode == "greedy_search" else "Majority Voting"
    accuracy_pattern = (
        r'EX Accuracy \(greedy search\):\s*([\d.]+)'
        if eval_mode == "greedy_search"
        else r'EX Accuracy \(major voting\):\s*([\d.]+)'
    )

    # Step 6: 执行评估
    print("\n" + "=" * 60)
    print(f"[Step 6/7] {eval_label} 评估...")
    print("=" * 60)

    success, stdout, stderr = run_command([
        "python3", os.path.join(eval_dir, "evaluation_maj.py"),
        "--pred", sql_outputs,
        "--gold", BIRD_DEV_DATA,
        "--db_path", BIRD_DB_PATH,
        "--mode", eval_mode,
        "--timeout", "300"
    ], capture_output=True)

    # 保存评估日志
    with open(eval_log, 'w') as f:
        f.write(stdout or "")
        if stderr:
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
    print(stdout)

    # 提取准确率
    accuracy = None
    if stdout:
        match = re.search(accuracy_pattern, stdout)
        if match:
            accuracy = float(match.group(1))

    print(f"\n准确率 ({eval_label}): {accuracy}")

    # 保存结果
    with open(result_file, 'w') as f:
        json.dump({
            "checkpoint_path": checkpoint_path,
            "accuracy": accuracy,
            "output_dir": output_dir,
            "mode": eval_mode,
            "num_samples": num_samples,
            "temperature": temperature,
            "retriever_merge_mode": retriever_merge_mode,
            "retriever_max_tokens": RETRIEVER_MAX_TOKENS,
            "sql_max_tokens": SQL_MAX_TOKENS,
            "retriever_lora_adapter": retriever_lora_adapter,
            "generator_lora_adapter": generator_lora_adapter,
            "vllm_dtype": vllm_dtype,
            "lora_dtype": lora_dtype if (retriever_lora_adapter or generator_lora_adapter) else None,
            "max_lora_rank": max_lora_rank if (retriever_lora_adapter or generator_lora_adapter) else None,
        }, f, indent=2)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Maj-Voting 模式单个检查点评估")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="检查点路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为 outputs/<checkpoint_name>_maj）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL_PATH,
        help="基础模型路径"
    )
    parser.add_argument(
        "--sql_model",
        type=str,
        default=SQL_MODEL_PATH,
        help="SQL生成模型路径"
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="跳过模型合并（如果已经是完整模型）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新评估（即使结果已存在）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=MAJ_NUM_SAMPLES,
        help=f"每个样本生成的数量 (默认: {MAJ_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=MAJ_TEMPERATURE,
        help=f"采样温度 (默认: {MAJ_TEMPERATURE})"
    )
    parser.add_argument(
        "--sql_prompts_file",
        type=str,
        default=None,
        help="直接使用指定的SQL提示词文件，跳过Step 0-4（合并LoRA、准备数据、检索器推理、合并检索结果、生成SQL提示词）"
    )
    parser.add_argument(
        "--retriever_merge_mode",
        type=str,
        default="union",
        choices=["union", "maj"],
        help="检索结果合并模式: union (取并集) 或 maj (majority voting)，默认 union"
    )
    parser.add_argument(
        "--no_add_primary_keys",
        action="store_true",
        help="不添加主键（默认添加主键）"
    )
    parser.add_argument(
        "--retriever_prompt_format",
        type=str,
        default="list",
        choices=["list", "json"],
        help="检索器提示词格式: list (列表格式) 或 json (JSON格式)，默认 list"
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="在推理时追加 /no_think 标志（禁用模型思考）"
    )
    parser.add_argument(
        "--merge_torch_dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="LoRA merge时加载基础模型的dtype，默认保持原有float16"
    )
    parser.add_argument(
        "--retriever_lora_adapter",
        type=str,
        default=None,
        help="可选：检索器推理时通过vLLM挂载的LoRA adapter路径"
    )
    parser.add_argument(
        "--generator_lora_adapter",
        type=str,
        default=None,
        help="可选：SQL生成时通过vLLM挂载的LoRA adapter路径"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="可选：检索器和生成器统一使用的tokenizer路径"
    )
    parser.add_argument(
        "--retriever_tokenizer_path",
        type=str,
        default=None,
        help="可选：检索器tokenizer路径，优先级高于tokenizer_path"
    )
    parser.add_argument(
        "--generator_tokenizer_path",
        type=str,
        default=None,
        help="可选：生成器tokenizer路径，优先级高于tokenizer_path"
    )
    parser.add_argument(
        "--vllm_dtype",
        type=str,
        default=VLLM_CONFIG["dtype"],
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM加载模型的dtype"
    )
    parser.add_argument(
        "--lora_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM LoRA权重dtype"
    )
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        help="vLLM LoRA最大rank，需不小于adapter_config.json中的r"
    )

    args = parser.parse_args()

    # 设置默认输出目录
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint_path.rstrip('/'))
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{checkpoint_name}_maj")

    merge_mode_desc = "取并集" if args.retriever_merge_mode == "union" else "majority voting"
    print("=" * 60)
    print("Maj-Voting 模式单个检查点评估")
    print("=" * 60)
    print(f"检查点路径: {args.checkpoint_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"基础模型: {args.base_model}")
    print(f"SQL模型: {args.sql_model}")
    print(f"跳过合并: {args.skip_merge}")
    print(f"强制重新评估: {args.force}")
    print(f"样本数量: {args.num_samples}")
    print(f"采样温度: {args.temperature}")
    print(f"检索合并模式: {args.retriever_merge_mode} ({merge_mode_desc})")
    print(f"添加主键: {not args.no_add_primary_keys}")
    print(f"检索器提示词格式: {args.retriever_prompt_format}")
    print(f"no_think: {args.no_think}")
    print(f"merge_torch_dtype: {args.merge_torch_dtype}")
    print(f"检索器LoRA挂载: {args.retriever_lora_adapter}")
    print(f"生成器LoRA挂载: {args.generator_lora_adapter}")
    print(f"统一tokenizer: {args.tokenizer_path}")
    print(f"检索器tokenizer: {args.retriever_tokenizer_path}")
    print(f"生成器tokenizer: {args.generator_tokenizer_path}")
    print(f"vLLM dtype: {args.vllm_dtype}")
    print(f"LoRA dtype: {args.lora_dtype}")
    print(f"max_lora_rank: {args.max_lora_rank}")
    print(f"检索器最大长度: {RETRIEVER_MAX_TOKENS}")
    print(f"SQL最大长度: {SQL_MAX_TOKENS}")
    print("=" * 60)

    accuracy = evaluate_single_checkpoint_maj(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        sql_model=args.sql_model,
        skip_merge=args.skip_merge,
        skip_if_exists=not args.force,
        num_samples=args.num_samples,
        temperature=args.temperature,
        sql_prompts_file=args.sql_prompts_file,
        retriever_merge_mode=args.retriever_merge_mode,
        add_primary_keys=not args.no_add_primary_keys,
        retriever_prompt_format=args.retriever_prompt_format,
        no_think=args.no_think,
        merge_torch_dtype=args.merge_torch_dtype,
        retriever_lora_adapter=args.retriever_lora_adapter,
        generator_lora_adapter=args.generator_lora_adapter,
        tokenizer_path=args.tokenizer_path,
        retriever_tokenizer_path=args.retriever_tokenizer_path,
        generator_tokenizer_path=args.generator_tokenizer_path,
        vllm_dtype=args.vllm_dtype,
        lora_dtype=args.lora_dtype,
        max_lora_rank=args.max_lora_rank,
    )

    if accuracy is not None:
        print(f"\n{'=' * 60}")
        print(f"评估完成！准确率 (Maj-Voting): {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"{'=' * 60}")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print("评估失败！")
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
