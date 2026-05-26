"""
SQL推理脚本
使用vLLM进行SQL生成
输出格式与 evaluate_bird.py 兼容
"""

import argparse
import json
import re
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import SQL_MODEL_PATH, VLLM_CONFIG, SQL_INFER_CONFIG


def patch_vllm_lru_touch_for_cachetools_6():
    """兼容 cachetools>=6 中 LRUCache 私有方法从 __update 改为 __touch。"""
    try:
        import vllm.utils as vllm_utils
    except Exception:
        return

    probe = vllm_utils.LRUCache(1)
    if hasattr(probe, "_LRUCache__update"):
        return
    if not hasattr(probe, "_LRUCache__touch"):
        return

    def touch(self, key):
        self._LRUCache__touch(key)  # type: ignore[attr-defined]

    vllm_utils.LRUCache.touch = touch


patch_vllm_lru_touch_for_cachetools_6()


def parse_sql_response(response: str) -> str:
    """
    解析SQL响应，提取SQL语句

    Args:
        response: 模型生成的响应文本

    Returns:
        解析后的SQL语句
    """
    if not response:
        return ""

    # 方法1：提取```sql代码块
    matches = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # 方法2：提取普通代码块
    matches = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if matches:
        sql_candidate = matches[-1].strip()
        if any(keyword in sql_candidate.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return sql_candidate

    # 方法3：查找SELECT语句
    if 'SELECT' in response.upper():
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            if 'SELECT' in line.upper():
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if ';' in line:
                    break
        if sql_lines:
            return '\n'.join(sql_lines).strip()

    # 方法4：返回原始响应
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="SQL推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default=SQL_MODEL_PATH,
        help="SQL生成模型路径（合并后的模型，或 vLLM LoRA 挂载模式下的基础模型）"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer路径（默认使用model_path；vLLM LoRA挂载时通常使用基础模型路径）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=VLLM_CONFIG["dtype"],
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM加载模型的dtype"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="可选：vLLM运行时挂载的LoRA adapter路径"
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default="generator",
        help="vLLM LoRA adapter名称"
    )
    parser.add_argument(
        "--lora_int_id",
        type=int,
        default=2,
        help="vLLM LoRA adapter整数ID"
    )
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        help="vLLM LoRA最大rank，需不小于adapter_config.json中的r"
    )
    parser.add_argument(
        "--lora_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM LoRA权重dtype"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入文件路径（SQL提示词）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=VLLM_CONFIG["tensor_parallel_size"],
        help="张量并行大小"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SQL_INFER_CONFIG["temperature"],
        help="采样温度"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=SQL_INFER_CONFIG["max_tokens"],
        help="最大生成token数"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="每个样本生成的数量 (用于maj-voting模式)"
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="在prompt末尾追加 /no_think 标志"
    )

    args = parser.parse_args()
    print(f"配置: {args}")

    if args.lora_adapter_path and not Path(args.lora_adapter_path).is_dir():
        raise FileNotFoundError(f"LoRA adapter路径不存在: {args.lora_adapter_path}")

    # 加载输入数据
    print(f"加载输入数据: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 加载tokenizer
    tokenizer_path = args.tokenizer_path or args.model_path
    print(f"加载tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 确定stop token
    stop_token_ids = SQL_INFER_CONFIG["stop_token_ids"]
    print(f"stop_token_ids: {stop_token_ids}")

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        stop_token_ids=stop_token_ids
    )

    # 初始化LLM
    print(f"初始化LLM: {args.model_path}")
    llm_kwargs = {
        "model": args.model_path,
        "tokenizer": tokenizer_path,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": VLLM_CONFIG["max_model_len"],
        "gpu_memory_utilization": VLLM_CONFIG["gpu_memory_utilization"],
        "swap_space": VLLM_CONFIG["swap_space"],
        "enforce_eager": VLLM_CONFIG["enforce_eager"],
        "disable_custom_all_reduce": VLLM_CONFIG["disable_custom_all_reduce"],
        "trust_remote_code": VLLM_CONFIG["trust_remote_code"],
        "max_num_seqs": VLLM_CONFIG.get("max_num_seqs", 256),
    }
    lora_request = None
    if args.lora_adapter_path:
        print(f"启用vLLM LoRA挂载: {args.lora_adapter_path}")
        llm_kwargs.update({
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": args.max_lora_rank,
            "lora_dtype": args.lora_dtype,
        })
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=args.lora_int_id,
            lora_path=args.lora_adapter_path,
        )
    llm = LLM(**llm_kwargs)

    # 构建chat prompts
    print("构建prompts...")
    chat_prompts = []
    for data in input_data:
        prompt = data.get("input_seq") or data.get("prompt")
        if args.no_think:
            prompt = prompt + " /no_think"
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        chat_prompts.append(chat_prompt)

    # 推理
    print(f"开始推理，共 {len(chat_prompts)} 个样本...")
    outputs = llm.generate(chat_prompts, sampling_params, lora_request=lora_request)

    # 处理结果（与 evaluate_bird.py 格式兼容）
    results = []
    for data, output in zip(input_data, outputs):
        if args.num_samples == 1:
            # 单样本模式（原有逻辑）
            response = output.outputs[0].text
            pred_sql = parse_sql_response(response)
            responses = [response]
            pred_sqls = [pred_sql]
        else:
            # 多样本模式（maj-voting）
            responses = [out.text for out in output.outputs]
            pred_sqls = [parse_sql_response(resp) for resp in responses]

        result = {
            "question_id": data.get("question_id"),
            "db_id": data.get("db_id"),
            "question": data.get("question"),
            "evidence": data.get("evidence", ""),
            "SQL": data.get("SQL"),
            "difficulty": data.get("difficulty", ""),
            # evaluate_bird.py 需要的字段
            "responses": responses,
            "pred_sqls": pred_sqls,
            # 额外信息
            "input_seq": data.get("input_seq"),
            "selected_schema": data.get("selected_schema", ""),
            "retriever_response": data.get("retriever_response", "")
        }
        results.append(result)

    # 保存结果
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"推理完成！结果保存到: {args.output_file}")
    print(f"总样本数: {len(results)}")


if __name__ == "__main__":
    main()
