"""
检索器推理脚本
使用vLLM进行批量推理
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

from config import VLLM_CONFIG, RETRIEVER_INFER_CONFIG


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


def parse_retriever_response(response: str) -> str:
    """
    解析检索器响应，提取列列表或JSON格式

    Args:
        response: 模型生成的响应文本

    Returns:
        解析后的列列表字符串或JSON字符串
    """
    # 方法1：提取 ```json ... ``` 代码块中的JSON对象
    json_block_pattern = re.compile(r'```json\s*\n?(.*?)\n?```', re.DOTALL)
    match = json_block_pattern.search(response)
    if match:
        return match.group(1).strip()

    # 方法2：提取代码块中的列表
    code_block_pattern = re.compile(r'```(?:python|sql|text)?\s*\n?\s*(\[[^\]]*\])\s*\n?```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        return match.group(1)

    # 方法3：普通列表格式
    list_pattern = re.compile(r'\[([^\]]+)\]')
    match = list_pattern.search(response)
    if match:
        return f"[{match.group(1)}]"

    # 方法4：返回原始响应（让后续处理来解析）
    return response


def main():
    parser = argparse.ArgumentParser(description="检索器推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（合并后的模型，或 vLLM LoRA 挂载模式下的基础模型）"
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
        default="retriever",
        help="vLLM LoRA adapter名称"
    )
    parser.add_argument(
        "--lora_int_id",
        type=int,
        default=1,
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
        help="输入文件路径（检索器提示词）"
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
        default=RETRIEVER_INFER_CONFIG["temperature"],
        help="采样温度"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=RETRIEVER_INFER_CONFIG["max_tokens"],
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
    stop_token_ids = RETRIEVER_INFER_CONFIG["stop_token_ids"]
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

    # 处理结果
    results = []
    for data, output in zip(input_data, outputs):
        if args.num_samples == 1:
            # 单样本模式（原有逻辑）
            response = output.outputs[0].text
            parsed_response = parse_retriever_response(response)
            result = {
                **data,
                "retriever_response": response,
                "parsed_columns": parsed_response
            }
        else:
            # 多样本模式（maj-voting）
            responses = [out.text for out in output.outputs]
            parsed_responses = [parse_retriever_response(resp) for resp in responses]
            result = {
                **data,
                "retriever_responses": responses,
                "parsed_columns_list": parsed_responses
            }
        results.append(result)

    # 保存结果
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"推理完成！结果保存到: {args.output_file}")


if __name__ == "__main__":
    main()
