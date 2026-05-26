"""
合并LoRA权重到基础模型
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_MODEL_PATH


def resolve_torch_dtype(dtype: str):
    """将命令行dtype名称转换为torch dtype。"""
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"不支持的torch_dtype: {dtype}")
    return dtype_map[dtype]


def merge_lora(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: str = "float16"
):
    """
    合并LoRA adapter到基础模型

    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA adapter路径
        output_path: 输出路径
        device_map: 设备映射
        torch_dtype: 基础模型加载和merge使用的dtype
    """
    print(f"加载基础模型: {base_model_path}")
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    print(f"merge torch_dtype: {torch_dtype}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print(f"加载LoRA adapter: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("合并LoRA权重到基础模型...")
    model = model.merge_and_unload()

    print(f"保存合并后的模型到: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)

    # 保存tokenizer
    print("保存tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("完成！")


def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=BASE_MODEL_PATH,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="LoRA adapter路径（checkpoint目录）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后模型的输出路径"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="设备映射"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="加载基础模型并合并LoRA时使用的dtype（默认保持原有float16；FP32 merge传float32）"
    )

    args = parser.parse_args()

    merge_lora(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_path=args.output_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype
    )


if __name__ == "__main__":
    main()
