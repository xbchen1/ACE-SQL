"""One full-parameter vLLM inference implementation for both policy roles."""
import argparse
import json
from pathlib import Path

from ace_sql.training.checkpoints import require_hf_checkpoint
from .config import VLLM_CONFIG, RETRIEVER_INFER_CONFIG, SQL_INFER_CONFIG
from .parsing import parse_retriever_response, parse_sql_response


def format_result(data, responses, role):
    if role == 'retriever':
        if len(responses) == 1:
            return {**data, 'retriever_response': responses[0],
                    'parsed_columns': parse_retriever_response(responses[0])}
        return {**data, 'retriever_responses': responses,
                'parsed_columns_list': [parse_retriever_response(s) for s in responses]}
    return {**data, 'responses': responses, 'pred_sqls': [parse_sql_response(s) for s in responses]}


def main(role, argv=None):
    defaults = RETRIEVER_INFER_CONFIG if role == 'retriever' else SQL_INFER_CONFIG
    parser = argparse.ArgumentParser(description=f'{role} inference with a complete HF checkpoint')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--dtype', default=VLLM_CONFIG['dtype'], choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--tensor_parallel_size', type=int, default=VLLM_CONFIG['tensor_parallel_size'])
    parser.add_argument('--temperature', type=float, default=defaults['temperature'])
    parser.add_argument('--max_tokens', type=int, default=defaults['max_tokens'])
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--no_think', action='store_true')
    args = parser.parse_args(argv)
    if args.num_samples < 1 or args.tensor_parallel_size < 1 or args.max_tokens < 1 or args.temperature < 0:
        parser.error('Sample count, GPU count and token limit must be positive; temperature nonnegative')
    model = require_hf_checkpoint(args.model_path)
    with open(args.input_file, encoding='utf-8') as stream:
        records = json.load(stream)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer_path = args.tokenizer_path or str(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=VLLM_CONFIG['trust_remote_code'])
    prompts = []
    for record in records:
        prompt = record.get('input_seq') or record.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('Every inference record must have a nonempty input_seq or prompt')
        prompts.append(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt + (' /no_think' if args.no_think else '')}],
            tokenize=False, add_generation_prompt=True))
    options = {**VLLM_CONFIG, 'model': str(model), 'tokenizer': tokenizer_path,
               'dtype': args.dtype, 'tensor_parallel_size': args.tensor_parallel_size}
    llm = LLM(**options)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                              n=args.num_samples, stop_token_ids=defaults['stop_token_ids'])
    outputs = llm.generate(prompts, sampling)
    results = [format_result(record, [sample.text for sample in output.outputs], role)
               for record, output in zip(records, outputs, strict=True)]
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {len(results)} {role} results to {output}')
