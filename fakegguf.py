#!/usr/bin/env python3
"""FakeGGUF: generate synthetic LLaMA-style GGUF files for testing."""
import argparse
import math
from pathlib import Path
import re
import numpy as np


def parse_params(value):
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([KMBT]?)\s*', value.upper().replace('_', ''))
    if not match:
        raise argparse.ArgumentTypeError('Use a positive count such as 125M, 1B or 7000000')
    result = int(float(match.group(1)) * {'': 1, 'K': 10**3, 'M': 10**6, 'B': 10**9, 'T': 10**12}[match.group(2)])
    if result <= 0:
        raise argparse.ArgumentTypeError('Parameter count must be positive')
    return result


def estimate_params(layers, hidden, intermediate, vocab, kv_heads, heads, tied):
    head_dim = hidden // heads
    attention = 2 * hidden * hidden + 2 * hidden * (kv_heads * head_dim)
    return vocab * hidden * (1 if tied else 2) + layers * (attention + 3 * hidden * intermediate + 2 * hidden) + hidden


def auto_config(target, vocab, heads, kv_heads, tied):
    best = None
    best_error = float('inf')
    step = math.lcm(64, heads)
    for layers in range(1, 129):
        for hidden in range(step, 8193, step):
            intermediate = math.ceil(hidden * 8 / 3 / 64) * 64
            count = estimate_params(layers, hidden, intermediate, vocab, kv_heads, heads, tied)
            error = abs(count - target)
            if error < best_error:
                best_error, best = error, (layers, hidden, intermediate, count)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--output', help='Output GGUF file path (overrides --output-dir and --filename)')
    p.add_argument('--output-dir', default='.')
    p.add_argument('--filename', help='Filename within output directory')
    p.add_argument('--name', default='Synthetic Random LLaMA')
    p.add_argument('--author', default='FakeGGUF')
    p.add_argument('--description', default='Synthetic model weights for testing only')
    p.add_argument('--version', default='1.0')
    p.add_argument('--params', type=parse_params, default=parse_params('125M'))
    p.add_argument('--layers', type=int)
    p.add_argument('--hidden-size', type=int)
    p.add_argument('--intermediate-size', type=int)
    p.add_argument('--vocab-size', type=int, default=32000)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--kv-heads', type=int)
    p.add_argument('--context-length', type=int, default=2048)
    p.add_argument('--rope-freq-base', type=float, default=10000.0)
    p.add_argument('--rms-eps', type=float, default=1e-5)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--std', type=float, default=0.02)
    p.add_argument('--fill', choices=['random', 'zeros'], default='random')
    p.add_argument('--tie-embeddings', action='store_true', help='Omit output.weight; use token embeddings for logits')
    p.add_argument('--bos-id', type=int, default=1)
    p.add_argument('--eos-id', type=int, default=2)
    p.add_argument('--unk-id', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true', help='Overwrite existing output file')
    args = p.parse_args()
    kv = args.kv_heads if args.kv_heads is not None else args.heads
    if any(x <= 0 for x in (args.vocab_size, args.heads, kv, args.context_length)):
        p.error('Vocab size, head counts and context length must be positive')
    if args.heads % kv:
        p.error('--heads must be divisible by --kv-heads')
    if not (math.isfinite(args.std) and args.std >= 0 and math.isfinite(args.rope_freq_base) and args.rope_freq_base > 0 and math.isfinite(args.rms_eps) and args.rms_eps > 0):
        p.error('Invalid --std, --rope-freq-base or --rms-eps')
    if any(not (0 <= i < args.vocab_size) for i in (args.bos_id, args.eos_id, args.unk_id)):
        p.error('Special token IDs must be within the vocabulary')
    if any(x is not None for x in (args.layers, args.hidden_size, args.intermediate_size)):
        if not all(x is not None for x in (args.layers, args.hidden_size, args.intermediate_size)):
            p.error('Specify --layers, --hidden-size and --intermediate-size together')
        layers, hidden, intermediate = args.layers, args.hidden_size, args.intermediate_size
        if min(layers, hidden, intermediate) <= 0 or hidden % args.heads:
            p.error('Dimensions must be positive and hidden size divisible by head count')
        actual = estimate_params(layers, hidden, intermediate, args.vocab_size, kv, args.heads, args.tie_embeddings)
    else:
        layers, hidden, intermediate, actual = auto_config(args.params, args.vocab_size, args.heads, kv, args.tie_embeddings)
    safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '-', args.name).strip('.-') or 'synthetic'
    filename = args.filename or f'{safe_name}-{actual / 1e6:.1f}M-{args.dtype}.gguf'
    output = Path(args.output) if args.output else Path(args.output_dir) / filename
    if output.suffix.lower() != '.gguf':
        output = output.with_suffix('.gguf')
    print(f'Model: {args.name}\nOutput: {output}\nParameters: {actual:,} (target {args.params:,})')
    print(f'Layers: {layers}; hidden: {hidden}; intermediate: {intermediate}; heads: {args.heads}; KV heads: {kv}')
    print(f'Estimated raw weight bytes: {actual * np.dtype(args.dtype).itemsize:,} (excluding GGUF overhead)')
    if args.dry_run:
        return
    if output.exists() and not args.force:
        p.error(f'{output} already exists (use --force to overwrite)')
    try:
        from gguf import GGUFWriter
    except ImportError:
        p.error('Missing dependency: pip install gguf numpy')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), 'llama')
    writer.add_name(args.name)
    writer.add_author(args.author)
    writer.add_description(f'{args.description} (version {args.version})')
    writer.add_block_count(layers)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_head_count(args.heads)
    writer.add_head_count_kv(kv)
    writer.add_context_length(args.context_length)
    writer.add_rope_freq_base(args.rope_freq_base)
    writer.add_layer_norm_rms_eps(args.rms_eps)
    writer.add_tokenizer_model('llama')
    writer.add_token_list([f'token_{i}' for i in range(args.vocab_size)])
    writer.add_tokenizer_pre('default')
    writer.add_bos_token_id(args.bos_id)
    writer.add_eos_token_id(args.eos_id)
    writer.add_unk_token_id(args.unk_id)
    rng = np.random.default_rng(args.seed)
    dtype = np.dtype(args.dtype)

    def add(name, shape, ones=False):
        if ones:
            data = np.ones(shape, dtype=dtype)
        elif args.fill == 'zeros':
            data = np.zeros(shape, dtype=dtype)
        else:
            data = rng.normal(0, args.std, shape).astype(dtype)
        writer.add_tensor(name, data)

    add('token_embd.weight', (args.vocab_size, hidden))
    kv_dim = kv * (hidden // args.heads)
    for i in range(layers):
        prefix = f'blk.{i}'
        add(f'{prefix}.attn_norm.weight', (hidden,), True)
        add(f'{prefix}.attn_q.weight', (hidden, hidden))
        add(f'{prefix}.attn_k.weight', (kv_dim, hidden))
        add(f'{prefix}.attn_v.weight', (kv_dim, hidden))
        add(f'{prefix}.attn_output.weight', (hidden, hidden))
        add(f'{prefix}.ffn_norm.weight', (hidden,), True)
        add(f'{prefix}.ffn_gate.weight', (intermediate, hidden))
        add(f'{prefix}.ffn_up.weight', (intermediate, hidden))
        add(f'{prefix}.ffn_down.weight', (hidden, intermediate))
        print(f'Layer {i + 1}/{layers}', end='\r', flush=True)
    add('output_norm.weight', (hidden,), True)
    if not args.tie_embeddings:
        add('output.weight', (args.vocab_size, hidden))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f'\nSaved {output}')


if __name__ == '__main__':
    main()
