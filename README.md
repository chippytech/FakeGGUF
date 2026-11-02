# FakeGGUF

Generate synthetic LLaMA-style GGUF files with configurable metadata, architecture and weights.

**Warning:** FakeGGUF produces random or zero-filled model weights, not trained models. The generated vocabulary is synthetic and is not a functional real-world tokenizer. Some inference runtimes may reject the tokenizer or other metadata; this tool is intended for loader and infrastructure experiments, not real inference.

## Installation

```bash
pip install -r requirements.txt
```

## Examples

```bash
python fakegguf.py --name "Test Model" --params 125M
python fakegguf.py --name "My Model" --filename my-model.gguf --output-dir ./models --author "Example" --description "GGUF test file"
python fakegguf.py --output custom.gguf --params 1B --dtype float16 --seed 123 --force
python fakegguf.py --layers 12 --hidden-size 768 --intermediate-size 2048 --heads 12 --kv-heads 4 --dry-run
python fakegguf.py --params 7B --dry-run
```

Use `python fakegguf.py --help` to see all available options. **Large configurations can require many gigabytes of RAM and disk space.** Parameter estimates count tensor elements, not GGUF metadata or padding. The automatic architecture search finds the closest parameter count within its search grid; it does not guarantee an exact match.

## License

MIT
