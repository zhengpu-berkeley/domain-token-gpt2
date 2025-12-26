# Upstream Provenance

This directory contains a vendored snapshot of [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt).

## Source
- **Repository:** https://github.com/karpathy/build-nanogpt
- **Commit:** `6104ab1b53920f6e2159749676073ff7d815c1fa`
- **Date vendored:** 2024-12-26
- **License:** MIT

## Files included
- `train_gpt2.py` - Main training script with GPT model definition
- `fineweb.py` - FineWeb dataset preparation script
- `hellaswag.py` - HellaSwag evaluation script
- `README_UPSTREAM.md` - Original upstream README

## Files excluded
- `input.txt` - Large text file (1.1MB), not needed
- `play.ipynb` - Jupyter notebook, not needed
- `.git/` - Git history

## Modifications
This section tracks modifications made to the vendored code.

### train_gpt2.py
- No modifications yet. Our wrapper (`pretrain/train.py`) imports/extends this.

## Usage in this project
We use the GPT model architecture and training utilities from this codebase.
Our wrapper scripts in `pretrain/` extend this to:
- Support configurable vocab_size (for mul-token experiments)
- Use our custom data loaders with different tokenizers
- Read config from YAML files

## License

MIT License (see https://github.com/karpathy/build-nanogpt for full text)

Copyright (c) Andrej Karpathy

