# Cluster-Side Debugging Prompt

I've completed the full 10B token experiment comparing baseline vs mul_tokens conditions on a 4× H200 cluster. Both conditions finished training, but the results show no improvement from mul-tokens (GSM8K: 0.23% baseline vs 0.15% mul_tokens, arithmetic probes identical at 0.71%). More concerning, the baseline model's performance is extremely low, which suggests there may be an issue with the training pipeline, model export, or evaluation setup rather than the tokenization intervention itself.

I've deployed a cheaper RTX 4090 pod (`reasonable_crimson_hookworm`, ID: `9d5kbgd6hpaqd6`) for debugging. The pod is accessible via `ssh runpod-debug` and has the network volume `zhengpu-storage` attached with all previous experiment data. I need you to SSH into this pod, set up the environment, and run HellaSwag evaluation on the baseline model to verify we're getting reasonable scores (a 124M GPT-2 should achieve ~30-40% on HellaSwag). This will help determine if the issue is with the model itself or specific to GSM8K/arithmetic tasks.

The baseline model checkpoints should be in `/workspace/domain-token-gpt2/outputs/hf_baseline_10b/` or on the network volume. Use the vendored `third_party/build-nanogpt/hellaswag.py` script or set up a proper evaluation harness. Report the HellaSwag score, and if it's reasonable, we'll investigate why GSM8K performance is so low. If HellaSwag is also very low, we'll need to debug the training/export pipeline. Clone the repo if needed, install dependencies with `uv sync`, and document your findings.

