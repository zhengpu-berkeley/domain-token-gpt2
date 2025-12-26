# Cluster Setup Guide

This document describes how to connect to and work with the Runpod GPU cluster for the domain-token-gpt2 experiment.

---

## Current Pod Information

| Field | Value |
|-------|-------|
| **Provider** | [Runpod](https://runpod.io) |
| **Pod Name** | `academic_magenta_dormouse` |
| **Pod ID** | `9ecuj1ypp6kwfu` |
| **GPU** | NVIDIA A40 × 1 (48GB VRAM) |
| **CPUs** | 96 cores |
| **Cost** | $0.41/hr |
| **Status** | Running ✅ |

---

## SSH Connection

### Quick Connect (Recommended)

```bash
ssh runpod-domain-token
```

### Manual Connect via Runpod Proxy

```bash
ssh 9ecuj1ypp6kwfu-64411be9@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### SSH Config (`~/.ssh/config`)

```sshconfig
Host runpod-domain-token
    HostName ssh.runpod.io
    User 9ecuj1ypp6kwfu-64411be9
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

# Alternative: Direct TCP (may require pod restart after adding SSH key)
Host runpod-domain-token-direct
    HostName 69.30.85.150
    User root
    Port 22154
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

### SSH Key Setup (Already Done)

1. Generated ED25519 key: `~/.ssh/id_ed25519`
2. Public key added to Runpod account settings: https://console.runpod.io/user/settings
3. Pod restarted to apply key

---

## Cursor IDE Remote Connection

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Remote-SSH: Connect to Host..."
3. Select `runpod-domain-token`
4. Cursor will open a new window connected to the pod

---

## First-Time Pod Setup

Once connected to the pod, run these commands to set up the environment:

```bash
# Clone the repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/domain-token-gpt2.git
cd domain-token-gpt2

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Sync dependencies
uv sync

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Running Experiments

### Smoke Test (Verify Setup)

```bash
cd /workspace/domain-token-gpt2
bash scripts/run_smoke.sh
```

### Full Training Run

See `research_spec.md` and `status_handoff.md` for the full experimental protocol.

```bash
# Example: Run baseline condition
uv run python data/prepare_text.py --condition baseline
uv run python data/tokenize_to_bin.py data/raw/baseline.txt --out data/processed/baseline
uv run python pretrain/train.py --config pretrain/configs/small.yaml --data data/processed/baseline

# Example: Run mul-token condition
uv run python data/prepare_text.py --condition mul
uv run python data/tokenize_to_bin.py data/raw/mul.txt --out data/processed/mul --inject-mul
uv run python pretrain/train.py --config pretrain/configs/small.yaml --data data/processed/mul
```

---

## Pod Management

### Via Runpod Console

- **Dashboard**: https://console.runpod.io/pods
- **Pod Details**: https://console.runpod.io/pods?id=9ecuj1ypp6kwfu

### Common Operations

| Action | How |
|--------|-----|
| **Stop Pod** | Console → Pod → Stop (saves money when not in use) |
| **Start Pod** | Console → Pod → Start |
| **Restart Pod** | Console → Pod → ⋮ menu → Restart Pod |
| **Jupyter Lab** | https://9ecuj1ypp6kwfu-8888.proxy.runpod.net (when running) |

### Cost Management

- **Current rate**: $0.41/hr
- **Daily cost (24h)**: ~$9.84
- **Tip**: Stop the pod when not actively using it!

---

## File Persistence

- `/workspace/` — Persistent storage (survives pod restarts)
- `/root/` — May be reset on pod restart
- **Recommendation**: Keep all project files in `/workspace/`

---

## Troubleshooting

### SSH Connection Issues

1. **"Permission denied"**: Check that your SSH key is added to Runpod settings and pod was restarted
2. **"Connection refused"**: Pod may be stopped; start it from the console
3. **Timeout**: Pod may be initializing; wait 1-2 minutes after starting

### GPU Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up
rm -rf ~/.cache/pip
rm -rf ~/.cache/uv
```

---

## Related Documentation

- `research_spec.md` — Full experiment specification
- `status_handoff.md` — Current implementation status and repo structure
- `README.md` — Project overview

---

*Last updated: December 26, 2024*

