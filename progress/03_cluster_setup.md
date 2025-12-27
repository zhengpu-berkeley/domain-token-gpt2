# Cluster Setup Guide

This document describes how to connect to and work with the Runpod GPU cluster for the domain-token-gpt2 experiment.

---

## Current Pod Information (Updated Dec 27, 2024)

| Field | Value |
|-------|-------|
| **Provider** | [Runpod](https://runpod.io) |
| **Pod Name** | `shaky_ivory_catfish` |
| **Pod ID** | `6wbpka2gwq6hf4` |
| **GPU** | NVIDIA H200 SXM × 4 (564 GB VRAM total!) |
| **RAM** | 1004 GB |
| **vCPU** | 96 cores |
| **Cost** | $14.36/hr |
| **Network Volume** | `zhengpu-storage` (200 GB) |
| **Data Center** | US-NC-1 (North Carolina) |
| **Status** | Running ✅ |

### Previous Pod (Stopped)
| Field | Value |
|-------|-------|
| **Pod Name** | `academic_magenta_dormouse` |
| **Pod ID** | `9ecuj1ypp6kwfu` |
| **GPU** | NVIDIA A40 × 1 |
| **Status** | Stopped (idle disk: $0.01/hr) |

---

## SSH Connection

### Quick Connect (Recommended)

```bash
ssh runpod-domain-token
```

### Manual Connect via Direct TCP (Supports SCP/SFTP)

```bash
ssh root@103.196.86.20 -p 37064 -i ~/.ssh/id_ed25519
```

### Manual Connect via Runpod Proxy (Terminal only)

```bash
ssh 6wbpka2gwq6hf4-64411ca9@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### SSH Config (`~/.ssh/config`)

```sshconfig
# ==============================================
# NEW: 4x H200 SXM Cluster (shaky_ivory_catfish)
# ==============================================

# Direct TCP - USE THIS FOR CURSOR REMOTE-SSH
Host runpod-domain-token
    HostName 103.196.86.20
    User root
    Port 37064
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

# Proxy connection (terminal only, no PTY support for IDE)
Host runpod-domain-token-proxy
    HostName ssh.runpod.io
    User 6wbpka2gwq6hf4-64411ca9
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

### SSH Key Setup (Already Done)

1. Generated ED25519 key: `~/.ssh/id_ed25519`
2. Public key added to Runpod account settings: https://console.runpod.io/user/settings
3. Key added to pod's `~/.ssh/authorized_keys`

---

## Cursor IDE Remote Connection

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Remote-SSH: Connect to Host..."
3. Select `runpod-domain-token`
4. Cursor will open a new window connected to the pod

---

## First-Time Pod Setup

The repo is already cloned at `/workspace/domain-token-gpt2`. To complete setup:

```bash
# Navigate to the repo
cd /workspace/domain-token-gpt2

# Pull latest changes
git pull

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Sync dependencies
uv sync

# Verify GPUs are available (should show 4 H200 GPUs!)
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

## Running Full Experiments (4x H200)

With 4× H200 GPUs (564 GB VRAM), you can run the full-scale experiments:

### Full FineWeb Pretrain (10B tokens)

```bash
# Prepare FineWeb data with mul-token injection
uv run python data/prepare_fineweb_pilot.py \
    --condition mul_tokens \
    --out-dir data/fineweb_full/mul_tokens \
    --target-tokens 10000000000

# Run distributed training on 4 GPUs
torchrun --standalone --nproc_per_node=4 pretrain/train_nanogpt.py \
    --config pretrain/configs/gpt2_124m_full.yaml \
    --data-root data/fineweb_full/mul_tokens \
    --output-dir outputs/pretrain_mul_tokens_full
```

### Compute Budget Estimates (4× H200)

| Stage | Tokens/Steps | Est. Time | Est. Cost |
|-------|--------------|-----------|-----------|
| Pretrain (124M, 10B tokens) | 19K steps | ~2-3 hours | ~$40 |
| SFT (GSM8K train) | 3 epochs | ~20 min | ~$5 |
| GRPO RL | 1000 updates | ~1 hour | ~$15 |
| **Total per condition** | — | ~4-5 hours | ~$60 |
| **Both conditions × 3 seeds** | — | ~30 hours | ~$400 |

---

## Network Volume

A persistent network volume is attached for data that survives pod changes:

| Field | Value |
|-------|-------|
| **Name** | `zhengpu-storage` |
| **Volume ID** | `4aan1ywdr1` |
| **Size** | 200 GB |
| **Data Center** | US-NC-1 |
| **Cost** | ~$14/month |

The volume is mounted and available inside the pod. Use it for:
- Large datasets (FineWeb shards)
- Model checkpoints you want to persist
- Results that should survive pod termination

---

## Pod Management

### Via Runpod Console

- **Dashboard**: https://console.runpod.io/pods
- **Pod Details**: https://console.runpod.io/pods?id=6wbpka2gwq6hf4

### Common Operations

| Action | How |
|--------|-----|
| **Stop Pod** | Console → Pod → ⋮ menu → Stop Pod (saves $14.36/hr!) |
| **Start Pod** | Console → Pod → Start |
| **Restart Pod** | Console → Pod → ⋮ menu → Restart Pod |
| **Jupyter Lab** | https://6wbpka2gwq6hf4-8888.proxy.runpod.net (when running) |

### Cost Management

- **Running rate**: $14.36/hr + $0.019/hr network volume
- **Daily cost (24h)**: ~$345
- **Stopped cost**: $0.006/hr (container disk) + $0.019/hr (network volume)
- **⚠️ IMPORTANT**: Stop the pod when not actively using it!

---

## File Persistence

- `/workspace/` — Persistent storage (survives pod restarts)
- `/runpod-volume/` — Network volume mount (survives pod termination)
- `/root/` — May be reset on pod restart
- **Recommendation**: Keep all project files in `/workspace/`, checkpoints in network volume

---

## Multi-GPU Training Notes

With 4× H200 SXM GPUs:

```bash
# Use torchrun for DDP training
torchrun --standalone --nproc_per_node=4 pretrain/train_nanogpt.py ...

# Check GPU utilization during training
watch -n 1 nvidia-smi

# Monitor training across all GPUs
nvidia-smi dmon -s u
```

### Expected Performance
- H200 SXM: ~3.9 PFLOPS FP8, 4.9 TB/s memory bandwidth
- 4× H200: Should process ~500K-1M tokens/second during training
- 10B token pretrain: ~2-3 hours

---

## Troubleshooting

### SSH Connection Issues

1. **"Permission denied"**: Check that your SSH key is added to pod's `~/.ssh/authorized_keys`
2. **"Connection refused"**: Pod may be stopped; start it from the console
3. **Timeout**: Pod may be initializing; wait 1-2 minutes after starting

### GPU Not Available

```bash
# Check NVIDIA driver and GPUs
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up caches
rm -rf ~/.cache/pip
rm -rf ~/.cache/uv
rm -rf ~/.cache/huggingface

# Move large files to network volume
mv /workspace/outputs /runpod-volume/outputs
ln -s /runpod-volume/outputs /workspace/outputs
```

### Network Volume Not Mounted

```bash
# Check if volume is mounted
ls -la /runpod-volume/

# If not visible, check pod configuration in Runpod console
```

---

## GitHub SSH Setup (Cluster → GitHub)

To enable `git push` and private repo cloning from the cluster, an SSH key was generated on the pod and added to GitHub.

### Steps Performed

1. **Generated SSH key on the cluster:**
   ```bash
   ssh-keygen -t ed25519 -C "runpod-h200-cluster" -f ~/.ssh/id_ed25519 -N ""
   ```

2. **Retrieved the public key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # Output: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE1z5DGrhsnIypx/d6Rol/fCEZBnrYt4JEx8u5RkXZdO runpod-h200-cluster
   ```

3. **Added the key to GitHub:**
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Title: `runpod-h200-cluster`
   - Key: (paste the public key from step 2)
   - Click "Add SSH key"

4. **Added GitHub to known hosts on the cluster:**
   ```bash
   ssh-keyscan github.com >> ~/.ssh/known_hosts
   ```

5. **Configured Git identity on the cluster:**
   ```bash
   git config --global user.name "zhengpu-berkeley"
   git config --global user.email "zhaozhengpu.graduate@gmail.com"
   ```

6. **Cloned the repository:**
   ```bash
   cd /workspace
   git clone git@github.com:zhengpu-berkeley/domain-token-gpt2.git
   ```

### Verifying GitHub SSH Connection

```bash
# Test GitHub SSH access
ssh -T git@github.com
# Expected: "Hi zhengpu-berkeley! You've successfully authenticated..."

# Check remote URLs
cd /workspace/domain-token-gpt2
git remote -v
# Should show: git@github.com:zhengpu-berkeley/domain-token-gpt2.git
```

### Key Details

| Field | Value |
|-------|-------|
| **Key Type** | ED25519 |
| **Key Name** | `runpod-h200-cluster` |
| **Key Fingerprint** | `SHA256:CivAL4riGC8bjkllPC2pkdgFPY8lSy0s/0wVijgi43k` |
| **GitHub Profile** | `zhengpu-berkeley` |

---

## Current State (Ready for Full Experiments!)

✅ **Cluster deployed:** 4× H200 SXM (564 GB VRAM total)  
✅ **SSH access configured:** `ssh runpod-domain-token`  
✅ **GitHub SSH key added:** Push/pull works from cluster  
✅ **Repo cloned:** `/workspace/domain-token-gpt2`  
✅ **Network volume attached:** 200 GB persistent storage  
✅ **Old A40 pod stopped:** Saving $0.40/hr  

### Next Steps

1. **SSH into the cluster:**
   ```bash
   ssh runpod-domain-token
   ```

2. **Navigate to the repo:**
   ```bash
   cd /workspace/domain-token-gpt2
   ```

3. **Set up Python environment:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   uv sync
   ```

4. **Run full experiments!** (See sections above for commands)

---

## Related Documentation

- `progress/01_research_spec.md` — Full experiment specification
- `progress/02_init_handoff.md` — Implementation status and repo structure
- `progress/04_pilot_smoketest.md` — Pilot results from A40 cluster
- `README.md` — Project overview

---

*Last updated: December 27, 2024*
