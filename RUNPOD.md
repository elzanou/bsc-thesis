# RunPod Inference Guide

## 0. Test locally before uploading (optional)

```bash
# Validate pipeline without GPU
python scripts/inference.py --task all --provider audio_flamingo --dry-run
```

## 1. Set pod connection variables

IP and port change each session. Fetch them automatically via the RunPod API:

```bash
# One-time setup: add these to ~/.bashrc
export RUNPOD_API_KEY=your_key        # from runpod.io → Settings → API Keys
export RUNPOD_POD_ID=your_pod_id      # from the pod URL in the dashboard

# Run this each session to set POD_IP and POD_PORT:
# pip install runpod  (one-time)
eval $(python3 - <<'EOF'
import runpod, os
runpod.api_key = os.environ["RUNPOD_API_KEY"]
pod = runpod.get_pod(os.environ["RUNPOD_POD_ID"])
ports = pod.get("runtime", {}).get("ports", [])
ssh = next((p for p in ports if p["privatePort"] == 22 and p["isIpPublic"]), None)
if ssh:
    print(f"export POD_IP={ssh['ip']}")
    print(f"export POD_PORT={ssh['publicPort']}")
else:
    print("echo 'Pod not running or SSH port not found'")
EOF
)
echo "Connecting to $POD_IP:$POD_PORT"
```

## 2. Upload project and raw data (from local machine, first time only)

We upload raw audios + the local YouTube cache (~265MB total) instead of processed data (~700MB).
Preprocessing runs on the pod after upload. Binary task is excluded — not needed.

```bash
cd /path/to/music-evalkit  # your local clone

# Create tarball with raw audios and YouTube cache
tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='data/results' \
    -czf /tmp/music-evalkit-raw.tar.gz \
    src scripts pyproject.toml config.yaml \
    data/audios data/cache data/mcq_updated.csv data/Music_samples.xlsx

# Upload to RunPod
scp -P $POD_PORT -i ~/.ssh/id_ed25519 \
  /tmp/music-evalkit-raw.tar.gz \
  root@$POD_IP:/workspace/

# Extract on RunPod
ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519 \
  "cd /workspace && rm -rf music-evalkit && mkdir -p music-evalkit && tar --no-same-owner -xzf music-evalkit-raw.tar.gz -C music-evalkit"
```

## 3. SSH into RunPod and run setup

```bash
ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519
cd /workspace/music-evalkit
bash scripts/runpod_setup.sh
```

## 4. Preprocess data (on the pod, first time only)

The YouTube cache is already included so no re-downloading needed.

```bash
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

python scripts/preprocess.py --task mcq
python scripts/preprocess.py --task open_ended
python scripts/preprocess.py --task pairwise
```

## 5. Run inference

```bash
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# Audio Flamingo 3
python scripts/inference.py --task all --provider audio_flamingo

# Music Flamingo
python scripts/inference.py --task all --provider music_flamingo

# Qwen2.5-Omni (local)
python scripts/inference.py --task all --provider qwen_local
```

## 6. Download results (from local machine)

```bash
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  root@$POD_IP:/workspace/music-evalkit/data/results/ \
  /path/to/music-evalkit/runpod_results/
```

## 7. Stop pod (preserve data)

**Stop** the pod from the RunPod dashboard — do NOT Terminate.

- **Stop** = pod paused, `/workspace` disk is kept, billing pauses (only storage cost ~$0.07/GB/month continues)
- **Terminate** = pod deleted, disk wiped

Everything in `/workspace` (uploaded data, model weights cached in `~/.cache/huggingface`, results) persists across stop/start cycles. Upload and preprocess only once.

## Troubleshooting

### "accelerate" not found
```bash
pip install accelerate
```

### "uv command not found" after activating venv
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### flash-attn build fails
```bash
pip install setuptools wheel
pip install flash-attn --no-build-isolation
```
