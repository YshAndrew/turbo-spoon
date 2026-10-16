# RTX 5090 部署 Qwen3.8-27B（vLLM）

> 单卡 5090（32GB，Blackwell）| 2026-10 | Docker / Windows(WSL2) 两套方案，命令可直接复制。

## 结论（就三条）
- BF16(67G)/FP8(38G) 放不进 32GB → 用 **NVFP4**：`Inferact/Qwen3.8-27B-NVFP4`（约 16G）
- 单卡 5090 **必加 `--enforce-eager`**，否则 CUDA Graph 捕获 OOM
- 版本：驱动 ≥ 570、vLLM ≥ 0.17、CUDA 12.8+（Blackwell 硬性要求）、`transformers>=5.8.0`

## 1. 下载模型
```bash
pip install -U "huggingface_hub[cli]"
hf download Inferact/Qwen3.8-27B-NVFP4 --local-dir /models/Qwen3.8-27B-NVFP4
# 国内慢：先 export HF_ENDPOINT=https://hf-mirror.com 再执行；或 modelscope download --model Qwen/Qwen3.8-27B --local_dir /models/Qwen3.8-27B
```

## 2. Docker 方案
```bash
# Linux 原生需先装 NVIDIA Container Toolkit（Windows 用 Docker Desktop+WSL2 跳过）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker run -d --name qwen38 --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_ENDPOINT=https://hf-mirror.com \
  vllm/vllm-openai:latest \
  --model Inferact/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 --max-model-len 32768 --kv-cache-dtype fp8 \
  --enforce-eager --reasoning-parser qwen3
```
日志出现 `INFO: Application startup complete.` 即成功；看日志 `docker logs -f qwen38`。

## 3. Windows 方案（WSL2 + pip）
```powershell
# PowerShell(管理员)：wsl --install 后重启；Windows 装 NVIDIA 驱动(≥570)，勿在 WSL 内装 Linux 驱动
```
```bash
# 以下在 WSL 内执行
nvidia-smi   # 看到 5090 即 GPU 透传成功
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv venv --python 3.12 --seed --managed-python && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
uv pip install -U "transformers>=5.8.0"

vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b --tensor-parallel-size 1 --max-model-len 32768 \
  --kv-cache-dtype fp8 --enforce-eager --reasoning-parser qwen3
```

## 4. 验证
```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":512}'
```

## 5. 关键坑（必读）
- 不加 `--enforce-eager` → 启动即 OOM；调 `--gpu-memory-utilization` 无效
- 不跑 `vllm download-kernels` → Blackwell 启动时现场 JIT 编译极慢
- 不加 `--reasoning-parser qwen3` → `<think>` 思考块会进 content
- `reasoning_effort` 只支持 `xhigh/medium/low`，不支持 `high`
- 单卡默认 32K 上下文；要更长：加 `--language-model-only --max-num-seqs 8` 扩 KV 池后再调大 `--max-model-len`
- 提速加 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`；关思考加 `--default-chat-template-kwargs '{"enable_thinking": false}'`
- 双卡：`unsloth/Qwen3.8-27B-NVFP4` + `--tensor-parallel-size 2 --max-model-len 262144`

## 6. GGUF 备选（vLLM 未验证，官方验证环境是 llama.cpp/Ollama）
```bash
uv pip install vllm-gguf-plugin
vllm serve ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S --tokenizer Qwen/Qwen3.8-27B \
  --max-model-len 32768 --kv-cache-dtype fp8 --enforce-eager --reasoning-parser qwen3
```

## 参考
recipes.vllm.ai/Qwen/Qwen3.8-27B · docs.vllm.ai/en/latest/getting_started/installation/gpu/ · huggingface.co/Inferact/Qwen3.8-27B-NVFP4
