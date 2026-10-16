# RTX 5090 部署 Qwen3.8-27B（vLLM）完整指南

> 适用：学校电脑 RTX 5090（单卡 32GB 显存，Blackwell SM120）
> 更新：2026-10-16 · 依据 vLLM 官方 Recipe（2026-10-08 更新）与生产环境实测整理
> 方案：① Docker 部署（Linux / Windows WSL2 通用）② Windows 原生路径（WSL2 + pip 安装）

---

## 0. 先说结论（为什么必须这么配）

| 项目 | 结论 |
|---|---|
| 模型 | **Qwen3.8-27B**：27B 稠密、原生多模态（图像/视频理解）、原生 262,144 上下文（可经 YaRN 扩到约 1M）、默认开启思考模式、自带 MTP 草稿头、Apache-2.0 |
| 单卡放不下 | BF16 权重 67GB、FP8 权重 38GB，**都放不进 32GB 显存** |
| 单卡能跑 | **NVFP4（W4A4 4bit）** 权重约 16GB，是单卡 5090 的正解；INT4（W4A16）24GB 也可用 |
| 关键参数 | 单卡 5090 必须加 **`--enforce-eager`**。不加会在 CUDA Graph 捕获阶段直接 OOM（启动即崩），调 `--gpu-memory-utilization` 没用 |
| 版本要求 | vLLM ≥ 0.17（实测 0.26.x / 0.27.1 均可）、PyTorch 需 CUDA 12.8+（Blackwell 硬性要求，vLLM 默认编 cu129）、`transformers>=5.8.0`、NVIDIA 驱动 ≥ 570（推荐最新 580.x） |
| 单卡上下文 | 默认配 32K 上下文最稳（KV 池约 91K tokens）；可通过 `--language-model-only`、`--max-num-seqs 8` 等把 KV 池扩到 15 万 tokens 再逐步加大 `--max-model-len` |

---

## 1. 前置检查（两种方案通用）

在终端执行，确认显卡与驱动：

```bash
nvidia-smi
```

预期输出里应看到 `NVIDIA GeForce RTX 5090`、显存约 `32768MiB`、驱动版本号 ≥ 570（推荐 580.x）。

> 注意：RTX 5090 实际可用显存约 31.4 GiB（不是 32 GiB），规划 KV Cache 时留余量。

---

## 2. 模型权重准备（两种方案通用，先下载好再用）

### 2.1 选哪个权重仓库

| 仓库 | 量化 | 体积 | 单卡 5090 适用性 |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | 67 GB | ❌ 放不下 |
| `Qwen/Qwen3.8-27B-FP8` | FP8 | 38 GB | ❌ 单卡放不下（双卡 TP2 可用） |
| **`Inferact/Qwen3.8-27B-NVFP4`** | NVFP4（W4A4） | 约 16 GB | ✅ **单卡推荐**（草稿接受率高） |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4（混合精度） | 约 16 GB | ✅ 单卡可用，KV 池比 Inferact 大一倍 |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 | 约 16 GB | ✅ 官方构建 |
| 另有 INT4（W4A16）24GB 变体，见 vLLM Recipe 页 | | | ✅ 可用 |

### 2.2 下载方式 A：HuggingFace（推荐，最省心）

```bash
pip install -U "huggingface_hub[cli]"

hf download Inferact/Qwen3.8-27B-NVFP4 \
  --local-dir /models/Qwen3.8-27B-NVFP4
```

> Windows（WSL2）路径同理，`/models/...` 换成 WSL 里的路径即可。

### 2.3 下载方式 B：国内加速（HF 镜像 / ModelScope）

HF 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download Inferact/Qwen3.8-27B-NVFP4 \
  --local-dir /models/Qwen3.8-27B-NVFP4
```

ModelScope（官方权重在此，NVFP4 变体以站内搜索为准）：

```bash
pip install -U modelscope

modelscope download --model Qwen/Qwen3.8-27B \
  --local_dir /models/Qwen3.8-27B
```

### 2.4 下载完成校验

```bash
ls /models/Qwen3.8-27B-NVFP4
# 应包含 config.json 和 *.safetensors 权重文件
```

---

## 3. 方案一：Docker 部署

> 适用于：Linux 服务器/电脑，以及 Windows（配合 WSL2 的 Docker Desktop）。两边的 `docker run` 命令完全一致。

### 3.1（Linux 原生）安装 NVIDIA Container Toolkit

如果 `docker run --gpus` 报错或容器里看不到 GPU，先装 Toolkit：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

> Windows 用 Docker Desktop（WSL2 后端）时**跳过 3.1**，Docker Desktop 自带 GPU 透传，勾选 Settings → General → "Use the WSL 2 based engine" 即可。

### 3.2 拉取镜像 + 验证 GPU 能进容器

```bash
docker pull vllm/vllm-openai:latest

docker run --rm --gpus all --entrypoint python \
  vllm/vllm-openai:latest \
  -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出：`True NVIDIA GeForce RTX 5090`

### 3.3 启动服务（单卡 NVFP4 标准配置）

```bash
docker run -d --name qwen38 \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_ENDPOINT=https://hf-mirror.com \
  vllm/vllm-openai:latest \
  --model Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

说明：

- `-d`：后台运行；看日志用 `docker logs -f qwen38`
- `-v ~/.cache/huggingface:/root/.cache/huggingface`：模型缓存持久化，重启容器不用重新下载
- `-e HF_ENDPOINT=https://hf-mirror.com`：国内下载加速；能直连 HF 可去掉
- 模型已提前下载到本机的话，改用本地挂载：`-v /models/Qwen3.8-27B-NVFP4:/models/Qwen3.8-27B-NVFP4`，并把 `--model` 改成 `/models/Qwen3.8-27B-NVFP4`
- 日志出现 `INFO: Application startup complete.` 即启动成功（首启含权重加载 + 预热，约几分钟）

### 3.4 参数逐项解释

| 参数 | 作用 | 说明 |
|---|---|---|
| `--served-model-name qwen3.8-27b` | 对外暴露的模型名 | 客户端请求时 `"model": "qwen3.8-27b"` |
| `--tensor-parallel-size 1` | 张量并行度 | 单卡固定 1 |
| `--max-model-len 32768` | 最大上下文（输入+输出） | 单卡默认 32K 最稳，显存富裕再加大 |
| `--kv-cache-dtype fp8` | KV Cache 用 FP8 | 单卡容量翻倍的关键，推荐 |
| `--enforce-eager` | 关闭 CUDA Graph 捕获 | **单卡 5090 必加**，否则启动 OOM |
| `--reasoning-parser qwen3` | 正确解析思考块 | 不加的话 `<think>` 整段会进 `message.content` |
| `--enable-auto-tool-choice` | 自动工具调用 | 接 Agent 时用，不需要可去掉 |
| `--tool-call-parser qwen3_coder` | 工具调用解析器 | Qwen3 原生格式，接 Agent 时用 |

### 3.5 可选增强

**开启 MTP 推测解码（提速，草稿头在 checkpoint 内自带）：**

```bash
# 在 3.3 命令末尾追加：
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

**控制思考模式（Qwen3.8 默认 xhigh）：**

```bash
# 关闭思考
--default-chat-template-kwargs '{"enable_thinking": false}'

# 降低推理强度（low / medium / xhigh）
--default-chat-template-kwargs '{"reasoning_effort": "low"}'
```

**扩大单卡上下文 / KV 池（按需，逐级叠加）：**

```bash
# 思路：先看启动日志里的 GPU KV cache size（默认 32K 下约 91,022 tokens）
# 步骤 1：纯文本模式省掉视觉编码器（加约 4.5 万 tokens 池）
--language-model-only
# 步骤 2：压低并发上限（再加约 1.7 万 tokens 池）
--max-num-seqs 8
# 步骤 3：确认池子够大后，再逐步调大 --max-model-len 65536 / 131072 ...
```

---

## 4. 方案二：Windows 部署（WSL2 + pip 原生安装）

> vLLM 官方不支持 Windows 原生运行，官方指定路径就是 WSL2。装好后 vLLM 跑在 WSL 的 Ubuntu 里，性能和 Linux 原生一致。

### 4.1 启用 WSL2 并安装 Ubuntu

PowerShell（管理员）执行：

```powershell
wsl --install
```

重启电脑后会自动装好 WSL2 + Ubuntu；如未自动安装 Ubuntu：

```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

### 4.2 Windows 侧安装 NVIDIA 驱动（关键）

- 在 **Windows** 上安装最新的 NVIDIA 驱动（GeForce Game Ready / Studio 均可，支持 WSL CUDA 的版本 ≥ 570，推荐 580.x）
- **绝对不要在 WSL 里再装 Linux 显卡驱动**——WSL2 通过 Windows 驱动直接透传 CUDA，装 Linux 驱动反而会覆盖掉透传层

### 4.3 WSL 内验证 GPU

```bash
nvidia-smi
```

能看到 RTX 5090 即透传成功（驱动版本显示的是 Windows 驱动）。

### 4.4 安装 uv + Python 3.12 环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

### 4.5 安装 vLLM（自动匹配 CUDA 版本）

```bash
uv pip install vllm --torch-backend=auto

# 安装 FlashInfer 预编译 kernel（Blackwell 上必须，否则启动时现场 JIT 编译极慢）
vllm download-kernels

# 安装匹配的 transformers（模型 config 由 5.8.0 生成）
uv pip install -U "transformers>=5.8.0"
```

> 备选（用 pip 时）：`pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129`

### 4.6 环境自检

```bash
python -c "import torch, vllm; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0))"
```

预期：`vllm 0.2x.x` / `torch 2.x.x` / `cuda 12.9` / `NVIDIA GeForce RTX 5090`

### 4.7 启动服务（命令与 Docker 方案同参）

```bash
vllm serve Inferact/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

看到 `INFO: Application startup complete.` 即成功。

### 4.8 用 tmux 让服务常驻（WSL 关窗口不中断）

```bash
sudo apt-get install -y tmux
tmux new -s vllm
# 在 tmux 里执行 4.7 的启动命令
# 按 Ctrl+B 再按 D 脱离；重新进入：tmux attach -t vllm
```

### 4.9 可选：Windows 上用 Docker Desktop 跑 Docker 方案

1. 安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Settings → General → 勾选 **Use the WSL 2 based engine**
3. Settings → Resources → WSL Integration → 打开 Ubuntu
4. 直接执行第 3 章所有 `docker` 命令

### 4.10 WSL 内存不够时（可选）

在 `C:\Users\<你的用户名>\.wslconfig` 写入：

```ini
[wsl2]
memory=48GB
processors=16
```

然后 PowerShell 执行 `wsl --shutdown` 重启 WSL 生效。Qwen3.8-27B NVFP4 权重约 16GB，整机建议 ≥ 32GB 内存。

---

## 5. 服务验证（两种方案通用）

服务跑起来后（默认端口 8000），开新终端：

**① 模型列表：**

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

**② 健康检查：**

```bash
curl -s http://127.0.0.1:8000/health
# 返回 OK
```

**③ 聊天测试：**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "计算 17 乘以 23，只给答案"}],
    "max_tokens": 1024
  }' | python3 -m json.tool
```

**④ OpenAI SDK 接入（Python）：**

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "用一句话介绍你自己"}],
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

> 服务默认开启了思考模式，回答里会先出推理过程，最后才是答案——这是正常现象。

---

## 6. 进阶：双卡 5090 跑 262K 上下文（可选）

如果学校电脑有两张 5090，用 `unsloth/Qwen3.8-27B-NVFP4`（KV 池大一倍）走 TP2，官方 Recipe 已验证可跑满 262K：

```bash
vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Docker 版同理，把 `vllm serve ...` 换成 `docker run ... vllm/vllm-openai:latest --model unsloth/Qwen3.8-27B-NVFP4 ...` 即可。

---

## 7. 常见问题排查

| 现象 | 原因与解决 |
|---|---|
| 启动时 `torch.OutOfMemoryError: Tried to allocate 784.00 MiB`（CUDA Graph 捕获阶段） | 单卡 5090 通病，**加 `--enforce-eager`**；调 `--gpu-memory-utilization` 无效 |
| 日志警告 FlashInfer 正在 JIT 编译 / 首次请求很慢 | 没装预编译 kernel，跑一次 `vllm download-kernels`（容器内：`docker exec -it qwen38 vllm download-kernels`） |
| 模型加载"成功"但 warmup 崩溃 | FP8 checkpoint 要求 bf16 输出 dtype，加 `--dtype bfloat16`（默认即 bf16，别显式设 float16） |
| 报 `Unexpected reasoning effort high` | Qwen3.8 只支持 `xhigh` / `medium` / `low`，不支持 `high`；客户端与服务端都改用 `xhigh` |
| 模型下载极慢 / 403 | 国内用 `HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope，再不行就提前下好挂载进容器 |
| 显存看着很满 | 正常。vLLM 按 `gpu-memory-utilization`（默认 0.9）启动时预分配 KV 池，跑起来后看 `/metrics` 里的 `vllm:kv_cache_usage_perc` 而不是 `nvidia-smi` 的总占用 |

---

## 8. 参考资料

- vLLM 官方 Recipe（Qwen3.8-27B，含 1×/2× RTX 5090 验证）：https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- vLLM GPU 安装文档：https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- 模型卡（BF16）：https://huggingface.co/Qwen/Qwen3.8-27B
- FP8 权重：https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- NVFP4（Inferact，单卡推荐）：https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4
- NVFP4（unsloth）：https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
- NVFP4（NVIDIA 官方）：https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
- CUDA on WSL 官方说明：https://docs.nvidia.com/cuda/wsl/
- NVIDIA Container Toolkit 安装：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
