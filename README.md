# Colab Quick Launcher

一键打通 **Google Colab 算力 + Google Drive 存储 + CUDA `llama.cpp`（含 `mmproj` 视觉多模态 & OpenCode 自动同步）+ ComfyUI（含 `custom_nodes` 快照恢复与 UI→API 编译执行）** 的自动化技能与独立 Notebook 启动器。

## 核心内容

- **[SKILL.md](SKILL.md)** & **[references/model_lookup_and_serving.md](references/model_lookup_and_serving.md)**：完整技能规范、模型选型与工作流四步构建手册
- **[Colab与谷歌云盘调用指南.md](Colab与谷歌云盘调用指南.md)**：高频场景一站式速查指南
- **[Colab_万能一键启动器.ipynb](Colab_万能一键启动器.ipynb)**（亦存一份于 `assets/`）：打开 Colab 浏览器即可运行的 4 单元独立 Notebook
- **自动化脚本（`scripts/`）**：
  - `colab_oneclick.py`：统一编排 CLI（OAuth/Drive 挂载检测、`set-hf-token`、`prefetch`、`llm`、`comfyui`、`run-profile`、`status`、`down`、`watch`、`reconnect`、`workflow`、`cleanup-drive`）
  - `colab_up.sh`：按 `G4 -> A100 -> L4 -> T4 -> CPU` 优先拉起 `--high-mem` 运行时并支持失败回退
  - `prefetch_to_drive.py`：CPU 模式预下载 HF `.gguf` 分片 / `mmproj` / `.safetensors` 到 Google Drive 并清理 `.cache` / `split_files/`
  - `serve_llamacpp.py`：CUDA `llama-server` 硬件自适应大上下文（最高 128K）、`-fa` + Q8 KV 量化、OOM 自动降级梯子与 `--mmproj` 视觉挂载
  - `sync_opencode.py`：探测 `/v1/models` & `/props`、1-token 冒烟测试、自动清理失效旧隧道并将新端点写入 `~/.config/opencode/opencode.jsonc`
  - `serve_comfyui.py`：本地高速 NVMe 部署 ComfyUI + 软链 Drive 模型 + `custom_nodes_snapshot.tar.gz` 秒级恢复
  - `comfy_workflow_builder.py`：模型盘点、内置模板检索、节点 Schema 查验、缺节点检查、UI→API 编译及命令行直跑出图
  - `expose_tunnel.py`：Cloudflare `http2` 协议公网隧道重建
- **示例 ComfyUI 工作流（`examples/h3_comfyui/`）**：包含文生图、多图参考关键帧、图生视频工作流 JSON 与客户端脚本 `h3_client.py`

## 安装到 Antigravity Skills

```bash
mkdir -p ~/.gemini/config/skills/colab-quick-launcher
cp -R SKILL.md scripts references assets ~/.gemini/config/skills/colab-quick-launcher/
chmod +x ~/.gemini/config/skills/colab-quick-launcher/scripts/*.py ~/.gemini/config/skills/colab-quick-launcher/scripts/*.sh
```

## 快速命令示例

```bash
ONECLICK="python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py"

# 1. 保存 HuggingFace 受限模型 Token
$ONECLICK set-hf-token --token "hf_你的Token"

# 2. CPU 预下载模型入网盘 -> 关停 CPU -> 开 GPU 部署 LLM 并同步 OpenCode（含闲置 30 分钟自动停机）
$ONECLICK llm --session llm-server --gpu AUTO --hf-repo "bartowski/Qwen_Qwen3-32B-GGUF" --auto-watch-idle 30

# 3. 一键复刻上次已验证配置
$ONECLICK run-profile --name last-llm

# 4. 启动 ComfyUI（含插件快照恢复）并运行新工作流
$ONECLICK comfyui --session comfy-prod --gpu AUTO --auto-watch-idle 30
$ONECLICK workflow inspect-models --url "https://<comfy-url>"
$ONECLICK workflow run --url "https://<comfy-url>" --file "./workflow.json" --prompt "cinematic shot" --out-dir "./outputs"

# 5. 全局体检与收工一键停机防扣费
$ONECLICK status
$ONECLICK down
```
