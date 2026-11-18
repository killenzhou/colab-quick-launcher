# Google Colab + Google Drive + LLM / ComfyUI 全自动调用指南

本指南总结了在 macOS 环境下全自动打通 **Google Colab 算力、Google Drive 存储、CPU 预下载入网盘、`llama.cpp`（含视觉 `mmproj` 与 OpenCode 同步）、ComfyUI（含 `custom_nodes` 网盘快照与工作流直跑）及独立一键 Notebook 启动文件** 的完整方案。

---

## 一、快速入口（两种直接执行方式）

1. **本地已验证文件直接打开/运行**
   - 独立 Notebook 启动文件：[Colab_万能一键启动器.ipynb](file:///Users/kunlinzhou/Downloads/谷歌colab/Colab_万能一键启动器.ipynb)（每次连接 Colab 都会自动同步到你的 Google Drive 根目录 `/content/drive/MyDrive/Colab_万能一键启动器.ipynb`，浏览器双击即可按单元运行）。
   - Mac 端一键总控脚本：[colab_oneclick.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py) 与完整技能说明 [SKILL.md](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/SKILL.md)。
2. **下次直接复刻已验证的配置 (`run-profile`)**
   每次成功启动 `llm` 或 `comfyui`，参数会自动存入 `~/.colab_verified_profiles.json`（含 `last-llm`、`last-comfyui` 或自定义 `--profile` 名称）：
   ```bash
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py run-profile --name last-llm
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py run-profile --name last-comfyui
   ```

---

## 二、日常全套常用命令速查

### 1. 首次准备：保存受限模型 HF Token
```bash
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py set-hf-token --token "hf_你的Token"
```

### 2. 网盘缺模型时：先用免费 CPU 预下载到网盘（下完自动关 CPU）
```bash
# 下载大模型（自动收齐 GGUF 分片 + 配套 mmproj 视觉文件，清完 .cache 自动停机）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py prefetch \
  --hf-repo "bartowski/Qwen_Qwen3-32B-GGUF" --kind llm

# 下载 ComfyUI 模型（自动剥离 split_files/ 冗余层级）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py prefetch \
  --hf-repo "Lightricks/LTX-2.5" \
  --hf-file "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors" \
  --kind comfyui
```

### 3. 一键开 GPU 跑开源大语言模型 + 自动同步 OpenCode（含多模态 + 防烧算力看门狗）
```bash
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py llm \
  --session llm-server --gpu AUTO \
  --hf-repo "bartowski/Qwen_Qwen3-32B-GGUF" \
  --model-id "qwen3-32b" \
  --profile "qwen3-32b" \
  --auto-watch-idle 30
```
- 会先用 CPU 预下载缺失模型再切 GPU，按显存动态拉满 `128K/64K/32K` 上下文（带 OOM 自动阶梯降级），发现 `mmproj` 自动挂载 `--mmproj` 并给 OpenCode 写入视觉输入支持，同时自动清理失效的旧 `*.trycloudflare.com` 配置。

### 4. 一键开 GPU 跑 ComfyUI（含第三方节点网盘快照与工作流直跑）
```bash
# 启动并安装/恢复第三方插件（快照存网盘后以后秒恢复）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py comfyui \
  --session comfy-prod --gpu AUTO \
  --install-node "Kosinkadink/ComfyUI-VideoHelperSuite" \
  --save-nodes-snapshot \
  --auto-watch-idle 30
```

**创建/检查/运行 ComfyUI 新工作流：**
```bash
WF="python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow"
$WF inspect-models --url "https://<comfy-url>"
$WF list-templates --url "https://<comfy-url>" --keyword "ltx"
$WF check-nodes --url "https://<comfy-url>" --file "./my_wf.json"
$WF adapt-template --url "https://<comfy-url>" --template "video_ltx2_5_i2v" --out "./new.json" --upload-name "new.json"
$WF run --url "https://<comfy-url>" --file "./new.json" --prompt "cinematic shot" --image "./ref.png" --out-dir "./outputs"
```

### 5. 全局体检、重连隧道与收工关机（防扣费）
```bash
# 查看所有活跃会话 GPU 利用率、公网隧道是否在线、Drive 空间与 OpenCode 状态
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py status

# 隧道断线时秒级重建域名并自动更新 OpenCode（不重启模型）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py reconnect --session llm-server --type llm

# 下班或睡前一键关闭所有 Colab 会话，立刻停计费
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py down

# 网盘满时清理 .cache 临时文件
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py cleanup-drive
```
