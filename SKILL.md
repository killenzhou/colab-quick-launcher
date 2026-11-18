---
name: colab-quick-launcher
description: >-
  万事俱备的 Google Colab + Google Drive 全自动部署技能。当用户需要：(1) 自动完成 Colab CLI 登录与 Google Drive 授权挂载，并将完整交互式 Notebook 文件 `Colab_万能一键启动器.ipynb` 同步到 Drive 根目录直接点击运行；(2) 在谷歌网盘缺模型时，先拉起免费 CPU 会话将 HuggingFace 模型（含多片 GGUF 与视觉多模态 `mmproj`）预下载至网盘、清理缓存后再切换 GPU 算力；(3) 用 CUDA 版 llama.cpp (llama-server) 后台部署开源大语言模型与视觉多模态服务，根据显存/内存动态拉满超大上下文（含 OOM 自动降级），经 1-token 冒烟测试后自动同步 `~/.config/opencode/opencode.jsonc`（含多模态字段）；(4) 用本地高速盘 + 软链 Drive 模型 + `custom_nodes` 快照恢复方式部署 ComfyUI，并支持工作流缺节点检测、UI->API 编译及 `workflow run` 一键出图拉回本地；(5) 一键体检 `status`、全部关机 `down`、防烧算力闲置自动关机 `watch`、复用已验证档案 `run-profile` 或断线重建隧道 `reconnect` 时，使用此技能。
---

# Colab Quick Launcher（万事俱备一键运行指南）

本技能把 **Google Colab 登录授权、Google Drive 挂载、浏览器直跑 Notebook 启动文件、CPU 预下载入网盘、GPU 自动择优、CUDA `llama-server`（含 `mmproj` 视觉多模态与 OOM 防护）、OpenCode 自动同步、`custom_nodes` 快照恢复、ComfyUI UI->API 编译运行、全局体检/保活与闲置自动停机** 全部封装完成，直接调用即可运行。

## 核心资产与脚本清单

- 统一命令行主入口：[colab_oneclick.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py)
- 独立 Colab Notebook 文件（每次连接会自动同步到 `/content/drive/MyDrive/Colab_万能一键启动器.ipynb`，浏览器打开即点即跑）：
  - 工作区副本：[Colab_万能一键启动器.ipynb](file:///Users/kunlinzhou/Downloads/谷歌colab/Colab_万能一键启动器.ipynb)
  - 技能内置模板：[Colab_万能一键启动器.ipynb](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/assets/Colab_万能一键启动器.ipynb)
- 下属模块：
  - [colab_up.sh](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/colab_up.sh)：`G4 -> A100 -> L4 -> T4 -> CPU` 择优启动（`--high-mem` 失败自动回退）
  - [prefetch_to_drive.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/prefetch_to_drive.py)：CPU 预下载 `.gguf` 分片 + `mmproj` 视觉投影或 `.safetensors` 到网盘，校验剩余空间（+5GB 缓冲）并清理 `.cache` / `split_files/`
  - [serve_llamacpp.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/serve_llamacpp.py)：CUDA `llama-server` 自适应上下文（最高 128K）、自动挂载 `--mmproj`、`launch_with_fallback` OOM 阶梯降级（`131072->65536->32768->16384->8192`）与 HTTP/2 隧道
  - [sync_opencode.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/sync_opencode.py)：探测 `/v1/models` & `/props`、1-token 冒烟测试、清除失效 `*.trycloudflare.com` 旧条目（保留 `seedance`），支持 `--vision` 写入图像输入能力到 `~/.config/opencode/opencode.jsonc`
  - [serve_comfyui.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/serve_comfyui.py)：NVMe 部署 `/content/ComfyUI` + 软链 Drive 模型 + 从 `custom_nodes_snapshot.tar.gz` 秒级恢复第三方节点插件
  - [comfy_workflow_builder.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/comfy_workflow_builder.py)：探查已装模型、内置模板、节点 Schema、缺节点检测 (`check-nodes`)、UI->API 编译 (`compile-api`) 与命令行直接出图拉回成品 (`run`)
  - [expose_tunnel.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/expose_tunnel.py)：秒级新建/重建 HTTP/2 公网隧道

---

## 一、把已验证成果固化为“直接执行”的文件与档案

你现在有两种“下次直接点/一键跑”的方式：

1. **直接复用历史已验证配置 (`run-profile`)**
   每次成功启动 `llm` 或 `comfyui`，参数会自动写入 `~/.colab_verified_profiles.json`（默认保存为 `last-llm` / `last-comfyui`，也可用 `--profile <名字>` 命名）。下次只需一行命令：
   ```bash
   # 查看所有已验证档案
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py list-profiles

   # 一键复刻上次验证通过的大语言模型环境
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py run-profile --name last-llm

   # 一键复刻上次验证通过的 ComfyUI 环境
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py run-profile --name last-comfyui
   ```
2. **浏览器打开 Drive 根目录的 `Colab_万能一键启动器.ipynb`**
   脚本每次连接 Colab 时都会把最新的 [Colab_万能一键启动器.ipynb](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/assets/Colab_万能一键启动器.ipynb) 自动同步到你 Google Drive 根目录。你也可以随时手动导出或推送：
   ```bash
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py export-notebook --out ./Colab_万能一键启动器.ipynb
   ```

---

## 二、前置授权 & 受限模型 Token

```bash
# 保存 HuggingFace Token（解决 LTX-2.5 等 gated 模型 401 报错）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py set-hf-token --token "hf_你的Token"
```
Google Colab 登录态与 Google Drive 挂载均在脚本中自动检测，失效或未挂载时会主动触发 `colab login` / 交互式 `colab drivemount`。

---

## 三、CPU 预下载 & 一键启动 LLM（含 `mmproj` 视觉 + OpenCode + 自动停机）

```bash
# 1. 仅用 CPU 预下载模型（含 GGUF 分片与配套 mmproj 视觉文件）入网盘，下完自动 stop CPU
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py prefetch \
  --hf-repo "Qwen/Qwen2.5-VL-7B-Instruct-GGUF" --kind llm

# 2. 一键开启 GPU 大模型（缺模型会自动走完上面 CPU 阶段，再开 GPU 并挂载 mmproj 与同步 OpenCode）
#    加上 --auto-watch-idle 30：若连续闲置 30 分钟无调用，自动关闭 GPU 省算力并语音提醒
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py llm \
  --session llm-server --gpu AUTO \
  --hf-repo "bartowski/Qwen_Qwen3-32B-GGUF" \
  --model-id "qwen3-32b" \
  --profile "qwen3-32b-verified" \
  --auto-watch-idle 30
```

---

## 四、一键启动 ComfyUI & 插件快照 & 工作流四步生成与执行

```bash
# 启动 ComfyUI（首次可带 --install-node 自动克隆并打包 snapshot 到网盘，以后开机自动秒恢复）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py comfyui \
  --session comfy-prod --gpu AUTO \
  --install-node "Kosinkadink/ComfyUI-VideoHelperSuite" \
  --save-nodes-snapshot \
  --auto-watch-idle 30
```

**根据用户需求制作并运行新 ComfyUI 工作流（详细原理见 [model_lookup_and_serving.md](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/references/model_lookup_and_serving.md)）：**

```bash
WF="python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow"

# 1. 盘点 Drive 已安装模型精确名称
$WF inspect-models --url "https://<comfy-url>"

# 2. 检索内置官方模板 & 检查本地工作流是否缺自定义节点
$WF list-templates --url "https://<comfy-url>" --keyword "ltx"
$WF check-nodes --url "https://<comfy-url>" --file "./my_flow.json"

# 3. 核对单节点参数 / 自动换名适配官方模板并推送进 WebUI 菜单
$WF inspect-node --url "https://<comfy-url>" --node "KSampler"
$WF adapt-template --url "https://<comfy-url>" --template "video_ltx2_5_i2v" --out "./new_flow.json" --upload-name "new_flow.json"

# 4. 无需打开网页，直接编译 UI->API 并传参考图跑出成品拉回 Mac 本地
$WF run --url "https://<comfy-url>" --file "./new_flow.json" \
  --prompt "cinematic shot, 4k" --image "./ref.png" --out-dir "./outputs"
```

---

## 五、全局体检 (`status`)、保活看门狗 (`watch`)、断线重建 (`reconnect`) 与全关 (`down`)

```bash
# 一键查看所有活跃会话 GPU 利用率、隧道状态、Drive 剩余空间、OpenCode 默认模型是否在线
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py status

# 仅隧道断线时重建 HTTP/2 公网域名并自动更新 OpenCode
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py reconnect --session llm-server --type llm

# 启动前台/后台保活与闲置自动关机守护（活跃时保活 Kernel，闲置 30 分钟自动停机并语音提醒）
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py watch --session llm-server --idle-minutes 30

# 下班/睡前一条命令停掉所有活跃 Colab 实例，防止漏关烧算力
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py down

# 网盘空间不足时用 CPU 会话一键深清 .cache
python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py cleanup-drive
```
