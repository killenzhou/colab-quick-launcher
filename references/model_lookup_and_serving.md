# 模型选型、存储规避、工作流定制、插件快照与运维参考

## 1. 不同 Colab 算力可稳定运行的开源 LLM 选型参考

[serve_llamacpp.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/serve_llamacpp.py) 会依据 **总模型体积（主 `.gguf` 全部分片 + 配套 `mmproj` 视觉文件）**、**剩余显存 (`VRAM - 模型总重 - 2GB`)**、**系统内存** 与 **Q8 KV 缓存量化 (`-ctk q8_0 -ctv q8_0` + `-fa`)** 计算初始上下文并自带 OOM 自动降级保护：

| Colab 硬件 | 显存 / 内存 (High-Mem) | 满血 GPU 卸载推荐模型 | 典型实测可用上下文 |
| :--- | :--- | :--- | :--- |
| **G4 (`RTX PRO 6000`)** | 96GB VRAM / 200+ GB RAM | **70B~72B Q4_K_M** (~43GB) / **120B~123B IQ3_XS~Q4** | 72B 可稳开 **64K~128K**；32B 直接拉满 **128K+** |
| **A100-80GB** | 80GB VRAM / 167GB RAM | **70B~72B Q4_K_M** (~43GB) / **32B Q8_0** | 72B 可开 **64K**；32B Q4 直接拉满 **128K** |
| **L4** | 24GB VRAM / 53GB RAM | **14B~27B Q4_K_M** / **32B IQ4_XS** (~18GB) | 14B~27B 可稳开 **32K~64K** |
| **T4** | 15GB VRAM / 51GB RAM | **7B~14B Q4_K_M** (~5~9GB) | 7B~9B 可开 **32K**；14B 开 **8K~16K** |
| **CPU (High-RAM)** | 0 VRAM / 51GB+ RAM | 仅作测试、预下载或低并发保底 | 自动回退 8K~32K |

> **多模态 (`mmproj`) 自动启用**：若同模型目录下存在 `*mmproj*.gguf`（`prefetch_to_drive.py` 在 `llm` 模式默认连同主 GGUF 一起下载），`serve_llamacpp.py` 会自动附带 `--mmproj <file>` 启动，并在 `sync_opencode.py --vision` 时为 OpenCode 写入 `"attachment": true` 与 `"modalities": {"input": ["text", "image"], "output": ["text"]}`。

---

## 2. Google Drive 存储加速与 `custom_nodes` 快照设计

1. **`huggingface_hub` 双份 `.cache` 与 `split_files/` 清理**：[prefetch_to_drive.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/prefetch_to_drive.py) 使用 `/content/drive/MyDrive/_hf_prefetch_tmp` 下载并移动，结束后删除临时目录和 `.cache/`，同时剥离 `split_files/` 前缀，避免 300GB 网盘被隐形翻倍占用。
2. **ComfyUI 程序装高速盘，模型软链 Drive**：[serve_comfyui.py](file:///Users/kunlinzhou/.gemini/config/skills/colab-quick-launcher/scripts/serve_comfyui.py) 始终把代码装在 `/content/ComfyUI` 本地 NVMe，`models/` 软链指向 Google Drive。
3. **`custom_nodes` 单包快照秒装（避免 Drive 小文件卡死）**：
   - 将第三方插件放在 Drive 默认会因数万个 `.py` 文件导致开机 `import` 极慢；
   - `serve_comfyui.py` 采用“**本地运行 + 单 `.tar.gz` 压缩包归档至 `/content/drive/MyDrive/ComfyUI_H3/custom_nodes_snapshot.tar.gz`**”方案：通过 `--install-node owner/repo --save-nodes-snapshot` 打包一次后，每次新建 GPU 运行时仅用数秒解压至本地 `/content/ComfyUI/custom_nodes` 并快速对齐依赖即可就绪。

---

## 3. 算力防呆、保活与断线重建机制

1. **为什么需要保活与闲置自动停机 (`watch`)？**
   - Colab 对长时间只走外部 Tunnel 而 Kernel 无活动的实例可能在 60~90 分钟回收；反过来，跑完任务忘关 `G4/A100` 会整夜扣 Compute Units。
   - `colab_oneclick.py watch --session <sess> --idle-minutes 30`（或启动 `llm`/`comfyui` 时直接传 `--auto-watch-idle 30`）会定期向 Colab Kernel 写心跳防被误杀，并实时读 `nvidia-smi` 利用率、ComfyUI `/queue` 与服务日志活跃时间；一旦连续闲置超过阈值即自动 `colab stop` 并触发 macOS `say` 语音提醒。
2. **一键全局收工 (`down`) 与健康检查 (`status`)**：
   - 睡前执行 `colab_oneclick.py down` 可自动枚举并关停所有活跃会话。

---

## 4. 根据用户需求创建、适配并执行 ComfyUI 工作流（四步闭环）

1. **第一步：盘点实际已安装模型**
   ```bash
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow inspect-models --url "https://<comfy-url>"
   ```
2. **第二步：检索模板 & 检测是否缺少第三方节点**
   ```bash
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow list-templates --url "https://<comfy-url>" --keyword "ltx"
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow check-nodes --url "https://<comfy-url>" --file "./workflow.json"
   ```
   若 `check-nodes` 报缺失节点（并给出对应 `--install-node owner/repo` 建议），用该参数重启一次 ComfyUI 即可自动安装并永久存入 `custom_nodes_snapshot.tar.gz`。
3. **第三步：核对未知节点 Schema 并自动替换模型名保存 + 上传 WebUI**
   ```bash
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow inspect-node --url "https://<comfy-url>" --node "LTXVSampler"
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow adapt-template \
     --url "https://<comfy-url>" --template "video_ltx2_5_i2v" \
     --out "./new_workflow.json" --upload-name "new_workflow.json"
   ```
4. **第四步：直接命令行编译执行并把成品拉回本地**
   ```bash
   # 仅转换为标准 /prompt API 格式保存（UI->API 自动追踪 Reroute 并映射 widgets_values）
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow compile-api \
     --url "https://<comfy-url>" --file "./new_workflow.json" --out "./new_workflow_api.json"

   # 或直接运行 UI / API 工作流，自动传参考图并下载生成图片/视频到本地 ./outputs
   python3 ~/.gemini/config/skills/colab-quick-launcher/scripts/colab_oneclick.py workflow run \
     --url "https://<comfy-url>" --file "./new_workflow.json" \
     --prompt "high quality cinematic scene" --image "./ref.png" --out-dir "./outputs"
   ```
