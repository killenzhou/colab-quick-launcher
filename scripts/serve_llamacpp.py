#!/usr/bin/env python3
"""在 Colab VM 内运行：自动部署 CUDA 版 llama-server，根据 GPU 显存与系统内存动态拉满上下文，并启动 HTTP/2 公网隧道。"""

import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "127.0.0.1")
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
HF_REPO = os.environ.get("HF_REPO", "").strip()
HF_FILE = os.environ.get("HF_FILE", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
ALIAS = os.environ.get("ALIAS", "").strip()
FORCE_CTX = int(os.environ.get("FORCE_CTX", "0") or "0")
MAX_OUTPUT_CAP = int(os.environ.get("MAX_OUTPUT_CAP", "32768") or "32768")
MMPROJ_PATH = os.environ.get("MMPROJ_PATH", "").strip()
NO_MMPROJ = os.environ.get("NO_MMPROJ", "0") == "1"

DRIVE_MODELS_DIR = Path("/content/drive/MyDrive/colab_models/llm")
DRIVE_TOOLS_DIR = Path("/content/drive/MyDrive/colab_tools/llama_bin")
LOCAL_BIN_DIR = Path("/content/llama_bin")
LOCAL_MODEL_DIR = Path("/content/models_cache")
SERVER_LOG = Path(f"/tmp/llama_server_{PORT}.log")
TUNNEL_LOG = Path(f"/tmp/cloudflared_{PORT}.log")


def detect_hardware() -> dict:
    """探测 GPU 型号、显存（GB）、CUDA 版本与系统内存（GB）"""
    gpu_name = "CPU"
    vram_gb = 0.0
    cuda_ver = "12.8"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_name = parts[0]
            vram_gb = round(float(parts[1]) / 1024.0, 2)
        smi_full = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi_full)
        if m:
            cuda_ver = m.group(1)
    except Exception:
        pass

    ram_gb = 0.0
    try:
        meminfo = Path("/proc/meminfo").read_text()
        m = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
        if m:
            ram_gb = round(int(m.group(1)) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "cuda_ver": cuda_ver,
    }


def _shard_siblings(path: Path) -> list[Path]:
    """若路径是 GGUF 分片 (*-00001-of-00004.gguf)，返回同目录下全部分片。"""
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    if not m:
        return [path]
    prefix, _, total = m.groups()
    pat = re.compile(rf"^{re.escape(prefix)}-\d{{5}}-of-{total}\.gguf$", re.IGNORECASE)
    siblings = sorted([p for p in path.parent.iterdir() if p.is_file() and pat.match(p.name)])
    return siblings or [path]


def _total_model_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in _shard_siblings(path) if p.exists())


def find_companion_mmproj(model_file: Path, search_dirs: list[Path]) -> Path | None:
    """查找同模型仓库配套的视觉投影文件 (*mmproj*.gguf)，必要时缓存到本地高速盘。"""
    if NO_MMPROJ:
        return None
    chosen: Path | None = None
    if MMPROJ_PATH and Path(MMPROJ_PATH).exists():
        chosen = Path(MMPROJ_PATH)
    else:
        cands: list[Path] = []
        seen: set[Path] = set()
        for d in search_dirs:
            if not d or not d.exists() or not d.is_dir():
                continue
            for p in d.rglob("*.gguf"):
                if "mmproj" in p.name.lower() and p.is_file() and p not in seen:
                    seen.add(p)
                    cands.append(p)
        if cands:
            for tag in ("f16", "bf16", "q8_0", "f32"):
                hits = [p for p in cands if tag in p.name.lower()]
                if hits:
                    chosen = sorted(hits, key=lambda p: len(p.name))[0]
                    break
            if chosen is None:
                chosen = sorted(cands, key=lambda p: len(p.name))[0]

    if chosen is None:
        return None

    if str(chosen).startswith("/content/drive/") and str(model_file).startswith(str(LOCAL_MODEL_DIR)):
        try:
            sz = chosen.stat().st_size
            free_bytes = shutil.disk_usage("/content").free
            if free_bytes > sz * 1.2 + 3 * (1024**3):
                LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                local_mm = LOCAL_MODEL_DIR / chosen.name
                if not local_mm.exists() or local_mm.stat().st_size != sz:
                    print(f"[vision] 复制视觉投影文件到本地高速盘: {chosen.name} ({sz / 1024**3:.2f} GB)...", flush=True)
                    shutil.copy2(chosen, local_mm)
                return local_mm
        except Exception:
            pass
    return chosen


def ensure_huggingface_model() -> Path:
    """解析或下载 .gguf 模型文件，优先从 Drive 持久化并清理缓存，视本地磁盘空间缓存到 /content"""
    drive_mounted = Path("/content/drive/MyDrive").exists()

    if MODEL_PATH and Path(MODEL_PATH).exists():
        src_path = Path(MODEL_PATH)
    elif HF_REPO and HF_FILE:
        base_dir = DRIVE_MODELS_DIR if drive_mounted else Path("/content/models/llm")
        repo_dir = base_dir / HF_REPO.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        target = repo_dir / HF_FILE
        if not target.exists():
            print(f"[model] 正在从 Hugging Face 下载 {HF_REPO} -> {HF_FILE} ...", flush=True)
            subprocess.run(
                ["bash", "-lc", "uv pip install --system -q huggingface_hub hf_transfer || pip install -q huggingface_hub hf_transfer"],
                check=False,
            )
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            from huggingface_hub import hf_hub_download

            dl = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILE,
                local_dir=str(repo_dir),
                token=HF_TOKEN or None,
            )
            src_path = Path(dl)
            # 清理 huggingface_hub 生成的 .cache 避免占双倍空间
            cache_dir = repo_dir / ".cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            print(f"[model] 复用已有模型: {target}", flush=True)
            src_path = target
    elif MODEL_PATH:
        raise FileNotFoundError(f"指定的 MODEL_PATH 不存在: {MODEL_PATH}")
    else:
        raise RuntimeError("请提供 MODEL_PATH，或同时提供 HF_REPO 与 HF_FILE。")

    # 若模型位于 Drive 且 /content 剩余空间充足，则连同全部分片复制到本地高速盘加速加载
    if str(src_path).startswith("/content/drive/"):
        shards = _shard_siblings(src_path)
        size_bytes = sum(p.stat().st_size for p in shards)
        free_bytes = shutil.disk_usage("/content").free
        if free_bytes > size_bytes * 1.2 + 5 * (1024**3):
            LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            for shard in shards:
                local_copy = LOCAL_MODEL_DIR / shard.name
                if not local_copy.exists() or local_copy.stat().st_size != shard.stat().st_size:
                    print(f"[model] 复制 {shard.name} 到本地高速盘 ({shard.stat().st_size / 1024**3:.2f} GB)...", flush=True)
                    shutil.copy2(shard, local_copy)
            return LOCAL_MODEL_DIR / src_path.name
    return src_path


def find_llama_server(bin_dir: Path) -> Path | None:
    if not bin_dir.exists():
        return None
    for cand in [bin_dir / "llama-server", bin_dir / "build" / "bin" / "llama-server"]:
        if cand.exists():
            return cand
    found = list(bin_dir.rglob("llama-server"))
    return found[0] if found else None


def ensure_llama_server(cuda_ver: str) -> tuple[Path, Path]:
    """确保 CUDA 版 llama-server 与配套动态库就绪"""
    drive_mounted = Path("/content/drive/MyDrive").exists()

    # 1. 优先复用本地 /content/llama_bin 或 Drive 缓存
    srv = find_llama_server(LOCAL_BIN_DIR)
    if srv:
        return srv, srv.parent

    if drive_mounted:
        cached = find_llama_server(DRIVE_TOOLS_DIR)
        if cached:
            print("[llama.cpp] 从 Google Drive 缓存恢复 llama-server...", flush=True)
            shutil.copytree(DRIVE_TOOLS_DIR, LOCAL_BIN_DIR, dirs_exist_ok=True)
            srv = find_llama_server(LOCAL_BIN_DIR)
            if srv:
                srv.chmod(0o755)
                return srv, srv.parent

    # 2. 从 ggml-org/llama.cpp 最新非 v1.0.0 rolling release 下载预编译 CUDA 包
    print("[llama.cpp] 查询最新官方预编译 CUDA 发行版...", flush=True)
    req = urllib.request.Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        releases = json.loads(resp.read().decode("utf-8"))

    # 优先匹配 cuda-12.8-x64（兼容 CUDA 12.8~13.x）或 cuda-13.5-x64（若机器 CUDA >= 13.5）
    try:
        cuda_num = float(cuda_ver)
    except Exception:
        cuda_num = 12.8
    preferred_cuda = "cuda-13.5-x64" if cuda_num >= 13.5 else "cuda-12.8-x64"

    chosen_llama_url = None
    chosen_cudart_url = None
    for rel in releases:
        assets = rel.get("assets", [])
        llama_url = None
        cudart_url = None
        for a in assets:
            name = a["name"]
            if name.startswith("llama-") and preferred_cuda in name and name.endswith(".tar.gz"):
                llama_url = a["browser_download_url"]
            elif name.startswith("cudart-llama-") and preferred_cuda in name and name.endswith(".tar.gz"):
                cudart_url = a["browser_download_url"]
        if not llama_url:
            for a in assets:
                name = a["name"]
                if name.startswith("llama-") and "ubuntu-cuda" in name and name.endswith("-x64.tar.gz"):
                    llama_url = a["browser_download_url"]
                    tag_cuda = re.search(r"(cuda-[0-9.]+-x64)", name)
                    if tag_cuda:
                        for b in assets:
                            if b["name"].startswith("cudart-llama-") and tag_cuda.group(1) in b["name"]:
                                cudart_url = b["browser_download_url"]
                                break
                    break
        if llama_url:
            chosen_llama_url = llama_url
            chosen_cudart_url = cudart_url
            break

    if not chosen_llama_url:
        raise RuntimeError("未找到可用的 llama.cpp Linux CUDA 预编译包。")

    LOCAL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    for url in [u for u in (chosen_cudart_url, chosen_llama_url) if u]:
        tar_path = Path("/tmp") / Path(url).name
        print(f"[llama.cpp] 下载 {tar_path.name} ...", flush=True)
        urllib.request.urlretrieve(url, str(tar_path))
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(LOCAL_BIN_DIR)
        tar_path.unlink(missing_ok=True)

    srv = find_llama_server(LOCAL_BIN_DIR)
    if not srv:
        raise RuntimeError(f"解压后未找到 llama-server: {LOCAL_BIN_DIR}")
    srv.chmod(0o755)

    # 缓存到 Drive 方便下次秒开
    if drive_mounted:
        try:
            DRIVE_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(srv.parent, DRIVE_TOOLS_DIR, dirs_exist_ok=True)
        except Exception as e:
            print(f"[llama.cpp] 缓存到 Drive 失败（可忽略）: {e}", flush=True)

    return srv, srv.parent


def compute_context_plan(vram_gb: float, ram_gb: float, model_gb: float) -> dict:
    """根据显存、系统内存与模型体积计算最优上下文和 GPU 卸载策略"""
    if vram_gb <= 0:
        # 纯 CPU 模式
        ctx = 32768 if ram_gb >= 64 else (16384 if ram_gb >= 32 else 8192)
        max_out = min(MAX_OUTPUT_CAP, ctx // 4)
        return {"ctx": ctx, "max_output": max_out, "ngl": 0, "kv_quant": True}

    free_vram = vram_gb - model_gb - 2.0
    if free_vram >= 36.0:
        ctx = 131072  # 128K
        max_out = min(MAX_OUTPUT_CAP, 32768)
        ngl = 999
    elif free_vram >= 16.0:
        ctx = 65536   # 64K
        max_out = min(MAX_OUTPUT_CAP, 16384)
        ngl = 999
    elif free_vram >= 6.0:
        ctx = 32768   # 32K
        max_out = min(MAX_OUTPUT_CAP, 16384)
        ngl = 999
    elif free_vram >= 0.0:
        ctx = 16384
        max_out = min(MAX_OUTPUT_CAP, 8192)
        ngl = 999
    else:
        # 显存不足以放整模型：使用系统内存混合卸载
        ratio = max(0.15, min(0.85, (vram_gb - 4.0) / max(model_gb, 1.0)))
        ngl = max(8, int(round(64 * ratio)))
        ctx = 32768 if ram_gb >= 64 else 16384
        max_out = min(MAX_OUTPUT_CAP, 8192)

    if FORCE_CTX > 0:
        ctx = FORCE_CTX
        max_out = min(MAX_OUTPUT_CAP, max(4096, ctx // 4))

    return {"ctx": ctx, "max_output": max_out, "ngl": ngl, "kv_quant": True}


def detect_fa_flags(srv: Path, lib_dir: Path) -> list[str]:
    """检测 llama-server 的 Flash Attention 参数写法"""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    try:
        help_text = subprocess.check_output([str(srv), "--help"], env=env, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ["-fa", "on"]
    if re.search(r"-fa,\s*--flash-attn\s*\[(on|off|auto)", help_text):
        return ["-fa", "on"]
    if "--flash-attn" in help_text:
        return ["--flash-attn"]
    return []


def start_tunnel(port: int, host: str) -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not Path(cf_bin).exists():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        print(f"[tunnel] 下载 cloudflared: {url}", flush=True)
        urllib.request.urlretrieve(url, "/usr/local/bin/cloudflared")
        Path("/usr/local/bin/cloudflared").chmod(0o755)
        cf_bin = "/usr/local/bin/cloudflared"

    subprocess.run(["pkill", "-f", f"cloudflared tunnel.*http://{host}:{port}"], check=False)
    TUNNEL_LOG.unlink(missing_ok=True)
    with open(TUNNEL_LOG, "w", encoding="utf-8") as lf:
        subprocess.Popen(
            [cf_bin, "tunnel", "--protocol", "http2", "--url", f"http://{host}:{port}", "--no-autoupdate"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 45
    while time.time() < deadline:
        if TUNNEL_LOG.exists():
            m = pattern.search(TUNNEL_LOG.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(0)
        time.sleep(1)
    raise RuntimeError("45s 内未获取到 cloudflared URL")


def launch_with_fallback(
    srv: Path,
    lib_dir: Path,
    model_file: Path,
    alias: str,
    plan: dict,
    mmproj_file: Path | None = None,
) -> tuple[int, int]:
    """启动 llama-server，若因 OOM 或显存不足退出则自动沿上下文阶梯降级重试。"""
    fa_flags = detect_fa_flags(srv, lib_dir)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    ladder = [plan["ctx"]] + [c for c in (131072, 65536, 32768, 16384, 8192) if c < plan["ctx"]]
    last_tail = ""

    for ctx_try in ladder:
        subprocess.run(["pkill", "-f", f"llama-server.*--port {PORT}"], check=False)
        time.sleep(1)

        cmd = [
            str(srv),
            "-m", str(model_file),
            "--alias", alias,
            "--host", HOST,
            "--port", str(PORT),
            "-c", str(ctx_try),
            "-np", "1",
            "-ngl", str(plan["ngl"]),
            "--jinja",
            *fa_flags,
        ]
        if mmproj_file is not None:
            cmd.extend(["--mmproj", str(mmproj_file)])
        if plan["kv_quant"]:
            cmd.extend(["-ctk", "q8_0", "-ctv", "q8_0"])

        print(f"[llama-server] 尝试 ctx={ctx_try}: {' '.join(cmd)}", flush=True)
        SERVER_LOG.unlink(missing_ok=True)
        with open(SERVER_LOG, "w", encoding="utf-8") as sf:
            proc = subprocess.Popen(cmd, env=env, stdout=sf, stderr=subprocess.STDOUT, start_new_session=True)

        deadline = time.time() + 300
        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                last_tail = SERVER_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SERVER_LOG.exists() else ""
                print(f"[llama-server] ctx={ctx_try} 启动退出 (code={ret})，尝试降级上下文...", flush=True)
                break
            try:
                with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=2) as r:
                    if r.status == 200:
                        max_out = min(MAX_OUTPUT_CAP, max(4096, ctx_try // 4))
                        return ctx_try, max_out
            except Exception:
                pass
            time.sleep(2)
        else:
            last_tail = SERVER_LOG.read_text(encoding="utf-8", errors="ignore")[-1500:] if SERVER_LOG.exists() else ""
            proc.terminate()

    raise RuntimeError(f"llama-server 在所有上下文阶梯 {ladder} 上均启动失败，最后日志：\n{last_tail}")


def main() -> None:
    hw = detect_hardware()
    model_file = ensure_huggingface_model()
    search_dirs = [model_file.parent]
    if MODEL_PATH:
        search_dirs.append(Path(MODEL_PATH).parent)
    if HF_REPO:
        search_dirs.append(DRIVE_MODELS_DIR / HF_REPO.replace("/", "__"))
    mmproj_file = find_companion_mmproj(model_file, search_dirs)
    total_bytes = _total_model_bytes(model_file) + (
        mmproj_file.stat().st_size if mmproj_file and mmproj_file.exists() else 0
    )
    model_gb = round(total_bytes / (1024.0**3), 2)
    plan = compute_context_plan(hw["vram_gb"], hw["ram_gb"], model_gb)
    srv, lib_dir = ensure_llama_server(hw["cuda_ver"])
    alias = ALIAS or model_file.stem

    actual_ctx, actual_max_out = launch_with_fallback(
        srv, lib_dir, model_file, alias, plan, mmproj_file=mmproj_file
    )

    public_url = start_tunnel(PORT, HOST)
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=5) as r:
        models_data = json.loads(r.read().decode("utf-8"))
    model_id = models_data.get("data", [{}])[0].get("id", alias)

    result = {
        "PUBLIC_URL": public_url,
        "BASE_URL": f"{public_url}/v1",
        "MODEL_ID": model_id,
        "CTX_SIZE": actual_ctx,
        "MAX_OUTPUT": actual_max_out,
        "NGL": plan["ngl"],
        "GPU_NAME": hw["gpu_name"],
        "VRAM_GB": hw["vram_gb"],
        "RAM_GB": hw["ram_gb"],
        "MODEL_GB": model_gb,
        "MODEL_PATH": str(model_file),
        "MMPROJ_PATH": str(mmproj_file) if mmproj_file else "",
        "VISION": bool(mmproj_file),
    }
    print("=== RESULT_JSON_START ===", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("=== RESULT_JSON_END ===", flush=True)


if __name__ == "__main__":
    main()
