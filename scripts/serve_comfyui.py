#!/usr/bin/env python3
"""在 Colab VM 内运行：本地高速盘拉起 ComfyUI + 软链 Google Drive 模型 + 输出回写 Drive + 启动 HTTP/2 隧道。"""

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("PORT", "8188"))
HOST = os.environ.get("HOST", "127.0.0.1")
DRIVE_MODELS = Path(os.environ.get("DRIVE_MODELS_DIR", "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models"))
DRIVE_OUTPUT = Path(os.environ.get("DRIVE_OUTPUT_DIR", "/content/drive/MyDrive/H3_Outputs"))
DRIVE_NODES_ARCHIVE = Path(
    os.environ.get(
        "DRIVE_NODES_ARCHIVE",
        "/content/drive/MyDrive/ComfyUI_H3/custom_nodes_snapshot.tar.gz",
    )
)
INSTALL_NODES = [x.strip() for x in os.environ.get("INSTALL_NODES", "").split(",") if x.strip()]
SAVE_NODES_SNAPSHOT = os.environ.get("SAVE_NODES_SNAPSHOT", "0") == "1"
LOCAL_COMFY = Path("/content/ComfyUI")
SERVER_LOG = Path(f"/tmp/comfyui_{PORT}.log")
TUNNEL_LOG = Path(f"/tmp/cloudflared_{PORT}.log")


def detect_gpu() -> tuple[str, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            name, mem = [x.strip() for x in out.splitlines()[0].split(",")]
            return name, round(float(mem) / 1024.0, 2)
    except Exception:
        pass
    return "CPU", 0.0


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


def _pip_install_requirements(req_file: Path) -> None:
    if not req_file.exists():
        return
    subprocess.run(
        [
            "bash",
            "-lc",
            f"uv pip install --system -q -r '{req_file}' || pip install -q -r '{req_file}'",
        ],
        check=False,
    )


def restore_custom_nodes_snapshot(local_comfy: Path, archive_path: Path) -> bool:
    """从 Google Drive 快照单包秒级恢复 third-party custom_nodes 到本地高速盘。"""
    if not archive_path.exists():
        return False
    nodes_dir = local_comfy / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    print(f"[comfyui] 从 Drive 快照恢复 custom_nodes: {archive_path} ...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(nodes_dir)
    for sub in sorted(nodes_dir.iterdir()):
        if sub.is_dir() and sub.name != "__pycache__":
            _pip_install_requirements(sub / "requirements.txt")
    return True


def install_custom_node_repos(local_comfy: Path, specs: list[str]) -> list[str]:
    """按 GitHub owner/repo 或完整 Git URL 安装自定义节点并装好依赖。"""
    if not specs:
        return []
    nodes_dir = local_comfy / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for raw in specs:
        spec = raw.strip()
        if not spec:
            continue
        if spec.startswith(("http://", "https://", "git@")):
            url = spec
        else:
            url = f"https://github.com/{spec.strip('/')}.git"
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        target = nodes_dir / repo_name
        if not target.exists():
            print(f"[comfyui] 克隆第三方节点 {url} -> {target} ...", flush=True)
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
        _pip_install_requirements(target / "requirements.txt")
        installed.append(repo_name)
    return installed


def save_custom_nodes_snapshot(local_comfy: Path, archive_path: Path) -> str:
    """将当前 custom_nodes 下的第三方子目录打包保存到 Google Drive（剔除 .git 与 __pycache__）。"""
    nodes_dir = local_comfy / "custom_nodes"
    if not nodes_dir.exists():
        return ""
    third_party = [
        d for d in sorted(nodes_dir.iterdir())
        if d.is_dir() and d.name not in ("__pycache__", ".git")
    ]
    if not third_party:
        return ""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar = Path(f"/tmp/{archive_path.name}")

    def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts
        if ".git" in parts or "__pycache__" in parts:
            return None
        return tarinfo

    with tarfile.open(tmp_tar, "w:gz") as tf:
        for d in third_party:
            tf.add(d, arcname=d.name, filter=_tar_filter)
    shutil.move(str(tmp_tar), str(archive_path))
    print(f"[comfyui] 已生成 custom_nodes 快照到 Drive: {archive_path}", flush=True)
    return str(archive_path)


def main() -> None:
    gpu_name, vram_gb = detect_gpu()
    drive_mounted = Path("/content/drive/MyDrive").exists()

    if not (LOCAL_COMFY / "main.py").exists():
        print("[comfyui] 正在克隆 ComfyUI 到本地高速盘 /content/ComfyUI ...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(LOCAL_COMFY)],
            check=True,
        )

    if drive_mounted:
        DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
        DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
        models_link = LOCAL_COMFY / "models"
        if models_link.is_symlink() or models_link.exists():
            if models_link.is_symlink():
                models_link.unlink()
            else:
                shutil.rmtree(models_link)
        models_link.symlink_to(DRIVE_MODELS)
        out_dir = DRIVE_OUTPUT
    else:
        out_dir = LOCAL_COMFY / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

    print("[comfyui] 检查并安装依赖...", flush=True)
    subprocess.run(
        ["bash", "-lc", f"cd {LOCAL_COMFY} && (uv pip install --system -q -r requirements.txt || pip install -q -r requirements.txt)"],
        check=True,
    )

    if drive_mounted:
        restore_custom_nodes_snapshot(LOCAL_COMFY, DRIVE_NODES_ARCHIVE)
    if INSTALL_NODES:
        install_custom_node_repos(LOCAL_COMFY, INSTALL_NODES)
    snapshot_saved = ""
    if drive_mounted and (INSTALL_NODES or SAVE_NODES_SNAPSHOT):
        snapshot_saved = save_custom_nodes_snapshot(LOCAL_COMFY, DRIVE_NODES_ARCHIVE)
    nodes_root = LOCAL_COMFY / "custom_nodes"
    custom_node_names = (
        [d.name for d in sorted(nodes_root.iterdir()) if d.is_dir() and d.name != "__pycache__"]
        if nodes_root.exists()
        else []
    )

    subprocess.run(["pkill", "-f", f"main.py.*--port {PORT}"], check=False)
    time.sleep(1)

    cmd = [
        "python3",
        str(LOCAL_COMFY / "main.py"),
        "--listen", HOST,
        "--port", str(PORT),
        "--output-directory", str(out_dir),
    ]
    print(f"[comfyui] 启动命令: {' '.join(cmd)}", flush=True)
    SERVER_LOG.unlink(missing_ok=True)
    with open(SERVER_LOG, "w", encoding="utf-8") as sf:
        subprocess.Popen(cmd, cwd=str(LOCAL_COMFY), stdout=sf, stderr=subprocess.STDOUT, start_new_session=True)

    deadline = time.time() + 180
    ready = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/system_stats", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        tail = SERVER_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:] if SERVER_LOG.exists() else ""
        raise RuntimeError(f"ComfyUI 未能在 180s 内就绪，日志尾部：\n{tail}")

    public_url = start_tunnel(PORT, HOST)
    result = {
        "PUBLIC_URL": public_url,
        "API_PROMPT_URL": f"{public_url}/prompt",
        "OUTPUT_DIR": str(out_dir),
        "MODELS_DIR": str(DRIVE_MODELS if drive_mounted else LOCAL_COMFY / "models"),
        "GPU_NAME": gpu_name,
        "VRAM_GB": vram_gb,
        "CUSTOM_NODES": custom_node_names,
        "NODES_SNAPSHOT": snapshot_saved or (str(DRIVE_NODES_ARCHIVE) if DRIVE_NODES_ARCHIVE.exists() else ""),
    }
    print("=== RESULT_JSON_START ===", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("=== RESULT_JSON_END ===", flush=True)


if __name__ == "__main__":
    main()
