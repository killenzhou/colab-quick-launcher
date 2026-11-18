import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

TARGET_PORT = int(os.environ.get("TARGET_PORT", "8000"))
TARGET_HOST = os.environ.get("TARGET_HOST", "127.0.0.1")
LOG_PATH = Path(f"/tmp/cloudflared_{TARGET_PORT}.log")
BIN_PATH = Path("/usr/local/bin/cloudflared")


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def ensure_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    if BIN_PATH.exists():
        BIN_PATH.chmod(0o755)
        return str(BIN_PATH)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print(f"[tunnel] 下载 cloudflared: {url}", flush=True)
    urllib.request.urlretrieve(url, str(BIN_PATH))
    BIN_PATH.chmod(0o755)
    return str(BIN_PATH)


def main() -> None:
    if not wait_for_port(TARGET_HOST, TARGET_PORT, timeout=30):
        raise RuntimeError(
            f"本地端口 {TARGET_HOST}:{TARGET_PORT} 未在 30s 内监听，请先确认后端服务已启动。"
        )

    cf_bin = ensure_cloudflared()

    # 清理针对同一端口的旧隧道进程
    subprocess.run(
        ["pkill", "-f", f"cloudflared tunnel.*http://{TARGET_HOST}:{TARGET_PORT}"],
        check=False,
    )
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    cmd = [
        cf_bin,
        "tunnel",
        "--protocol",
        "http2",
        "--url",
        f"http://{TARGET_HOST}:{TARGET_PORT}",
        "--no-autoupdate",
    ]
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 45
    public_url = None
    while time.time() < deadline:
        if LOG_PATH.exists():
            text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
            match = pattern.search(text)
            if match:
                public_url = match.group(0)
                break
        time.sleep(1)

    if not public_url:
        tail = LOG_PATH.read_text(encoding="utf-8", errors="ignore")[-1000:] if LOG_PATH.exists() else ""
        raise RuntimeError(f"45s 内未获取到 trycloudflare URL。日志尾部：\n{tail}")

    print("==========================================", flush=True)
    print(f"LOCAL_TARGET: http://{TARGET_HOST}:{TARGET_PORT}", flush=True)
    print(f"PUBLIC_URL: {public_url}", flush=True)
    print(f"TUNNEL_LOG: {LOG_PATH}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
