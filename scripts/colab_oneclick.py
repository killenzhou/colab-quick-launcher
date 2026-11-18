#!/usr/bin/env python3
"""Turnkey launcher for Colab + Drive + llama.cpp + ComfyUI + OpenCode."""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPTS_DIR.parent / "assets"
NOTEBOOK_CANDIDATES = (
    ASSETS_DIR / "Colab_万能一键启动器.ipynb",
    Path("/Users/kunlinzhou/Downloads/谷歌colab/Colab_万能一键启动器.ipynb"),
)
PROFILES_FILE = Path.home() / ".colab_verified_profiles.json"

COLAB_CACERT = (
    "/Users/kunlinzhou/.local/share/uv/tools/google-colab-cli/"
    "lib/python3.13/site-packages/certifi/cacert.pem"
)

HF_TOKEN_PATHS = (
    Path("~/.config/huggingface/token").expanduser(),
    Path("~/.cache/huggingface/token").expanduser(),
    Path("~/.hf_token").expanduser(),
)


def get_colab_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = str(Path("~/.local/bin").expanduser())
    hb_bin = str(Path("~/homebrew/bin").expanduser())
    path_parts = [p for p in env.get("PATH", "").split(":") if p]
    for extra in (local_bin, hb_bin):
        if extra not in path_parts:
            path_parts.insert(0, extra)
    env["PATH"] = ":".join(path_parts)
    if os.path.exists(COLAB_CACERT):
        env.setdefault("SSL_CERT_FILE", COLAB_CACERT)
        env.setdefault("REQUESTS_CA_BUNDLE", COLAB_CACERT)
    return env


def get_ssl_context() -> ssl.SSLContext:
    if os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ssl.create_default_context()


def colab_bin(env: dict[str, str]) -> str:
    found = shutil.which("colab", path=env.get("PATH"))
    if found:
        return found
    candidate = Path("~/.local/bin/colab").expanduser()
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("未找到 `colab` CLI，请确认 google-colab-cli 已安装并在 PATH 中。")


def load_hf_token(explicit_token: str = "") -> str:
    if explicit_token.strip():
        return explicit_token.strip()
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    for tok_file in HF_TOKEN_PATHS:
        if tok_file.exists():
            val = tok_file.read_text(encoding="utf-8").strip()
            if val:
                return val
    return ""


def cmd_set_hf_token(args: argparse.Namespace) -> int:
    token = args.token.strip()
    if not token:
        print("❌ Token 不能为空。", file=sys.stderr)
        return 1
    saved_paths: list[str] = []
    for target in HF_TOKEN_PATHS[:2]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(token + "\n", encoding="utf-8")
        os.chmod(target, 0o600)
        saved_paths.append(str(target))
    print("✅ HuggingFace Token 已安全写入（权限 600）：")
    for p in saved_paths:
        print(f"  - {p}")
    return 0


def ensure_colab_auth(env: dict[str, str]) -> str:
    """Verify Google Colab OAuth session; trigger `colab login` when expired/missing."""
    cbin = colab_bin(env)
    check = subprocess.run(
        [cbin, "sessions"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{check.stdout}\n{check.stderr}"
    auth_markers = ("401", "403", "unauthorized", "login", "credential", "token", "oauth")
    if check.returncode == 0:
        return check.stdout

    if any(m in combined.lower() for m in auth_markers):
        print("⚠️ Google Colab 登录态已失效或尚未登录，正在拉起浏览器完成 `colab login` OAuth 授权...")
    else:
        print(f"⚠️ `colab sessions` 检查异常（exit={check.returncode}），尝试重新执行 `colab login`...")

    subprocess.run([cbin, "login"], env=env, check=True)
    recheck = subprocess.run(
        [cbin, "sessions"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    print("✅ Google Colab OAuth 授权正常。")
    return recheck.stdout


def parse_active_sessions(sessions_output: str) -> list[str]:
    """Parse active session names from `colab sessions` stdout cleanly."""
    names: list[str] = []
    skip_words = {"name", "session", "sessions", "id", "runtime", "gpu", "status", "no"}
    for raw_line in sessions_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line) <= set("-=|+ "):
            continue
        if line.startswith("|"):
            line = line.strip("|").strip()
        parts = re.split(r"[\s|]+", line)
        if not parts:
            continue
        cand = parts[0].strip()
        if not cand or cand.lower() in skip_words:
            continue
        if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", cand) and cand not in names:
            names.append(cand)
    return names


def session_exists(sessions_output: str, session_name: str) -> bool:
    return session_name in parse_active_sessions(sessions_output)


def ensure_session(session: str, gpu_pref: str, high_mem: bool, env: dict[str, str]) -> None:
    up_script = SCRIPTS_DIR / "colab_up.sh"
    mem_flag = "--high-mem" if high_mem else "--no-high-mem"
    print(f"🔌 [1/4] 连接或启动 Colab 会话 `{session}` (策略: {gpu_pref}, {mem_flag})...")
    subprocess.run(["bash", str(up_script), session, gpu_pref, mem_flag], env=env, check=True)


def stop_session(session: str, env: dict[str, str]) -> None:
    cbin = colab_bin(env)
    print(f"🛑 正在释放/停止 Colab 会话 `{session}`...")
    res = subprocess.run([cbin, "stop", "-s", session], env=env, check=False)
    if res.returncode == 0:
        print(f"✅ 已停止会话 `{session}`。")
    else:
        print(f"⚠️ 停止会话 `{session}` 返回 exit={res.returncode}（可能已关闭）。")


def ensure_drive_mounted(session: str, env: dict[str, str]) -> None:
    cbin = colab_bin(env)
    print(f"💾 [2/4] 检查 Google Drive 挂载状态 ({session})...")
    probe = subprocess.run(
        [
            cbin,
            "exec",
            "-s",
            session,
            "ls /content/drive/MyDrive >/dev/null 2>&1 && echo MOUNTED || echo NOT_MOUNTED",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if "MOUNTED" in probe.stdout and "NOT_MOUNTED" not in probe.stdout:
        print("✅ Google Drive 已在 Colab 中挂载就绪。")
        return

    print("🔑 尚未挂载 Google Drive，即将拉起 `colab drivemount`（请在浏览器完成授权并在提示时按回车确认）...")
    subprocess.run([cbin, "drivemount", "-s", session], env=env, check=True)

    verify = subprocess.run(
        [
            cbin,
            "exec",
            "-s",
            session,
            "ls /content/drive/MyDrive >/dev/null 2>&1 && echo MOUNTED || echo NOT_MOUNTED",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if "MOUNTED" not in verify.stdout or "NOT_MOUNTED" in verify.stdout:
        raise RuntimeError(
            "Google Drive 挂载验证失败（/content/drive/MyDrive 不可访问）。请检查授权后重试。"
        )
    print("✅ Google Drive 挂载验证成功。")


def sync_notebook_to_drive(session: str, env: dict[str, str]) -> None:
    """Sync verified Colab_万能一键启动器.ipynb to /content/drive/MyDrive/ when mounted."""
    src = next((p for p in NOTEBOOK_CANDIDATES if p.exists()), None)
    if not src:
        return
    try:
        raw = src.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        b64 = base64.b64encode(raw).decode("ascii")
        remote_py = (
            "import base64, hashlib\n"
            "from pathlib import Path\n"
            "drv = Path('/content/drive/MyDrive')\n"
            "if drv.exists():\n"
            "    dst = drv / 'Colab_万能一键启动器.ipynb'\n"
            f"    want = '{digest}'\n"
            "    cur = hashlib.sha256(dst.read_bytes()).hexdigest() if dst.exists() else ''\n"
            "    if cur != want:\n"
            f"        dst.write_bytes(base64.b64decode('{b64}'))\n"
            "        print('📓 已同步最新版 Colab_万能一键启动器.ipynb 到 Google Drive 根目录')\n"
        )
        cbin = colab_bin(env)
        subprocess.run([cbin, "exec", "-s", session, remote_py], env=env, check=False)
    except Exception:
        pass


def save_verified_profile(
    name: str,
    payload: dict,
    profiles_path: Path = PROFILES_FILE,
    also_save_alias: str = "",
) -> Path:
    """Persist a verified Colab launch configuration to ~/.colab_verified_profiles.json."""
    data: dict = {}
    if profiles_path.exists():
        try:
            data = json.loads(profiles_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    record = dict(payload)
    record["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    if name:
        data[name] = record
    if also_save_alias and also_save_alias != name:
        data[also_save_alias] = record
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return profiles_path


def load_verified_profile(name: str, profiles_path: Path = PROFILES_FILE) -> dict:
    if not profiles_path.exists():
        raise FileNotFoundError(f"尚未保存任何已验证配置: {profiles_path}")
    data = json.loads(profiles_path.read_text(encoding="utf-8"))
    if name not in data:
        avail = ", ".join(sorted(data.keys())) or "无"
        raise KeyError(f"未找到配置 `{name}`（可用: {avail}）")
    return dict(data[name])


def run_remote_script(
    session: str,
    script_path: Path,
    env_vars: dict[str, str],
    env: dict[str, str],
) -> str:
    cbin = colab_bin(env)
    cmd = [cbin, "exec", "-s", session]
    for k, v in env_vars.items():
        if v is not None and str(v) != "":
            cmd.extend(["--env", f"{k}={v}"])
    cmd.extend(["-f", str(script_path)])

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    rc = proc.wait()
    full_output = "".join(chunks)
    if rc != 0:
        raise RuntimeError(f"远端脚本 `{script_path.name}` 执行失败 (exit={rc})")
    return full_output


def parse_json_block(output: str, start_marker: str, end_marker: str) -> dict:
    m = re.search(rf"{re.escape(start_marker)}\s*(\{{.*?\}})\s*{re.escape(end_marker)}", output, re.S)
    if not m:
        raise RuntimeError(f"未能从远端输出解析 JSON ({start_marker})。")
    return json.loads(m.group(1))


def spawn_background_watcher(session: str, idle_minutes: int, env: dict[str, str]) -> Path:
    log_path = Path(f"/tmp/colab_watch_{session}.log")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "watch",
        "--session",
        session,
        "--idle-minutes",
        str(idle_minutes),
    ]
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"🛡️ 已启动后台保活与闲置自动关机守护 (PID={proc.pid}, 闲置>{idle_minutes}分钟自动 stop, 日志: {log_path})")
    return log_path


def prefetch_model_to_drive(
    *,
    hf_repo: str,
    hf_file: str,
    kind: str,
    subdir: str,
    hf_token: str,
    cpu_session: str,
    stop_after: bool,
    env: dict[str, str],
    drive_comfy_models: str = "",
    skip_mmproj: bool = False,
) -> dict:
    """Boot a CPU runtime, prefetch HF weights into Google Drive, and optionally stop CPU."""
    print(f"📥 [预下载阶段] 先启用 `{cpu_session}` (CPU) 下载 `{hf_repo}` 到 Google Drive，节省 GPU 算力...")
    ensure_session(cpu_session, "CPU", True, env)
    ensure_drive_mounted(cpu_session, env)

    env_vars = {
        "HF_REPO": hf_repo,
        "HF_FILE": hf_file,
        "HF_TOKEN": hf_token,
        "MODEL_KIND": kind,
        "COMFY_SUBDIR": subdir,
        "SKIP_MMPROJ": "1" if skip_mmproj else "0",
    }
    if drive_comfy_models:
        env_vars["DRIVE_COMFY_MODELS"] = drive_comfy_models
    output = run_remote_script(
        cpu_session,
        SCRIPTS_DIR / "prefetch_to_drive.py",
        env_vars,
        env,
    )
    result = parse_json_block(
        output, "=== PREFETCH_JSON_START ===", "=== PREFETCH_JSON_END ==="
    )
    sync_notebook_to_drive(cpu_session, env)
    if stop_after:
        stop_session(cpu_session, env)
    return result


def cmd_prefetch(args: argparse.Namespace) -> int:
    env = get_colab_env()
    ensure_colab_auth(env)
    token = load_hf_token(args.hf_token)
    result = prefetch_model_to_drive(
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        kind=args.kind,
        subdir=args.subdir,
        hf_token=token,
        cpu_session=args.session,
        stop_after=not args.no_stop,
        env=env,
        skip_mmproj=args.no_mmproj,
    )
    print("\n🎉 CPU 预下载至 Google Drive 完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_llm(args: argparse.Namespace) -> int:
    env = get_colab_env()
    sessions_out = ensure_colab_auth(env)
    hf_token = load_hf_token(args.hf_token)

    resolved_model_path = args.model_path.strip()
    resolved_mmproj_path = args.mmproj_path.strip()

    if args.hf_repo and not resolved_model_path and not args.no_prefetch:
        if not session_exists(sessions_out, args.session):
            pf = prefetch_model_to_drive(
                hf_repo=args.hf_repo,
                hf_file=args.hf_file,
                kind="llm",
                subdir="",
                hf_token=hf_token,
                cpu_session=args.cpu_session,
                stop_after=True,
                env=env,
                skip_mmproj=args.no_mmproj,
            )
            if pf.get("FINAL_PATH"):
                resolved_model_path = str(pf["FINAL_PATH"])
                print(f"🧭 将直接加载已预下载到网盘的模型：{resolved_model_path}")
            if pf.get("MMPROJ_PATH") and not resolved_mmproj_path:
                resolved_mmproj_path = str(pf["MMPROJ_PATH"])
                print(f"👁️ 检测到视觉投影文件 (mmproj)：{resolved_mmproj_path}")

    ensure_session(args.session, args.gpu, not args.no_high_mem, env)
    ensure_drive_mounted(args.session, env)
    sync_notebook_to_drive(args.session, env)

    print("🚀 [3/4] 在 Colab 部署 CUDA llama-server 并自适应拉满上下文...")
    remote_env = {
        "PORT": str(args.port),
        "HOST": "127.0.0.1",
        "MODEL_PATH": resolved_model_path,
        "HF_REPO": args.hf_repo,
        "HF_FILE": args.hf_file,
        "HF_TOKEN": hf_token,
        "ALIAS": args.model_id,
        "FORCE_CTX": str(args.force_ctx),
        "MAX_OUTPUT_CAP": str(args.max_output),
        "MMPROJ_PATH": resolved_mmproj_path,
        "NO_MMPROJ": "1" if args.no_mmproj else "0",
    }
    output = run_remote_script(args.session, SCRIPTS_DIR / "serve_llamacpp.py", remote_env, env)
    info = parse_json_block(output, "=== RESULT_JSON_START ===", "=== RESULT_JSON_END ===")

    if not args.no_sync_opencode:
        print("🔄 [4/4] 探测模型并同步上下文至 OpenCode...")
        sync_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "sync_opencode.py"),
            "--url",
            info["BASE_URL"],
            "--provider",
            args.provider,
            "--provider-name",
            f"Google Colab ({info['GPU_NAME']} llama.cpp)",
            "--model-id",
            args.model_id or info["MODEL_ID"],
            "--display-name",
            args.display_name or f"{info['MODEL_ID']} ({info['CTX_SIZE'] // 1024}K @ {info['GPU_NAME']})",
            "--context",
            str(info["CTX_SIZE"]),
            "--output",
            str(info["MAX_OUTPUT"]),
        ]
        if info.get("VISION"):
            sync_cmd.append("--vision")
        if args.no_default:
            sync_cmd.append("--no-default")
        subprocess.run(sync_cmd, env=env, check=True)

    profile_payload = {
        "kind": "llm",
        "session": args.session,
        "gpu": args.gpu,
        "no_high_mem": bool(args.no_high_mem),
        "port": int(args.port),
        "model_path": resolved_model_path or info.get("MODEL_PATH", ""),
        "mmproj_path": resolved_mmproj_path or info.get("MMPROJ_PATH", ""),
        "no_mmproj": bool(args.no_mmproj),
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
        "model_id": args.model_id or info.get("MODEL_ID", ""),
        "display_name": args.display_name,
        "provider": args.provider,
        "force_ctx": int(args.force_ctx),
        "max_output": int(args.max_output),
        "no_sync_opencode": bool(args.no_sync_opencode),
        "no_default": bool(args.no_default),
        "auto_watch_idle": int(args.auto_watch_idle),
    }
    pfile = save_verified_profile(
        args.profile or "last-llm", profile_payload, also_save_alias="last-llm"
    )
    print(f"💾 已固化本次已验证参数到 `{pfile}` (下次可直接 `run-profile --name {args.profile or 'last-llm'}`)。")

    if args.auto_watch_idle > 0:
        spawn_background_watcher(args.session, args.auto_watch_idle, env)

    print("\n🎉 全部就绪！大模型服务摘要：")
    print(f"  - GPU 型号: {info['GPU_NAME']} (VRAM {info['VRAM_GB']} GB / RAM {info['RAM_GB']} GB)")
    print(f"  - 模型路径: {info['MODEL_PATH']} ({info['MODEL_GB']} GB)")
    if info.get("MMPROJ_PATH"):
        print(f"  - 视觉多模态: 已启用 ({info['MMPROJ_PATH']})")
    print(f"  - 上下文长度: {info['CTX_SIZE']} tokens (Max Output: {info['MAX_OUTPUT']})")
    print(f"  - OpenAI Base URL: {info['BASE_URL']}")
    print(f"  - Model ID: {info['MODEL_ID']}")
    return 0


def cmd_comfyui(args: argparse.Namespace) -> int:
    env = get_colab_env()
    sessions_out = ensure_colab_auth(env)
    hf_token = load_hf_token(args.hf_token)

    if args.ensure_model and not args.no_prefetch and not session_exists(sessions_out, args.session):
        for spec in args.ensure_model:
            parts = spec.split(":", 2)
            repo = parts[0].strip()
            hf_file = parts[1].strip() if len(parts) > 1 else ""
            subdir = parts[2].strip() if len(parts) > 2 else ""
            prefetch_model_to_drive(
                hf_repo=repo,
                hf_file=hf_file,
                kind="comfyui",
                subdir=subdir,
                hf_token=hf_token,
                cpu_session=args.cpu_session,
                stop_after=False,
                env=env,
                drive_comfy_models=args.drive_models,
            )
        stop_session(args.cpu_session, env)

    ensure_session(args.session, args.gpu, not args.no_high_mem, env)
    ensure_drive_mounted(args.session, env)
    sync_notebook_to_drive(args.session, env)

    print("🎨 [3/4] 正在启动 ComfyUI（含 custom_nodes 快照恢复）并建立 HTTP/2 公网隧道...")
    remote_env = {
        "PORT": str(args.port),
        "HOST": "0.0.0.0",
        "DRIVE_MODELS_DIR": args.drive_models,
        "DRIVE_OUTPUT_DIR": args.drive_output,
        "INSTALL_NODES": ",".join(args.install_node),
        "SAVE_NODES_SNAPSHOT": "1" if args.save_nodes_snapshot else "0",
    }
    output = run_remote_script(args.session, SCRIPTS_DIR / "serve_comfyui.py", remote_env, env)
    info = parse_json_block(output, "=== RESULT_JSON_START ===", "=== RESULT_JSON_END ===")

    profile_payload = {
        "kind": "comfyui",
        "session": args.session,
        "gpu": args.gpu,
        "no_high_mem": bool(args.no_high_mem),
        "port": int(args.port),
        "drive_models": args.drive_models,
        "drive_output": args.drive_output,
        "ensure_model": list(args.ensure_model),
        "install_node": list(args.install_node),
        "save_nodes_snapshot": bool(args.save_nodes_snapshot),
        "auto_watch_idle": int(args.auto_watch_idle),
    }
    pfile = save_verified_profile(
        args.profile or "last-comfyui", profile_payload, also_save_alias="last-comfyui"
    )
    print(f"💾 已固化本次已验证配置到 `{pfile}` (下次可直接 `run-profile --name {args.profile or 'last-comfyui'}`)。")

    if args.auto_watch_idle > 0:
        spawn_background_watcher(args.session, args.auto_watch_idle, env)

    print("\n🎉 ComfyUI 全部就绪：")
    print(f"  - GPU 型号: {info['GPU_NAME']} (VRAM {info['VRAM_GB']} GB)")
    print(f"  - WebUI 地址: {info['PUBLIC_URL']}")
    print(f"  - API Prompt: {info['API_PROMPT_URL']}")
    print(f"  - 输出持久化: {info['OUTPUT_DIR']}")
    if info.get("CUSTOM_NODES"):
        print(f"  - 已载入第三方节点: {', '.join(info['CUSTOM_NODES'])}")
    return 0


def cmd_run_profile(args: argparse.Namespace) -> int:
    prof = load_verified_profile(args.name)
    kind = prof.get("kind", "")
    print(f"⚡ 正在直接按已验证档案 `{args.name}` ({kind}) 启动...")

    if kind == "llm":
        ns = argparse.Namespace(
            session=args.session or prof.get("session", "llm-server"),
            gpu=args.gpu or prof.get("gpu", "AUTO"),
            no_high_mem=bool(prof.get("no_high_mem", False)),
            port=int(prof.get("port", 8080)),
            model_path=prof.get("model_path", ""),
            mmproj_path=prof.get("mmproj_path", ""),
            no_mmproj=bool(prof.get("no_mmproj", False)),
            hf_repo=prof.get("hf_repo", ""),
            hf_file=prof.get("hf_file", ""),
            hf_token="",
            no_prefetch=False,
            cpu_session="cpu-dl",
            model_id=prof.get("model_id", ""),
            display_name=prof.get("display_name", ""),
            provider=prof.get("provider", "colab-llama"),
            force_ctx=int(prof.get("force_ctx", 0)),
            max_output=int(prof.get("max_output", 16384)),
            no_sync_opencode=bool(prof.get("no_sync_opencode", False)),
            no_default=bool(prof.get("no_default", False)),
            profile=args.name,
            auto_watch_idle=int(prof.get("auto_watch_idle", 0)),
        )
        return cmd_llm(ns)

    if kind == "comfyui":
        ns = argparse.Namespace(
            session=args.session or prof.get("session", "comfy-prod"),
            gpu=args.gpu or prof.get("gpu", "AUTO"),
            no_high_mem=bool(prof.get("no_high_mem", False)),
            port=int(prof.get("port", 8188)),
            drive_models=prof.get("drive_models", "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models"),
            drive_output=prof.get("drive_output", "/content/drive/MyDrive/H3_Outputs"),
            ensure_model=list(prof.get("ensure_model", [])),
            hf_token="",
            cpu_session="cpu-dl",
            no_prefetch=False,
            install_node=list(prof.get("install_node", [])),
            save_nodes_snapshot=bool(prof.get("save_nodes_snapshot", False)),
            profile=args.name,
            auto_watch_idle=int(prof.get("auto_watch_idle", 0)),
        )
        return cmd_comfyui(ns)

    raise RuntimeError(f"未知档案类型: {kind}")


def cmd_list_profiles(args: argparse.Namespace) -> int:
    if not PROFILES_FILE.exists():
        print("⚠️ 尚无已验证配置档案。")
        return 0
    data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    for name, info in data.items():
        print(
            f"- {name}: kind={info.get('kind')} session={info.get('session')} "
            f"gpu={info.get('gpu')} saved_at={info.get('saved_at', '')}"
        )
    return 0


def cmd_export_notebook(args: argparse.Namespace) -> int:
    src = next((p for p in NOTEBOOK_CANDIDATES if p.exists()), None)
    if not src:
        raise FileNotFoundError("未找到内置 `Colab_万能一键启动器.ipynb` 模板。")
    dst = Path(args.out).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    print(f"📓 已输出 Colab 一键启动文件: {dst}")
    if args.push_session:
        env = get_colab_env()
        ensure_colab_auth(env)
        ensure_drive_mounted(args.push_session, env)
        sync_notebook_to_drive(args.push_session, env)
    return 0


def _probe_url_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, context=get_ssl_context(), timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def cmd_status(args: argparse.Namespace) -> int:
    env = get_colab_env()
    sessions_out = ensure_colab_auth(env)
    print("=== 1. Colab 活跃会话 ===")
    print(sessions_out.strip() or "(当前无活跃会话)")

    targets = [args.session] if args.session else parse_active_sessions(sessions_out)
    cbin = colab_bin(env)
    probe_py = r'''
import json, os, re, shutil, subprocess
from pathlib import Path

gpu = "CPU"
vram_gb = 0.0
gpu_util = 0
try:
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, stderr=subprocess.DEVNULL
    ).strip().splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    gpu, vram_gb, gpu_util = parts[0], round(float(parts[1]) / 1024.0, 2), int(float(parts[2]))
except Exception:
    pass

ram_gb = 0.0
try:
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text())
    if m:
        ram_gb = round(int(m.group(1)) / (1024.0 * 1024.0), 2)
except Exception:
    pass

drive_free = None
if Path("/content/drive/MyDrive").exists():
    drive_free = round(shutil.disk_usage("/content/drive/MyDrive").free / 1024**3, 2)

ps_out = subprocess.check_output(["ps", "-eo", "args"], text=True, stderr=subprocess.DEVNULL)
running = []
if "llama-server" in ps_out:
    running.append("llama-server")
if "ComfyUI/main.py" in ps_out or "main.py --listen" in ps_out:
    running.append("comfyui")

urls = []
for log in sorted(Path("/tmp").glob("cloudflared_*.log")):
    txt = log.read_text(errors="ignore")
    found = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
    if found:
        urls.append(found[-1])

print("STATUS_JSON:" + json.dumps({
    "gpu": gpu, "vram_gb": vram_gb, "gpu_util": gpu_util,
    "ram_gb": ram_gb, "drive_free_gb": drive_free,
    "services": running, "urls": urls,
}, ensure_ascii=False))
'''
    if targets:
        print("\n=== 2. 远端节点实测体检 ===")
    for sess in targets:
        res = subprocess.run([cbin, "exec", "-s", sess, probe_py], env=env, capture_output=True, text=True, check=False)
        m = re.search(r"STATUS_JSON:(\{.*\})", res.stdout)
        if not m:
            print(f"  - [{sess}] 暂无法探测内部状态 (exit={res.returncode})")
            continue
        st = json.loads(m.group(1))
        url_status = []
        for u in st.get("urls", []):
            ok = _probe_url_ok(f"{u}/system_stats") or _probe_url_ok(f"{u}/v1/models")
            url_status.append(f"{u} ({'✅在线' if ok else '❌不通'})")
        print(
            f"  - [{sess}] GPU={st['gpu']} ({st['vram_gb']}GB, 利用率={st['gpu_util']}%) | "
            f"RAM={st['ram_gb']}GB | Drive剩余={st['drive_free_gb']}GB | 服务={st['services'] or '无'}"
        )
        for us in url_status:
            print(f"      隧道: {us}")

    opencode_cfg = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    print("\n=== 3. OpenCode 当前配置 ===")
    if opencode_cfg.exists():
        sys.path.insert(0, str(SCRIPTS_DIR))
        import sync_opencode  # noqa: E402

        cfg = json.loads(sync_opencode.strip_jsonc(opencode_cfg.read_text(encoding="utf-8")))
        default_model = cfg.get("model", "")
        print(f"  - 默认模型: {default_model or '(未设置)'}")
        if "/" in default_model:
            prov = default_model.split("/", 1)[0]
            base_url = cfg.get("provider", {}).get(prov, {}).get("options", {}).get("baseURL", "")
            if base_url:
                alive = _probe_url_ok(f"{base_url.rstrip('/')}/models")
                print(f"  - 对应地址: {base_url} ({'✅连通正常' if alive else '❌已失效，请运行 reconnect'})")
    else:
        print("  - 未发现 ~/.config/opencode/opencode.jsonc")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    env = get_colab_env()
    sessions_out = ensure_colab_auth(env)
    targets = [args.session] if args.session else parse_active_sessions(sessions_out)
    if not targets:
        print("✅ 当前没有任何活跃的 Colab 会话需要关闭。")
        return 0
    for sess in targets:
        stop_session(sess, env)
    print(f"🌙 已全部安全停机 ({len(targets)} 个会话)，算力计费已停止。")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    env = get_colab_env()
    ensure_colab_auth(env)
    cbin = colab_bin(env)
    idle_limit = int(args.idle_minutes * 60)
    interval = max(10, int(args.interval_seconds))
    idle_sec = 0
    checks = 0

    probe_py = r'''
import json, subprocess, time, urllib.request
from pathlib import Path

Path("/tmp/.colab_keepalive").write_text(str(time.time()))
gpu_util = 0
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, stderr=subprocess.DEVNULL
    ).strip()
    if out:
        gpu_util = int(float(out.splitlines()[0].strip()))
except Exception:
    pass

active_queue = 0
for port in (8188,):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=2) as r:
            q = json.loads(r.read().decode("utf-8"))
            active_queue += len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    except Exception:
        pass

now = time.time()
ages = []
for p in list(Path("/tmp").glob("llama_server_*.log")) + list(Path("/tmp").glob("comfyui_*.log")):
    try:
        ages.append(now - p.stat().st_mtime)
    except Exception:
        pass
recent_age = round(min(ages), 1) if ages else 999999.0
print("WATCH_JSON:" + json.dumps({"gpu_util": gpu_util, "active_queue": active_queue, "recent_log_age_sec": recent_age}))
'''
    print(f"👀 开始守护会话 `{args.session}`：每 {interval}s 心跳一次，连续闲置 {args.idle_minutes} 分钟将自动停机。")
    while True:
        checks += 1
        res = subprocess.run([cbin, "exec", "-s", args.session, probe_py], env=env, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"⚠️ 会话 `{args.session}` 已不可达或已提前关闭，守护退出。")
            return 0
        m = re.search(r"WATCH_JSON:(\{.*\})", res.stdout)
        data = json.loads(m.group(1)) if m else {"gpu_util": 0, "active_queue": 0, "recent_log_age_sec": 999999.0}
        is_busy = (
            data.get("gpu_util", 0) > 3
            or data.get("active_queue", 0) > 0
            or data.get("recent_log_age_sec", 999999.0) <= interval
        )
        if is_busy:
            idle_sec = 0
        else:
            idle_sec += interval

        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] session={args.session} "
            f"gpu={data.get('gpu_util')}% queue={data.get('active_queue')} "
            f"idle={idle_sec}/{idle_limit}s",
            flush=True,
        )
        if idle_sec >= idle_limit:
            print(f"⏰ 会话 `{args.session}` 已连续闲置超过 {args.idle_minutes} 分钟，正在自动释放算力...")
            stop_session(args.session, env)
            if not args.no_voice and sys.platform == "darwin":
                subprocess.run(["say", "Colab 算力已因闲置自动释放"], check=False)
            return 0
        if args.max_checks > 0 and checks >= args.max_checks:
            return 0
        time.sleep(interval)


def cmd_reconnect(args: argparse.Namespace) -> int:
    env = get_colab_env()
    ensure_colab_auth(env)
    port = args.port if args.port else (8080 if args.type == "llm" else 8188)
    print(f"🔗 正在为已有会话 `{args.session}` 重建 HTTP/2 公网隧道 (端口 {port})...")
    output = run_remote_script(
        args.session,
        SCRIPTS_DIR / "expose_tunnel.py",
        {"TARGET_PORT": str(port), "TARGET_HOST": "127.0.0.1"},
        env,
    )
    m = re.search(r"PUBLIC_URL:\s*(https://\S+)", output)
    if not m:
        raise RuntimeError("未能从 expose_tunnel.py 输出解析 PUBLIC_URL。")
    public_url = m.group(1).strip().rstrip("/")
    print(f"✅ 新公网地址: {public_url}")

    if args.type == "llm" and not args.no_sync_opencode:
        sync_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "sync_opencode.py"),
            "--url",
            f"{public_url}/v1",
            "--provider",
            args.provider,
        ]
        if args.model_id:
            sync_cmd.extend(["--model-id", args.model_id])
        subprocess.run(sync_cmd, env=env, check=True)
    elif args.type == "comfyui":
        print(f"  - ComfyUI WebUI: {public_url}")
        print(f"  - API Prompt: {public_url}/prompt")
    return 0


def cmd_sync_opencode(args: argparse.Namespace) -> int:
    sync_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "sync_opencode.py"),
        "--url",
        args.url,
        "--provider",
        args.provider,
        "--provider-name",
        args.provider_name,
        "--model-id",
        args.model_id,
        "--display-name",
        args.display_name,
        "--context",
        str(args.context),
        "--output",
        str(args.output),
    ]
    if args.vision:
        sync_cmd.append("--vision")
    if args.no_default:
        sync_cmd.append("--no-default")
    subprocess.run(sync_cmd, check=True)
    return 0


def cmd_cleanup_drive(args: argparse.Namespace) -> int:
    env = get_colab_env()
    ensure_colab_auth(env)
    ensure_session(args.session, "CPU", False, env)
    ensure_drive_mounted(args.session, env)
    cbin = colab_bin(env)
    script = r'''
import os, shutil
targets = [
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/diffusion_models/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/text_encoders/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/vae/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/loras/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/latent_upscale_models/.cache",
    "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models/split_files",
    "/content/drive/MyDrive/ComfyUI_Models/.cache",
    "/content/drive/MyDrive/ComfyUI_Models/split_files",
    "/content/drive/MyDrive/colab_models/llm/.cache",
    "/content/drive/MyDrive/_hf_prefetch_tmp",
]
for t in targets:
    if os.path.exists(t):
        shutil.rmtree(t, ignore_errors=True)
        print("已清理:", t)
os.system("df -h /content/drive")
'''
    subprocess.run([cbin, "exec", "-s", args.session, script], env=env, check=True)
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "workflow":
        res = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "comfy_workflow_builder.py"), *sys.argv[2:]]
        )
        return res.returncode

    parser = argparse.ArgumentParser(
        description="Turnkey Google Colab + Drive + llama.cpp + ComfyUI + OpenCode launcher"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tok = sub.add_parser("set-hf-token", help="Save HuggingFace token safely for gated models")
    p_tok.add_argument("--token", required=True, help="hf_... access token")
    p_tok.set_defaults(func=cmd_set_hf_token)

    p_pf = sub.add_parser(
        "prefetch",
        help="Start CPU runtime, download HF model to Google Drive cleanly, and stop CPU",
    )
    p_pf.add_argument("--hf-repo", required=True, help="HuggingFace repo ID")
    p_pf.add_argument("--hf-file", default="", help="Optional filename inside repo")
    p_pf.add_argument("--kind", choices=["llm", "comfyui"], default="llm")
    p_pf.add_argument("--subdir", default="", help="ComfyUI target subdir override")
    p_pf.add_argument("--hf-token", default="", help="HuggingFace token override")
    p_pf.add_argument("--session", default="cpu-dl", help="Temporary CPU Colab session name")
    p_pf.add_argument("--no-mmproj", action="store_true", help="Skip companion mmproj vision download")
    p_pf.add_argument("--no-stop", action="store_true", help="Keep CPU session running after download")
    p_pf.set_defaults(func=cmd_prefetch)

    p_llm = sub.add_parser("llm", help="Launch CUDA llama-server on Colab and sync OpenCode")
    p_llm.add_argument("--session", default="llm-server", help="Colab session name")
    p_llm.add_argument("--gpu", default="AUTO", help="AUTO|H100|G4|A100|L4|T4|CPU")
    p_llm.add_argument("--no-high-mem", action="store_true", help="Disable --high-mem flag")
    p_llm.add_argument("--port", type=int, default=8080)
    p_llm.add_argument("--model-path", default="", help="Existing Drive GGUF path")
    p_llm.add_argument("--mmproj-path", default="", help="Existing Drive mmproj path for vision models")
    p_llm.add_argument("--no-mmproj", action="store_true", help="Disable auto mmproj detection/mount")
    p_llm.add_argument("--hf-repo", default="", help="HuggingFace repo ID")
    p_llm.add_argument("--hf-file", default="", help="Exact GGUF filename in repo")
    p_llm.add_argument("--hf-token", default="", help="HuggingFace token")
    p_llm.add_argument("--no-prefetch", action="store_true", help="Skip CPU-first Drive download")
    p_llm.add_argument("--cpu-session", default="cpu-dl", help="CPU session name used for prefetch")
    p_llm.add_argument("--model-id", default="", help="Served model ID alias")
    p_llm.add_argument("--display-name", default="", help="OpenCode display name")
    p_llm.add_argument("--provider", default="colab-llama", help="OpenCode provider key")
    p_llm.add_argument("--force-ctx", type=int, default=0, help="Force context window size")
    p_llm.add_argument("--max-output", type=int, default=16384, help="Max output tokens cap")
    p_llm.add_argument("--no-sync-opencode", action="store_true")
    p_llm.add_argument("--no-default", action="store_true")
    p_llm.add_argument("--profile", default="", help="Save verified launch params under this profile name")
    p_llm.add_argument("--auto-watch-idle", type=int, default=0, help="Auto-stop idle GPU after N minutes")
    p_llm.set_defaults(func=cmd_llm)

    p_comfy = sub.add_parser("comfyui", help="Launch ComfyUI on Colab with Drive models & custom_nodes snapshot")
    p_comfy.add_argument("--session", default="comfy-prod", help="Colab session name")
    p_comfy.add_argument("--gpu", default="AUTO", help="AUTO|H100|G4|A100|L4|T4|CPU")
    p_comfy.add_argument("--no-high-mem", action="store_true")
    p_comfy.add_argument("--port", type=int, default=8188)
    p_comfy.add_argument("--drive-models", default="/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models")
    p_comfy.add_argument("--drive-output", default="/content/drive/MyDrive/H3_Outputs")
    p_comfy.add_argument("--ensure-model", action="append", default=[], help="Prefetch repo[:file[:subdir]] on CPU first")
    p_comfy.add_argument("--install-node", action="append", default=[], help="GitHub owner/repo or URL to install & snapshot")
    p_comfy.add_argument("--save-nodes-snapshot", action="store_true", help="Save custom_nodes snapshot tar.gz to Drive")
    p_comfy.add_argument("--hf-token", default="", help="HuggingFace token")
    p_comfy.add_argument("--cpu-session", default="cpu-dl")
    p_comfy.add_argument("--no-prefetch", action="store_true")
    p_comfy.add_argument("--profile", default="", help="Save verified launch params under this profile name")
    p_comfy.add_argument("--auto-watch-idle", type=int, default=0, help="Auto-stop idle GPU after N minutes")
    p_comfy.set_defaults(func=cmd_comfyui)

    p_runp = sub.add_parser("run-profile", help="Re-launch directly from a saved verified profile")
    p_runp.add_argument("--name", default="last-llm", help="Profile name (e.g. last-llm, last-comfyui, custom)")
    p_runp.add_argument("--session", default="", help="Optional session name override")
    p_runp.add_argument("--gpu", default="", help="Optional GPU override")
    p_runp.set_defaults(func=cmd_run_profile)

    p_lsp = sub.add_parser("list-profiles", help="List saved verified Colab launch profiles")
    p_lsp.set_defaults(func=cmd_list_profiles)

    p_exp = sub.add_parser("export-notebook", help="Export standalone Colab_万能一键启动器.ipynb")
    p_exp.add_argument("--out", default="./Colab_万能一键启动器.ipynb")
    p_exp.add_argument("--push-session", default="", help="Optional mounted Colab session to sync to Drive root")
    p_exp.set_defaults(func=cmd_export_notebook)

    p_st = sub.add_parser("status", help="Inspect active Colab sessions, GPU, tunnels, Drive space, and OpenCode")
    p_st.add_argument("--session", default="", help="Optional specific session to inspect")
    p_st.set_defaults(func=cmd_status)

    p_dn = sub.add_parser("down", help="Stop specified or all active Colab sessions to save compute units")
    p_dn.add_argument("--session", default="", help="Specific session to stop (empty stops all)")
    p_dn.set_defaults(func=cmd_down)

    p_wt = sub.add_parser("watch", help="Keep active Colab kernel alive and auto-stop when idle")
    p_wt.add_argument("--session", required=True, help="Colab session to monitor")
    p_wt.add_argument("--idle-minutes", type=int, default=30, help="Idle minutes before auto-stop")
    p_wt.add_argument("--interval-seconds", type=int, default=60, help="Heartbeat probe interval seconds")
    p_wt.add_argument("--max-checks", type=int, default=0, help="Stop watcher after N checks (0=unlimited)")
    p_wt.add_argument("--no-voice", action="store_true", help="Disable macOS `say` voice alert on auto-stop")
    p_wt.set_defaults(func=cmd_watch)

    p_rec = sub.add_parser("reconnect", help="Rebuild cloudflared HTTP/2 tunnel for a running Colab session")
    p_rec.add_argument("--session", required=True, help="Existing Colab session name")
    p_rec.add_argument("--type", choices=["llm", "comfyui"], default="llm")
    p_rec.add_argument("--port", type=int, default=0, help="Target port (default 8080 llm / 8188 comfyui)")
    p_rec.add_argument("--provider", default="colab-llama", help="OpenCode provider key")
    p_rec.add_argument("--model-id", default="", help="Optional model ID override")
    p_rec.add_argument("--no-sync-opencode", action="store_true")
    p_rec.set_defaults(func=cmd_reconnect)

    p_sync = sub.add_parser("sync-opencode", help="Sync existing Colab llama-server URL to OpenCode")
    p_sync.add_argument("--url", required=True)
    p_sync.add_argument("--provider", default="colab-llama")
    p_sync.add_argument("--provider-name", default="Google Colab (llama.cpp)")
    p_sync.add_argument("--model-id", default="")
    p_sync.add_argument("--display-name", default="")
    p_sync.add_argument("--context", type=int, default=0)
    p_sync.add_argument("--output", type=int, default=16384)
    p_sync.add_argument("--vision", action="store_true")
    p_sync.add_argument("--no-default", action="store_true")
    p_sync.set_defaults(func=cmd_sync_opencode)

    p_clean = sub.add_parser("cleanup-drive", help="Free Google Drive quota using a CPU session")
    p_clean.add_argument("--session", default="cpu-cleaner")
    p_clean.set_defaults(func=cmd_cleanup_drive)

    sub.add_parser(
        "workflow",
        help="Inspect ComfyUI models/templates/nodes, compile UI->API, and run workflows",
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
