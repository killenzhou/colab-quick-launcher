#!/usr/bin/env python3
"""自动探测 Colab llama.cpp / OpenAI 兼容服务端，并将 URL、模型 ID 与最大上下文同步至 ~/.config/opencode/opencode.jsonc。"""

import argparse
import json
import os
import re
import ssl
import urllib.request
from pathlib import Path

DEFAULT_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"


def get_ssl_context() -> ssl.SSLContext:
    if os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = get_ssl_context()


def strip_jsonc(text: str) -> str:
    """安全剥离 JSONC 中的 // 与 /* */ 注释以及尾逗号"""
    out = []
    i = 0
    n = len(text)
    in_str = False
    escape = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
            elif text[i : i + 2] == "//":
                while i < n and text[i] not in ("\n", "\r"):
                    i += 1
            elif text[i : i + 2] == "/*":
                end = text.find("*/", i + 2)
                i = n if end == -1 else end + 2
            else:
                out.append(ch)
                i += 1
    cleaned = "".join(out)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


def probe_endpoint(base_url: str, api_key: str) -> dict:
    """请求 /v1/models 与 /props 获取真实模型名与上下文窗口"""
    base_v1 = base_url.rstrip("/")
    if not base_v1.endswith("/v1"):
        base_v1 = f"{base_v1}/v1"
    root_url = base_v1[:-3]

    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "opencode-sync/1.0"}
    req = urllib.request.Request(f"{base_v1}/models", headers=headers)
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/v1/models 返回非 200 状态: {resp.status}")
        models_payload = json.loads(resp.read().decode("utf-8"))

    models = models_payload.get("data", [])
    detected_model = models[0].get("id") if models else None

    detected_ctx = None
    try:
        req_props = urllib.request.Request(f"{root_url}/props", headers=headers)
        with urllib.request.urlopen(req_props, context=SSL_CTX, timeout=10) as r:
            if r.status == 200:
                props = json.loads(r.read().decode("utf-8"))
                detected_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    except Exception:
        pass

    return {
        "base_url_v1": base_v1,
        "model_id": detected_model,
        "n_ctx": detected_ctx,
    }


def smoke_test_chat(base_url_v1: str, model_id: str, api_key: str) -> str:
    """发送 1-token 对话请求验证真实推理链路畅通"""
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url_v1}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "opencode-sync/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/v1/chat/completions 返回非 200: {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}).get("content", "") or choice.get("text", "")
    return str(msg).strip()


def is_url_alive(base_url_v1: str, api_key: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{base_url_v1.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "opencode-sync/1.0"},
        )
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def save_endpoint_backup(provider_id: str, base_url_v1: str, model_id: str, context_limit: int, output_limit: int) -> None:
    backup_path = Path.home() / ".colab_last_endpoints.json"
    data = {}
    if backup_path.exists():
        try:
            data = json.loads(backup_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[provider_id] = {
        "base_url": base_url_v1,
        "root_url": base_url_v1[:-3] if base_url_v1.endswith("/v1") else base_url_v1,
        "model_id": model_id,
        "context": int(context_limit),
        "output": int(output_limit),
    }
    backup_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sync_config(
    config_path: Path,
    base_url_v1: str,
    provider_id: str,
    provider_name: str,
    model_id: str,
    display_name: str,
    context_limit: int,
    output_limit: int,
    api_key: str,
    set_default: bool,
    prune_stale: bool = False,
    vision: bool = False,
) -> dict:
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")
        cfg = json.loads(strip_jsonc(raw)) if raw.strip() else {}
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {"$schema": "https://opencode.ai/config.json"}

    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    providers = cfg.setdefault("provider", {})

    if prune_stale:
        for old_key in list(providers.keys()):
            if old_key != provider_id and old_key.startswith("colab-"):
                old_url = providers[old_key].get("options", {}).get("baseURL", "")
                old_key_val = providers[old_key].get("options", {}).get("apiKey", "ollama")
                if ".trycloudflare.com" in old_url and not is_url_alive(old_url, old_key_val):
                    print(f"🧹 清理已失效的旧临时 Colab 配置: {old_key} ({old_url})")
                    del providers[old_key]

    model_entry: dict = {
        "name": display_name,
        "limit": {
            "context": int(context_limit),
            "output": int(output_limit),
        },
    }
    if vision:
        model_entry["attachment"] = True
        model_entry["modalities"] = {"input": ["text", "image"], "output": ["text"]}

    providers[provider_id] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": provider_name,
        "options": {
            "baseURL": base_url_v1,
            "apiKey": api_key,
        },
        "models": {
            model_id: model_entry,
        },
    }
    if set_default:
        cfg["model"] = f"{provider_id}/{model_id}"

    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Colab 大模型服务自动同步到 OpenCode 配置")
    parser.add_argument("--url", required=True, help="Colab 暴露的公网根地址或 /v1 地址")
    parser.add_argument("--provider", default="colab-llama", help="OpenCode provider key（默认 colab-llama）")
    parser.add_argument("--provider-name", default="Colab llama.cpp", help="Provider 显示名称")
    parser.add_argument("--model-id", default="", help="模型 ID（留空则自动从 /v1/models 探测）")
    parser.add_argument("--display-name", default="", help="OpenCode 中显示的模型名")
    parser.add_argument("--context", type=int, default=0, help="上下文窗口大小（0 则自动从 /props 探测，默认 131072）")
    parser.add_argument("--output", type=int, default=0, help="最大输出 token（0 则按 context//4 且上限 32768）")
    parser.add_argument("--api-key", default="sk-colab-llamacpp", help="OpenAI 兼容 API Key 占位符")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="opencode.jsonc 路径")
    parser.add_argument("--no-default", action="store_true", help="只添加/更新 provider，不切换默认 model")
    parser.add_argument("--skip-probe", action="store_true", help="跳过远程连通性探测（仅写配置）")
    parser.add_argument("--skip-chat-test", action="store_true", help="跳过 1-token /v1/chat/completions 真实推理冒烟测试")
    parser.add_argument("--no-prune", action="store_true", help="不自动清理失效的旧 colab-* 临时隧道配置")
    parser.add_argument("--vision", action="store_true", help="标记模型支持图像多模态输入 (attachment + image modality)")
    args = parser.parse_args()

    base_v1 = args.url.rstrip("/")
    if not base_v1.endswith("/v1"):
        base_v1 = f"{base_v1}/v1"

    probed_model = None
    probed_ctx = None
    if not args.skip_probe:
        info = probe_endpoint(base_v1, args.api_key)
        base_v1 = info["base_url_v1"]
        probed_model = info["model_id"]
        probed_ctx = info["n_ctx"]
        print(f"✅ 连通性检查通过: {base_v1}/models -> model={probed_model}, n_ctx={probed_ctx}")
        if not args.skip_chat_test and probed_model:
            reply_preview = smoke_test_chat(base_v1, probed_model, args.api_key)
            print(f"✅ 1-token 推理冒烟测试通过 (reply={reply_preview!r})")

    model_id = args.model_id or probed_model or "colab-model"
    context_limit = args.context or probed_ctx or 131072
    output_limit = args.output or min(32768, max(4096, context_limit // 4))
    ctx_k = f"{context_limit // 1024}K" if context_limit >= 1024 else str(context_limit)
    display_name = args.display_name or f"{model_id} ({ctx_k})"

    cfg_path = Path(args.config)
    sync_config(
        config_path=cfg_path,
        base_url_v1=base_v1,
        provider_id=args.provider,
        provider_name=args.provider_name,
        model_id=model_id,
        display_name=display_name,
        context_limit=context_limit,
        output_limit=output_limit,
        api_key=args.api_key,
        set_default=not args.no_default,
        prune_stale=not args.no_prune and not args.skip_probe,
        vision=args.vision,
    )
    save_endpoint_backup(args.provider, base_v1, model_id, context_limit, output_limit)
    print(f"✅ OpenCode 配置已更新: {cfg_path}")
    print(f"   默认模型: {args.provider}/{model_id} | Context={context_limit} | Output={output_limit}")


if __name__ == "__main__":
    main()
