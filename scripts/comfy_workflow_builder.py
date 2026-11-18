#!/usr/bin/env python3
"""ComfyUI workflow discovery, node inspection, UI->API compilation, adaptation, and execution."""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

MODEL_CATEGORIES = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "latent_upscale_models",
    "checkpoints",
    "clip",
    "controlnet",
)

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
PRECISION_BONUS = (
    ("int8_convrot", 0.35),
    ("int8", 0.25),
    ("fp8", 0.20),
    ("nvfp4", 0.15),
    ("bf16", 0.10),
    ("fp16", 0.08),
)

UI_ONLY_NODES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}

KNOWN_NODE_REPOS: list[tuple[str, str]] = [
    ("VHS_", "Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("UnetLoaderGGUF", "city96/ComfyUI-GGUF"),
    ("CLIPLoaderGGUF", "city96/ComfyUI-GGUF"),
    ("DualCLIPLoaderGGUF", "city96/ComfyUI-GGUF"),
    ("LTXV", "Lightricks/ComfyUI-LTXVideo"),
    ("KJNodes", "kijai/ComfyUI-KJNodes"),
    ("GetImageSizeAndCount", "kijai/ComfyUI-KJNodes"),
    ("ImageResizeKJ", "kijai/ComfyUI-KJNodes"),
    ("Impact", "ltdrdata/ComfyUI-Impact-Pack"),
    ("DWPreprocessor", "Fannovel16/comfyui_controlnet_aux"),
    ("CannyEdgePreprocessor", "Fannovel16/comfyui_controlnet_aux"),
    ("WanVideo", "kijai/ComfyUI-WanVideoWrapper"),
    ("Florence2", "kijai/ComfyUI-Florence2"),
]


def get_ssl_context() -> ssl.SSLContext:
    """Return an SSL context using macOS system certs when available."""
    ca_path = "/etc/ssl/cert.pem"
    if os.path.exists(ca_path):
        return ssl.create_default_context(cafile=ca_path)
    return ssl.create_default_context()


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes and /api suffix to obtain ComfyUI root URL."""
    base = url.strip().rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base


def http_request(
    url: str,
    method: str = "GET",
    payload: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    """Perform an HTTP JSON/text request against ComfyUI."""
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    data = None
    if payload is not None:
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = payload

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    ctx = get_ssl_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def http_download_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, context=get_ssl_context(), timeout=timeout) as resp:
        return resp.read()


def fetch_installed_models(base_url: str) -> dict[str, list[str]]:
    """Query ComfyUI /models/<category> endpoints and return installed model lists."""
    base = normalize_base_url(base_url)
    installed: dict[str, list[str]] = {}
    for cat in MODEL_CATEGORIES:
        url = f"{base}/models/{urllib.parse.quote(cat)}"
        try:
            res = http_request(url, timeout=20)
            if isinstance(res, list):
                installed[cat] = [str(x) for x in res]
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                continue
            raise
    return installed


def _tokenize_model_name(name: str) -> list[str]:
    stem = os.path.basename(name)
    for ext in MODEL_EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return [t for t in re.split(r"[^a-z0-9]+", stem.lower()) if t]


def _guess_category_hint(ref_name: str) -> str | None:
    lower = ref_name.lower()
    if "diffusion_models/" in lower or "unet/" in lower:
        return "diffusion_models"
    if "text_encoders/" in lower or "clip/" in lower or "gemma" in lower or "t5" in lower:
        return "text_encoders"
    if "vae/" in lower or "_vae" in lower or "-vae" in lower:
        return "vae"
    if "loras/" in lower or "lora" in lower:
        return "loras"
    if "upscale" in lower:
        return "latent_upscale_models"
    return None


def match_best_model(
    ref_name: str, installed_by_cat: dict[str, list[str]]
) -> tuple[str | None, float, bool]:
    """Match a workflow model reference to the closest installed ComfyUI model entry."""
    ref_clean = ref_name.strip().replace("\\", "/")
    ref_base = os.path.basename(ref_clean).lower()
    ref_lower = ref_clean.lower()

    for _, entries in installed_by_cat.items():
        for entry in entries:
            entry_norm = entry.replace("\\", "/")
            if entry_norm.lower() == ref_lower or os.path.basename(entry_norm).lower() == ref_base:
                return entry, 10.0, True

    ref_tokens = _tokenize_model_name(ref_clean)
    if not ref_tokens:
        return None, 0.0, False
    ref_set = set(ref_tokens)
    cat_hint = _guess_category_hint(ref_clean)

    best_entry: str | None = None
    best_score = 0.0
    ignore_tokens = {"comfy", "fp8", "bf16", "fp16", "int8", "convrot", "nvfp4", "scaled"}

    for cat, entries in installed_by_cat.items():
        for entry in entries:
            cand_tokens = _tokenize_model_name(entry)
            if not cand_tokens:
                continue
            cand_set = set(cand_tokens)
            core_ref = ref_set - ignore_tokens
            core_cand = cand_set - ignore_tokens
            union = core_ref | core_cand
            if not union:
                continue
            inter = core_ref & core_cand
            if not inter:
                continue

            jaccard = len(inter) / len(union)
            prefix_bonus = 0.35 if cand_tokens[0] == ref_tokens[0] else 0.0
            cat_bonus = 0.25 if (cat_hint and cat == cat_hint) else 0.0
            prec_bonus = 0.0
            entry_lower = entry.lower()
            for marker, bonus in PRECISION_BONUS:
                if marker in entry_lower:
                    prec_bonus = bonus
                    break

            role_mismatch_penalty = 0.0
            for role in ("distilled", "dev", "vae", "upscaler", "lora", "t2v", "i2v"):
                if (role in core_ref) != (role in core_cand):
                    role_mismatch_penalty += 0.25

            score = jaccard + prefix_bonus + cat_bonus + prec_bonus - role_mismatch_penalty
            if score > best_score:
                best_score = score
                best_entry = entry

    if best_entry is not None and best_score >= 0.45:
        return best_entry, round(best_score, 3), False
    return None, round(best_score, 3), False


def _looks_like_model_ref(value: str) -> bool:
    s = value.strip()
    if not s or "\n" in s or " " in s.strip():
        return False
    return s.lower().endswith(MODEL_EXTENSIONS)


def adapt_workflow_dict(
    wf_obj: Any, installed_by_cat: dict[str, list[str]]
) -> tuple[Any, list[dict[str, Any]]]:
    """Recursively adapt model file strings inside a ComfyUI JSON object."""
    seen_changes: dict[str, dict[str, Any]] = {}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, str) and _looks_like_model_ref(node):
            matched, score, is_exact = match_best_model(node, installed_by_cat)
            if is_exact and matched:
                rec = {"old": node, "new": matched, "status": "exact", "score": score}
                seen_changes.setdefault(node, rec)
                return matched
            if matched:
                rec = {"old": node, "new": matched, "status": "replaced", "score": score}
                seen_changes.setdefault(node, rec)
                return matched
            rec = {"old": node, "new": node, "status": "missing", "score": score}
            seen_changes.setdefault(node, rec)
            return node
        return node

    adapted = _walk(wf_obj)
    return adapted, list(seen_changes.values())


def _is_api_format(wf_obj: Any) -> bool:
    if not isinstance(wf_obj, dict) or "nodes" in wf_obj:
        return False
    if not wf_obj:
        return False
    return all(
        isinstance(v, dict) and "class_type" in v and "inputs" in v
        for v in wf_obj.values()
    )


def extract_workflow_node_types(wf_obj: Any) -> list[str]:
    """Extract unique executable ComfyUI node types from UI or API workflow JSON."""
    seen: list[str] = []
    if _is_api_format(wf_obj):
        for node in wf_obj.values():
            ct = str(node.get("class_type", "")).strip()
            if ct and ct not in UI_ONLY_NODES and ct not in seen:
                seen.append(ct)
        return seen

    if isinstance(wf_obj, dict) and isinstance(wf_obj.get("nodes"), list):
        for node in wf_obj["nodes"]:
            if not isinstance(node, dict):
                continue
            if node.get("mode") in (2, 4):
                continue
            ct = str(node.get("type", "")).strip()
            if ct and ct not in UI_ONLY_NODES and ct not in seen:
                seen.append(ct)
    return seen


def _suggest_repo_for_node(node_type: str) -> str:
    for prefix, repo in KNOWN_NODE_REPOS:
        if prefix.lower() in node_type.lower():
            return repo
    return ""


def _is_widget_spec(spec_val: Any) -> bool:
    if not isinstance(spec_val, (list, tuple)) or not spec_val:
        return False
    first = spec_val[0]
    if isinstance(first, list):
        return True
    return first in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def ui_to_api_prompt(
    wf_obj: dict[str, Any], object_info: dict[str, Any]
) -> dict[str, Any]:
    """Compile ComfyUI UI graph JSON (nodes + links) to API Prompt format using /object_info."""
    if _is_api_format(wf_obj):
        return copy.deepcopy(wf_obj)

    nodes = wf_obj.get("nodes") or []
    links = wf_obj.get("links") or []

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            nodes_by_id[str(n["id"])] = n

    link_map: dict[Any, tuple[str, int]] = {}
    for lk in links:
        if isinstance(lk, list) and len(lk) >= 5:
            link_map[lk[0]] = (str(lk[1]), int(lk[2]))

    def resolve_source(origin_id: str, origin_slot: int, depth: int = 0) -> tuple[str, int] | None:
        if depth > 20:
            return None
        orig_node = nodes_by_id.get(origin_id)
        if not orig_node:
            return (origin_id, origin_slot)
        ntype = orig_node.get("type")
        if ntype == "Reroute":
            for inp in orig_node.get("inputs") or []:
                lk_id = inp.get("link")
                if lk_id in link_map:
                    up_id, up_slot = link_map[lk_id]
                    return resolve_source(up_id, up_slot, depth + 1)
            return None
        return (origin_id, origin_slot)

    prompt: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node:
            continue
        if node.get("mode") in (2, 4):
            continue
        class_type = str(node.get("type", "")).strip()
        if not class_type or class_type in UI_ONLY_NODES:
            continue

        node_id = str(node["id"])
        inputs: dict[str, Any] = {}

        for inp in node.get("inputs") or []:
            if not isinstance(inp, dict):
                continue
            lk_id = inp.get("link")
            if lk_id is not None and lk_id in link_map:
                up = resolve_source(*link_map[lk_id])
                if up is not None:
                    inputs[inp["name"]] = [up[0], up[1]]

        meta = object_info.get(class_type, {})
        input_defs = meta.get("input", {}) if isinstance(meta, dict) else {}
        ordered_specs: list[tuple[str, Any]] = []
        for sec in ("required", "optional"):
            sec_dict = input_defs.get(sec) or {}
            if isinstance(sec_dict, dict):
                for k, v in sec_dict.items():
                    if _is_widget_spec(v):
                        ordered_specs.append((k, v))

        w_vals = node.get("widgets_values")
        if isinstance(w_vals, dict):
            valid_names = {name for name, _ in ordered_specs}
            for k, v in w_vals.items():
                if (not valid_names or k in valid_names) and k not in inputs:
                    inputs[k] = v
        elif isinstance(w_vals, list):
            idx = 0
            for inp_name, spec_val in ordered_specs:
                if idx >= len(w_vals):
                    break
                val = w_vals[idx]
                idx += 1
                if inp_name not in inputs and val is not None:
                    inputs[inp_name] = val

                extra = spec_val[1] if len(spec_val) > 1 and isinstance(spec_val[1], dict) else {}
                has_cag = extra.get("control_after_generate")
                is_seed = inp_name in ("seed", "noise_seed") or bool(has_cag)
                if is_seed and has_cag is not False and idx < len(w_vals):
                    if w_vals[idx] in ("fixed", "increment", "decrement", "randomize"):
                        idx += 1

        title = node.get("title") or meta.get("display_name") or class_type
        prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": title},
        }
    return prompt


def apply_prompt_overrides(
    api_prompt: dict[str, Any],
    positive: str = "",
    negative: str = "",
    image_name: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    """Patch prompt text, LoadImage target, and seed inside a compiled API prompt."""
    patched = copy.deepcopy(api_prompt)

    if image_name:
        for node in patched.values():
            ct = node.get("class_type", "")
            inps = node.get("inputs", {})
            if ct.startswith("LoadImage") and "image" in inps:
                inps["image"] = image_name

    if seed is not None:
        for node in patched.values():
            inps = node.get("inputs", {})
            for key in ("seed", "noise_seed"):
                if key in inps and not isinstance(inps[key], list):
                    inps[key] = int(seed)

    if positive or negative:
        pos_targets: set[str] = set()
        neg_targets: set[str] = set()
        for node in patched.values():
            inps = node.get("inputs", {})
            for port_name, role_set in (("positive", pos_targets), ("negative", neg_targets)):
                link = inps.get(port_name)
                if isinstance(link, list) and link:
                    role_set.add(str(link[0]))

        def _trace_text_source(nid: str, depth: int = 0) -> str | None:
            if depth > 6 or nid not in patched:
                return None
            n = patched[nid]
            inps = n.get("inputs", {})
            for key in ("text", "prompt", "string", "value"):
                if key in inps:
                    if isinstance(inps[key], list) and inps[key]:
                        sub = _trace_text_source(str(inps[key][0]), depth + 1)
                        if sub:
                            return sub
                    elif isinstance(inps[key], str):
                        return nid
            return None

        text_nodes: list[str] = []
        for nid, node in patched.items():
            inps = node.get("inputs", {})
            ct = node.get("class_type", "")
            if any(k in inps and isinstance(inps[k], str) for k in ("text", "prompt", "string", "value")) and (
                "Text" in ct or "CLIP" in ct or "String" in ct or "Prompt" in ct
            ):
                text_nodes.append(nid)

        pos_ids: set[str] = set()
        neg_ids: set[str] = set()
        for t in pos_targets:
            src = _trace_text_source(t)
            if src:
                pos_ids.add(src)
        for t in neg_targets:
            src = _trace_text_source(t)
            if src:
                neg_ids.add(src)

        for nid in text_nodes:
            title = str(patched[nid].get("_meta", {}).get("title", "")).lower()
            if any(w in title for w in ("negative", "负面", "neg")):
                neg_ids.add(nid)
            elif any(w in title for w in ("positive", "正向", "pos")):
                pos_ids.add(nid)

        remaining = [nid for nid in text_nodes if nid not in pos_ids and nid not in neg_ids]
        for nid in remaining:
            raw_txt = " ".join(
                str(patched[nid]["inputs"].get(k, ""))
                for k in ("text", "prompt", "string", "value")
                if isinstance(patched[nid]["inputs"].get(k), str)
            ).lower()
            if any(w in raw_txt for w in ("worst quality", "lowres", "blurry", "watermark")):
                neg_ids.add(nid)
            elif not pos_ids:
                pos_ids.add(nid)
            elif not neg_ids:
                neg_ids.add(nid)

        def _set_text(nid: str, new_text: str) -> None:
            inps = patched[nid]["inputs"]
            for key in ("text", "prompt", "string", "value"):
                if key in inps and isinstance(inps[key], str):
                    inps[key] = new_text
                    return

        if positive:
            for nid in pos_ids:
                _set_text(nid, positive)
        if negative:
            for nid in neg_ids:
                _set_text(nid, negative)

    return patched


def upload_image(base_url: str, image_path: Path) -> str:
    """Upload a local reference image to ComfyUI /upload/image via multipart/form-data."""
    base = normalize_base_url(base_url)
    boundary = f"----ComfyBoundary{uuid.uuid4().hex}"
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "application/octet-stream"
    file_bytes = image_path.read_bytes()

    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    res = http_request(
        f"{base}/upload/image",
        method="POST",
        payload=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=120,
    )
    if not isinstance(res, dict) or not res.get("name"):
        raise RuntimeError(f"上传参考图失败: {res}")
    subfolder = str(res.get("subfolder", "")).strip()
    name = str(res["name"])
    return f"{subfolder}/{name}" if subfolder else name


def _flatten_templates(raw_index: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(raw_index, list):
        for entry in raw_index:
            if not isinstance(entry, dict):
                continue
            if "templates" in entry and isinstance(entry["templates"], list):
                cat = entry.get("title") or entry.get("category") or ""
                for sub in entry["templates"]:
                    if isinstance(sub, dict):
                        row = dict(sub)
                        row.setdefault("category", cat)
                        items.append(row)
            else:
                items.append(entry)
    elif isinstance(raw_index, dict):
        for key, val in raw_index.items():
            if isinstance(val, list):
                for sub in val:
                    if isinstance(sub, dict):
                        row = dict(sub)
                        row.setdefault("category", key)
                        items.append(row)
    return items


def cmd_inspect_models(args: argparse.Namespace) -> int:
    installed = fetch_installed_models(args.url)
    if args.json:
        print(json.dumps(installed, ensure_ascii=False, indent=2))
        return 0
    total = 0
    for cat, items in installed.items():
        if not items:
            continue
        print(f"[{cat}] ({len(items)})")
        for item in items:
            print(f"  - {item}")
            total += 1
    if total == 0:
        print("⚠️ 未在 ComfyUI 中检测到已安装模型。")
    return 0


def cmd_list_templates(args: argparse.Namespace) -> int:
    base = normalize_base_url(args.url)
    raw = http_request(f"{base}/templates/index.json", timeout=30)
    templates = _flatten_templates(raw)
    kw = (args.keyword or "").strip().lower()
    if kw:
        templates = [t for t in templates if kw in json.dumps(t, ensure_ascii=False).lower()]
    if args.json:
        print(json.dumps(templates, ensure_ascii=False, indent=2))
        return 0
    for tpl in templates:
        name = tpl.get("name", "")
        title = tpl.get("title", "")
        cat = tpl.get("category", "")
        models = tpl.get("models") or []
        models_str = f" models={models}" if models else ""
        print(f"- {name} | {title} [{cat}]{models_str}")
    print(f"共匹配 {len(templates)} 个模板。")
    return 0


def cmd_inspect_node(args: argparse.Namespace) -> int:
    base = normalize_base_url(args.url)
    node_q = urllib.parse.quote(args.node, safe="")
    info = http_request(f"{base}/object_info/{node_q}", timeout=20)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    node_meta = info.get(args.node, info) if isinstance(info, dict) else info
    if not isinstance(node_meta, dict) or not node_meta:
        print(f"❌ 未找到节点 `{args.node}`。", file=sys.stderr)
        return 1
    inputs = node_meta.get("input", {})
    print(f"节点: {args.node}")
    for section in ("required", "optional"):
        sec_data = inputs.get(section, {})
        if sec_data:
            print(f"  [{section}]")
            for k, v in sec_data.items():
                print(f"    - {k}: {v}")
    outputs = node_meta.get("output_name") or node_meta.get("output") or []
    print(f"  [outputs]: {outputs}")
    return 0


def _upload_workflow_to_server(base_url: str, local_data: Any, upload_name: str) -> str:
    base = normalize_base_url(base_url)
    clean_name = os.path.basename(upload_name.strip())
    if not clean_name.endswith(".json"):
        clean_name += ".json"
    encoded_path = urllib.parse.quote(f"workflows/{clean_name}", safe="")
    url = f"{base}/userdata/{encoded_path}?overwrite=true"
    http_request(url, method="POST", payload=local_data, timeout=30)
    return clean_name


def cmd_adapt_template(args: argparse.Namespace) -> int:
    base = normalize_base_url(args.url)
    tpl_name = args.template.strip()
    tpl_file = tpl_name if tpl_name.endswith(".json") else f"{tpl_name}.json"

    tpl_url = f"{base}/templates/{urllib.parse.quote(tpl_file)}"
    wf_data = http_request(tpl_url, timeout=30)
    installed = fetch_installed_models(base)
    adapted, report = adapt_workflow_dict(wf_data, installed)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"✅ 已保存适配后工作流到: {out_path}")
    for row in report:
        print(f"  [{row['status']}] {row['old']} -> {row['new']} (score={row['score']})")

    if args.upload_name:
        remote_name = _upload_workflow_to_server(base, adapted, args.upload_name)
        print(f"☁️ 已同步上传至 ComfyUI WebUI: workflows/{remote_name}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"❌ 文件不存在: {src}", file=sys.stderr)
        return 1
    data = json.loads(src.read_text(encoding="utf-8"))
    upload_name = args.name or src.name
    remote_name = _upload_workflow_to_server(args.url, data, upload_name)
    print(f"☁️ 已上传工作流至 ComfyUI WebUI: workflows/{remote_name}")
    return 0


def cmd_check_nodes(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"❌ 文件不存在: {src}", file=sys.stderr)
        return 1
    wf_data = json.loads(src.read_text(encoding="utf-8"))
    base = normalize_base_url(args.url)
    obj_info = http_request(f"{base}/object_info", timeout=60)
    needed = extract_workflow_node_types(wf_data)
    missing = [nt for nt in needed if nt not in obj_info]
    repos = sorted({r for r in (_suggest_repo_for_node(nt) for nt in missing) if r})

    payload = {
        "total_nodes": len(needed),
        "missing_nodes": missing,
        "suggested_repos": repos,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not missing else 2

    if not missing:
        print(f"✅ 工作流所需的 {len(needed)} 个节点已全部就绪，无缺失自定义插件。")
        return 0

    print(f"⚠️ 工作流缺失 {len(missing)} 个节点：")
    for nt in missing:
        sug = _suggest_repo_for_node(nt)
        extra = f" -> 建议插件: {sug}" if sug else ""
        print(f"  - {nt}{extra}")
    if repos:
        flags = " ".join(f"--install-node {r}" for r in repos)
        print(f"💡 可重启/启动 ComfyUI 时自动安装并写入快照：\n   colab_oneclick.py comfyui {flags}")
    return 2


def cmd_compile_api(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"❌ 文件不存在: {src}", file=sys.stderr)
        return 1
    wf_data = json.loads(src.read_text(encoding="utf-8"))
    base = normalize_base_url(args.url)
    installed = fetch_installed_models(base)
    adapted, report = adapt_workflow_dict(wf_data, installed)
    obj_info = http_request(f"{base}/object_info", timeout=60)
    api_prompt = ui_to_api_prompt(adapted, obj_info)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(api_prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"✅ 已编译为可执行 API Prompt JSON ({len(api_prompt)} 个节点): {out_path}")
    for row in report:
        if row["status"] != "exact":
            print(f"  [{row['status']}] {row['old']} -> {row['new']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"❌ 工作流文件不存在: {src}", file=sys.stderr)
        return 1
    wf_data = json.loads(src.read_text(encoding="utf-8"))
    base = normalize_base_url(args.url)

    installed = fetch_installed_models(base)
    adapted, report = adapt_workflow_dict(wf_data, installed)
    for row in report:
        if row["status"] == "missing":
            print(f"⚠️ 警告：模型可能缺失 {row['old']}", file=sys.stderr)

    obj_info = http_request(f"{base}/object_info", timeout=60)
    needed = extract_workflow_node_types(adapted)
    missing_nodes = [nt for nt in needed if nt not in obj_info]
    if missing_nodes:
        print(f"❌ 服务端缺少运行所需节点: {missing_nodes}", file=sys.stderr)
        return 1

    api_prompt = ui_to_api_prompt(adapted, obj_info)
    uploaded_name = ""
    if args.image:
        img_path = Path(args.image).expanduser()
        if not img_path.exists():
            print(f"❌ 参考图不存在: {img_path}", file=sys.stderr)
            return 1
        uploaded_name = upload_image(base, img_path)
        print(f"📤 已上传参考图: {uploaded_name}")

    api_prompt = apply_prompt_overrides(
        api_prompt,
        positive=args.prompt,
        negative=args.negative,
        image_name=uploaded_name,
        seed=args.seed,
    )

    client_id = uuid.uuid4().hex
    res = http_request(
        f"{base}/prompt",
        method="POST",
        payload={"prompt": api_prompt, "client_id": client_id},
        timeout=60,
    )
    prompt_id = res.get("prompt_id") if isinstance(res, dict) else None
    if not prompt_id:
        print(f"❌ 提交 /prompt 失败: {res}", file=sys.stderr)
        return 1
    print(f"🚀 已提交 ComfyUI 任务: prompt_id={prompt_id}，等待执行完成...")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.timeout
    downloaded: list[Path] = []

    while time.time() < deadline:
        hist = http_request(f"{base}/history/{urllib.parse.quote(str(prompt_id))}", timeout=30)
        entry = hist.get(prompt_id) if isinstance(hist, dict) else None
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                print(f"❌ ComfyUI 执行报错:\n{json.dumps(status, ensure_ascii=False, indent=2)}", file=sys.stderr)
                return 1
            outputs = entry.get("outputs") or {}
            if outputs or status.get("completed"):
                for node_out in outputs.values():
                    if not isinstance(node_out, dict):
                        continue
                    for media_key in ("images", "videos", "gifs", "audio"):
                        for item in node_out.get(media_key) or []:
                            if not isinstance(item, dict) or not item.get("filename"):
                                continue
                            qs = urllib.parse.urlencode(
                                {
                                    "filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output"),
                                }
                            )
                            raw_bytes = http_download_bytes(f"{base}/view?{qs}")
                            local_file = out_dir / os.path.basename(item["filename"])
                            local_file.write_bytes(raw_bytes)
                            downloaded.append(local_file)
                            print(f"💾 已下载成品: {local_file}")
                break
        time.sleep(3)
    else:
        print(f"❌ 等待任务 {prompt_id} 超时（{args.timeout}s）。", file=sys.stderr)
        return 1

    print(f"🎉 执行完成，共拉回 {len(downloaded)} 个成品至 {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect ComfyUI models/templates/nodes, compile UI->API, and run workflows."
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_models = sub.add_parser("inspect-models", help="List installed ComfyUI models")
    p_models.add_argument("--url", required=True, help="ComfyUI base URL")
    p_models.add_argument("--json", action="store_true")
    p_models.set_defaults(func=cmd_inspect_models)

    p_tpl = sub.add_parser("list-templates", help="Search ComfyUI built-in templates")
    p_tpl.add_argument("--url", required=True, help="ComfyUI base URL")
    p_tpl.add_argument("--keyword", default="", help="Optional search keyword")
    p_tpl.add_argument("--json", action="store_true")
    p_tpl.set_defaults(func=cmd_list_templates)

    p_node = sub.add_parser("inspect-node", help="Inspect ComfyUI node schema via /object_info")
    p_node.add_argument("--url", required=True, help="ComfyUI base URL")
    p_node.add_argument("--node", required=True, help="ComfyUI node class name")
    p_node.add_argument("--json", action="store_true")
    p_node.set_defaults(func=cmd_inspect_node)

    p_adapt = sub.add_parser(
        "adapt-template",
        help="Download a ComfyUI template, map model names to installed files, and optionally upload",
    )
    p_adapt.add_argument("--url", required=True, help="ComfyUI base URL")
    p_adapt.add_argument("--template", required=True, help="Template name (with or without .json)")
    p_adapt.add_argument("--out", required=True, help="Local output JSON file path")
    p_adapt.add_argument("--upload-name", default="", help="Optional WebUI workflows/ target name")
    p_adapt.set_defaults(func=cmd_adapt_template)

    p_up = sub.add_parser("upload", help="Upload a local UI workflow JSON to ComfyUI userdata/workflows")
    p_up.add_argument("--url", required=True, help="ComfyUI base URL")
    p_up.add_argument("--file", required=True, help="Local workflow JSON file")
    p_up.add_argument("--name", default="", help="Target filename in ComfyUI workflows/")
    p_up.set_defaults(func=cmd_upload)

    p_chk = sub.add_parser("check-nodes", help="Check whether custom nodes required by a workflow are installed")
    p_chk.add_argument("--url", required=True, help="ComfyUI base URL")
    p_chk.add_argument("--file", required=True, help="Local UI or API workflow JSON file")
    p_chk.add_argument("--json", action="store_true")
    p_chk.set_defaults(func=cmd_check_nodes)

    p_cmp = sub.add_parser("compile-api", help="Convert UI workflow JSON into ComfyUI /prompt API JSON")
    p_cmp.add_argument("--url", required=True, help="ComfyUI base URL")
    p_cmp.add_argument("--file", required=True, help="Input UI workflow JSON file")
    p_cmp.add_argument("--out", required=True, help="Output API prompt JSON file")
    p_cmp.set_defaults(func=cmd_compile_api)

    p_run = sub.add_parser("run", help="Execute a UI or API workflow JSON on ComfyUI and download outputs")
    p_run.add_argument("--url", required=True, help="ComfyUI base URL")
    p_run.add_argument("--file", required=True, help="Workflow JSON file (UI or API format)")
    p_run.add_argument("--prompt", default="", help="Optional positive prompt override")
    p_run.add_argument("--negative", default="", help="Optional negative prompt override")
    p_run.add_argument("--image", default="", help="Optional local reference image to upload")
    p_run.add_argument("--seed", type=int, default=None, help="Optional seed override")
    p_run.add_argument("--out-dir", default="./outputs", help="Local folder to download outputs")
    p_run.add_argument("--timeout", type=int, default=900, help="Timeout seconds")
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        print(f"❌ HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"❌ 无法连接到 ComfyUI ({args.url}): {exc.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
