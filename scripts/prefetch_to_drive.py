#!/usr/bin/env python3
"""在 Colab (推荐 CPU 运行时) 中执行：自动检索仓库文件、校验 Google Drive 剩余空间、下载模型至网盘并清理 .cache。"""

import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

HF_REPO = os.environ.get("HF_REPO", "").strip()
HF_FILE = os.environ.get("HF_FILE", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
SKIP_MMPROJ = os.environ.get("SKIP_MMPROJ", "0") == "1"
MODEL_KIND = os.environ.get("MODEL_KIND", "llm").strip().lower()  # llm | comfyui
COMFY_SUBDIR = os.environ.get("COMFY_SUBDIR", "").strip()
DRIVE_LLM_DIR = Path(os.environ.get("DRIVE_LLM_DIR", "/content/drive/MyDrive/colab_models/llm"))
DRIVE_COMFY_MODELS = Path(os.environ.get("DRIVE_COMFY_MODELS", "/content/drive/MyDrive/ComfyUI_H3/ComfyUI/models"))
TMP_DIR = Path("/content/drive/MyDrive/_hf_prefetch_tmp")


def fetch_repo_tree(repo_id: str, token: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
    headers = {"User-Agent": "colab-prefetch/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    return [x for x in items if isinstance(x, dict) and x.get("type") == "file"]


def expand_sharded_files(chosen_path: str, all_files: list[dict]) -> list[dict]:
    """若选中的文件是分片文件 (*-00001-of-00004.gguf)，自动收集全部同组分片"""
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})(\.[^.]+)$", chosen_path)
    if not m:
        return [f for f in all_files if f["path"] == chosen_path]
    prefix, _, total, ext = m.groups()
    pat = re.compile(rf"^{re.escape(prefix)}-\d{{5}}-of-{total}{re.escape(ext)}$")
    shards = sorted([f for f in all_files if pat.match(f["path"])], key=lambda x: x["path"])
    return shards or [f for f in all_files if f["path"] == chosen_path]


def pick_mmproj_file(all_files: list[dict]) -> list[dict]:
    """从 GGUF 仓库中挑选配套的多模态视觉投影文件 (mmproj)"""
    mm_cands = [
        f
        for f in all_files
        if f["path"].lower().endswith(".gguf") and "mmproj" in f["path"].lower()
    ]
    if not mm_cands:
        return []
    for tag in ("f16", "bf16", "q8_0", "f32"):
        hits = [f for f in mm_cands if tag in f["path"].lower()]
        if hits:
            return [sorted(hits, key=lambda x: len(x["path"]))[0]]
    return [sorted(mm_cands, key=lambda x: len(x["path"]))[0]]


def pick_best_files(all_files: list[dict], kind: str, explicit_file: str) -> list[dict]:
    if explicit_file:
        matches = [f for f in all_files if f["path"] == explicit_file]
        if matches:
            selected = expand_sharded_files(explicit_file, all_files)
        else:
            # 容错：按文件名后缀匹配（例如用户只传 basename 而非完整子路径）
            base_matches = [f for f in all_files if Path(f["path"]).name == explicit_file]
            if base_matches:
                selected = expand_sharded_files(base_matches[0]["path"], all_files)
            else:
                selected = [{"path": explicit_file, "size": 0}]
        if kind == "llm" and not SKIP_MMPROJ:
            existing = {x["path"] for x in selected}
            for mm in pick_mmproj_file(all_files):
                if mm["path"] not in existing:
                    selected.append(mm)
        return selected

    if kind == "llm":
        cands = [
            f for f in all_files
            if f["path"].lower().endswith(".gguf") and "mmproj" not in f["path"].lower()
        ]
        if not cands:
            raise RuntimeError(f"仓库 {HF_REPO} 中未找到 .gguf 模型文件")
        # 优先仅保留单文件或分片的第 1 片再统一展开
        priorities = ["q4_k_m", "iq4_xs", "q5_k_m", "q4_k_s", "q6_k", "q8_0", "q3_k_m"]
        chosen = None
        for p in priorities:
            hits = [f for f in cands if p in f["path"].lower()]
            if hits:
                # 优先选分片00001或单文件
                hits_sorted = sorted(hits, key=lambda x: ("00001-of-" not in x["path"], len(x["path"])))
                chosen = hits_sorted[0]["path"]
                break
        if not chosen:
            chosen = sorted(cands, key=lambda x: ("00001-of-" not in x["path"], x["path"]))[0]["path"]
        selected = expand_sharded_files(chosen, all_files)
        if not SKIP_MMPROJ:
            existing = {x["path"] for x in selected}
            for mm in pick_mmproj_file(all_files):
                if mm["path"] not in existing:
                    selected.append(mm)
        return selected

    # comfyui
    cands = [f for f in all_files if f["path"].lower().endswith(".safetensors")]
    if not cands:
        raise RuntimeError(f"仓库 {HF_REPO} 中未找到 .safetensors 模型文件")
    priorities = ["int8_convrot", "fp8_scaled", "fp8mixed", "fp8_e4m3fn", "nvfp4", "bf16", "fp16"]
    for p in priorities:
        hits = [f for f in cands if p in f["path"].lower()]
        if hits:
            return [hits[0]]
    return [cands[0]]


def resolve_target_path(remote_path: str, kind: str) -> Path:
    if kind == "llm":
        repo_dir = DRIVE_LLM_DIR / HF_REPO.replace("/", "__")
        return repo_dir / remote_path
    clean_rel = remote_path
    if clean_rel.startswith("split_files/"):
        clean_rel = clean_rel[len("split_files/"):]
    if COMFY_SUBDIR:
        return DRIVE_COMFY_MODELS / COMFY_SUBDIR / Path(clean_rel).name
    if "/" in clean_rel:
        return DRIVE_COMFY_MODELS / clean_rel
    return DRIVE_COMFY_MODELS / "diffusion_models" / clean_rel


def report_large_dirs() -> None:
    print("[storage] 当前 Google Drive 占用较大的模型目录：", flush=True)
    for d in [DRIVE_COMFY_MODELS, DRIVE_LLM_DIR]:
        if d.exists():
            subprocess.run(["du", "-sh", *map(str, sorted(d.glob("*"))[:20])], check=False)


def main() -> None:
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Google Drive 未挂载（/content/drive/MyDrive 不存在），请先授权挂载 Drive。")
    if not HF_REPO:
        raise RuntimeError("请设置环境变量 HF_REPO。")

    all_files = fetch_repo_tree(HF_REPO, HF_TOKEN)
    targets = pick_best_files(all_files, MODEL_KIND, HF_FILE)

    planned = []
    missing = []
    for item in targets:
        rpath = item["path"]
        expected_sz = int(item.get("size", 0) or 0)
        tpath = resolve_target_path(rpath, MODEL_KIND)
        planned.append((rpath, expected_sz, tpath))
        if not tpath.exists() or (expected_sz > 0 and tpath.stat().st_size < expected_sz * 0.99):
            missing.append((rpath, expected_sz, tpath))

    primary_files = [p for p in planned if "mmproj" not in p[0].lower()] or planned
    mmproj_files = [p for p in planned if "mmproj" in p[0].lower()]

    if not missing:
        total_gb = round(sum(p[2].stat().st_size for p in planned) / (1024**3), 2)
        free_gb = round(shutil.disk_usage("/content/drive/MyDrive").free / (1024**3), 2)
        result = {
            "STATUS": "ALREADY_EXISTS",
            "FINAL_PATH": str(primary_files[0][2]),
            "MMPROJ_PATH": str(mmproj_files[0][2]) if mmproj_files else "",
            "FILES": [str(p[2]) for p in planned],
            "TOTAL_GB": total_gb,
            "DRIVE_FREE_GB": free_gb,
        }
        print("=== PREFETCH_JSON_START ===", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print("=== PREFETCH_JSON_END ===", flush=True)
        return

    need_bytes = sum(m[1] for m in missing)
    free_bytes = shutil.disk_usage("/content/drive/MyDrive").free
    buffer_bytes = 5 * (1024**3)
    if need_bytes > 0 and free_bytes < need_bytes + buffer_bytes:
        report_large_dirs()
        raise RuntimeError(
            f"Google Drive 剩余空间不足！需要 {need_bytes/1024**3:.2f} GB (+5GB 缓冲)，"
            f"当前仅剩 {free_bytes/1024**3:.2f} GB。请先清理旧缓存或非必要模型。"
        )

    subprocess.run(
        ["bash", "-lc", "uv pip install --system -q huggingface_hub hf_transfer || pip install -q huggingface_hub hf_transfer"],
        check=False,
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    for rpath, _, tpath in missing:
        tpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prefetch] 下载 {HF_REPO}/{rpath} -> {tpath} ...", flush=True)
        dl = hf_hub_download(
            repo_id=HF_REPO,
            filename=rpath,
            local_dir=str(TMP_DIR),
            token=HF_TOKEN or None,
        )
        shutil.move(dl, str(tpath))
        print(f"[prefetch] 已保存: {tpath} ({tpath.stat().st_size/1024**3:.2f} GB)", flush=True)

    # 彻底清理临时目录与 .cache
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for base in (DRIVE_LLM_DIR, DRIVE_COMFY_MODELS):
        if base.exists():
            for c in base.rglob(".cache"):
                if c.is_dir():
                    shutil.rmtree(c, ignore_errors=True)

    total_gb = round(sum(p[2].stat().st_size for p in planned) / (1024**3), 2)
    free_gb = round(shutil.disk_usage("/content/drive/MyDrive").free / (1024**3), 2)
    result = {
        "STATUS": "DOWNLOADED",
        "FINAL_PATH": str(primary_files[0][2]),
        "MMPROJ_PATH": str(mmproj_files[0][2]) if mmproj_files else "",
        "FILES": [str(p[2]) for p in planned],
        "TOTAL_GB": total_gb,
        "DRIVE_FREE_GB": free_gb,
    }
    print("=== PREFETCH_JSON_START ===", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("=== PREFETCH_JSON_END ===", flush=True)


if __name__ == "__main__":
    main()
