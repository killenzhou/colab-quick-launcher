#!/usr/bin/env python3
"""Colab MiniMax-H3 一条龙 API 客户端（1.资产定妆 t2i / 2.多图参考改图 edit / 3.图生视频 i2v）"""

import argparse
import json
import mimetypes
import os
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_URL = os.environ.get(
    "H3_URL", "https://ministries-occasional-despite-thy.trycloudflare.com"
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def get_ssl_context() -> ssl.SSLContext:
    """兼容 macOS Python 默认未安装证书链的情况，优先加载系统 cert.pem"""
    if os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = get_ssl_context()


def http_json(url: str, data: dict | None = None, timeout: int = 30) -> dict:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_image(base_url: str, image_path: str) -> str:
    """上传本地图片到 ComfyUI /upload/image，返回服务端文件名"""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(image_path)
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return res["name"]


def duration_to_frames(duration: float, fps: int = 24) -> int:
    """按 H3 的 17k+5 帧数网格换算总帧数"""
    base = max(5, round(duration * fps))
    return base + (5 - (base % 17)) % 17


def build_h3_image_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    uploaded_images: list[str],
    prefix: str,
) -> dict:
    """构造 H3 图像 API 工作流（t2i / edit）"""
    wf = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"},
        },
        "5": {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
            },
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["5", 1],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": prefix, "images": ["8", 0]},
        },
    }

    for idx, remote_name in enumerate(uploaded_images, start=1):
        node_id = str(100 + idx)
        wf[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": remote_name},
        }
        wf["5"]["inputs"][f"images.image_{idx}"] = [node_id, 0]

    return wf


def build_h3_video_workflow(
    prompt: str,
    duration: float,
    megapixels: float,
    width: int,
    height: int,
    seed: int,
    steps: int,
    first_frame_name: str | None,
    last_frame_name: str | None,
    prefix: str,
) -> dict:
    """构造 H3 视频 API 工作流（支持 i2v / fl2v / t2v，默认开启 4 步 Turbo LoRA）"""
    length = duration_to_frames(duration, fps=24)

    wf = {
        "6": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "121": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["6", 0],
                "lora_name": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                "strength_model": 1.0,
            },
        },
        "13": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "11": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "24": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "104": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["13", 0],
                "vae": ["11", 0],
                "prompt": prompt,
                "width": width if width > 0 else 1344,
                "height": height if height > 0 else 768,
                "length": length,
            },
        },
        "15": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "16": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["121", 0],
                "conditioning": ["104", 0],
            },
        },
        "17": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["121", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16", 0],
                "sampler": ["17", 0],
                "sigmas": ["9", 0],
                "latent_image": ["104", 1],
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["11", 0]},
        },
        "23": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["14", 0], "vae": ["24", 0]},
        },
        "91": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["10", 0],
                "audio": ["23", 0],
                "fps": 24.0,
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["91", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }

    if first_frame_name:
        wf["114"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_frame_name},
        }
        wf["119"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["114", 0],
                "upscale_method": "nearest-exact",
                "megapixels": megapixels,
                "resolution_steps": 32,
            },
        }
        wf["104"]["inputs"]["first_frame"] = ["119", 0]
        if width <= 0 or height <= 0:
            wf["120"] = {
                "class_type": "GetImageSize",
                "inputs": {"image": ["119", 0]},
            }
            wf["104"]["inputs"]["width"] = ["120", 0]
            wf["104"]["inputs"]["height"] = ["120", 1]

    if last_frame_name:
        wf["124"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame_name},
        }
        wf["125"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["124", 0],
                "upscale_method": "nearest-exact",
                "megapixels": megapixels,
                "resolution_steps": 32,
            },
        }
        wf["104"]["inputs"]["last_frame"] = ["125", 0]

    return wf


def run_and_download(base_url: str, workflow: dict, out_dir: str) -> list[str]:
    base_url = base_url.rstrip("/")
    resp = http_json(f"{base_url}/prompt", {"prompt": workflow})
    prompt_id = resp["prompt_id"]
    print(f"⏳ 已提交任务 prompt_id={prompt_id}，等待 A100 渲染...")

    t0 = time.time()
    while True:
        time.sleep(2)
        hist = http_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"服务端执行出错: {json.dumps(status, ensure_ascii=False)}")
            outputs = entry.get("outputs", {})
            if outputs:
                break
        if time.time() - t0 > 900:
            raise TimeoutError("等待生成超时（>15分钟）")

    os.makedirs(out_dir, exist_ok=True)
    saved_files = []
    seen = set()
    for node_out in outputs.values():
        for field in ("images", "videos", "gifs"):
            for item in node_out.get(field, []):
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                key = (item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                if key in seen:
                    continue
                seen.add(key)
                q = urllib.parse.urlencode(
                    {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                )
                local_path = os.path.join(out_dir, item["filename"])
                print(f"⬇️ 正在下载成片 ({field}): {item['filename']} ...")
                with urllib.request.urlopen(
                    f"{base_url}/view?{q}", context=SSL_CONTEXT, timeout=120
                ) as resp, open(local_path, "wb") as out_f:
                    out_f.write(resp.read())
                saved_files.append(local_path)
                print(f"✅ 已保存 ({time.time() - t0:.1f}s): {local_path}")

    if not saved_files:
        print("⚠️ 未在标准输出字段中发现文件，原始 outputs 结构如下：")
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Colab MiniMax-H3 图像与视频 API 客户端")
    parser.add_argument("--url", default=DEFAULT_URL, help="ComfyUI 公网地址（或设置环境变量 H3_URL）")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="本地输出文件夹")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t2i = sub.add_parser("t2i", help="工作流①：资产定妆 / 文生图")
    p_t2i.add_argument("--prompt", required=True, help="生图提示词")
    p_t2i.add_argument("--width", type=int, default=2048, help="宽度（建议 2048/2560，必须是32的倍数）")
    p_t2i.add_argument("--height", type=int, default=1152, help="高度（建议 1152/1440/2048，必须是32的倍数）")
    p_t2i.add_argument("--steps", type=int, default=6, help="采样步数（默认 6）")
    p_t2i.add_argument("--seed", type=int, default=None, help="随机种子")
    p_t2i.add_argument("--prefix", default="h3_t2i", help="输出文件名前缀")

    p_edit = sub.add_parser("edit", help="工作流②：分镜关键帧 / 多图参考改图（支持 <Picture 1>, <Picture 2>...）")
    p_edit.add_argument("--prompt", required=True, help="改图指令（引用原图用 <Picture 1>）")
    p_edit.add_argument("--image", action="append", required=True, help="参考图路径，可多次传入")
    p_edit.add_argument("--width", type=int, default=0, help="宽度（0 表示自动沿用图1原比例）")
    p_edit.add_argument("--height", type=int, default=0, help="高度（0 表示自动沿用图1原比例）")
    p_edit.add_argument("--steps", type=int, default=6, help="采样步数（默认 6）")
    p_edit.add_argument("--seed", type=int, default=None, help="随机种子")
    p_edit.add_argument("--prefix", default="h3_edit", help="输出文件名前缀")

    p_i2v = sub.add_parser("i2v", help="工作流③：镜头视频 / 图生视频（默认开启 4 步 Turbo LoRA）")
    p_i2v.add_argument("--prompt", required=True, help="视频动作、运镜与音效提示词")
    p_i2v.add_argument("--image", default=None, help="首帧参考图路径（可选；不传则为纯文生视频）")
    p_i2v.add_argument("--last-image", default=None, help="尾帧参考图路径（可选，用于首尾帧过渡）")
    p_i2v.add_argument("--duration", type=float, default=5.0, help="视频时长秒数（默认 5.0 秒）")
    p_i2v.add_argument("--megapixels", type=float, default=0.9, help="目标像素量（默认 0.9 MP ≈ 768p）")
    p_i2v.add_argument("--width", type=int, default=0, help="宽度（0 表示有首帧时自动按比例缩放，无首帧默认 1344）")
    p_i2v.add_argument("--height", type=int, default=0, help="高度（0 表示有首帧时自动按比例缩放，无首帧默认 768）")
    p_i2v.add_argument("--steps", type=int, default=4, help="采样步数（默认 4 步配 Turbo LoRA）")
    p_i2v.add_argument("--seed", type=int, default=None, help="随机种子")
    p_i2v.add_argument("--prefix", default="h3_video", help="输出文件名前缀")

    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)

    if args.cmd in ("t2i", "edit"):
        uploaded = []
        if args.cmd == "edit":
            for img_path in args.image:
                print(f"⬆️ 正在上传参考图: {img_path} ...")
                uploaded.append(upload_image(args.url, img_path))
        wf = build_h3_image_workflow(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            seed=seed,
            steps=args.steps,
            uploaded_images=uploaded,
            prefix=args.prefix,
        )
    else:
        first_name = None
        last_name = None
        if args.image:
            print(f"⬆️ 正在上传首帧图: {args.image} ...")
            first_name = upload_image(args.url, args.image)
        if args.last_image:
            print(f"⬆️ 正在上传尾帧图: {args.last_image} ...")
            last_name = upload_image(args.url, args.last_image)
        wf = build_h3_video_workflow(
            prompt=args.prompt,
            duration=args.duration,
            megapixels=args.megapixels,
            width=args.width,
            height=args.height,
            seed=seed,
            steps=args.steps,
            first_frame_name=first_name,
            last_frame_name=last_name,
            prefix=args.prefix,
        )

    run_and_download(args.url, wf, args.out)


if __name__ == "__main__":
    main()
