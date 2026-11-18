#!/usr/bin/env bash
set -euo pipefail

# 自动加载 uv 管理的 google-colab-cli CA 证书，避免 macOS WebSocket SSL 报错
CERT_PATH="/Users/kunlinzhou/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/certifi/cacert.pem"
if [[ -f "$CERT_PATH" ]]; then
  export SSL_CERT_FILE="$CERT_PATH"
elif [[ -x "/Users/kunlinzhou/.local/share/uv/tools/google-colab-cli/bin/python" ]]; then
  FALLBACK_CERT="$(/Users/kunlinzhou/.local/share/uv/tools/google-colab-cli/bin/python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [[ -n "$FALLBACK_CERT" && -f "$FALLBACK_CERT" ]]; then
    export SSL_CERT_FILE="$FALLBACK_CERT"
  fi
fi

COLAB_BIN="${COLAB_BIN:-/Users/kunlinzhou/.local/bin/colab}"
if ! command -v "$COLAB_BIN" >/dev/null 2>&1; then
  COLAB_BIN="$(command -v colab)"
fi

SESSION_NAME="${1:-}"
REQUESTED_ACCEL="${2:-AUTO}"
HIGH_MEM="${3:---high-mem}"

if [[ -z "$SESSION_NAME" ]]; then
  echo "用法: $0 <session_name> [GPU类型: AUTO|H100|G4|A100|L4|T4|CPU] [--high-mem|--no-high-mem]" >&2
  exit 1
fi

echo "[1/3] 检查现有 Colab 会话..."
SESSIONS_OUT="$("$COLAB_BIN" sessions 2>&1 || true)"
if echo "$SESSIONS_OUT" | grep -qw "$SESSION_NAME"; then
  echo "会话 '$SESSION_NAME' 已存在，直接复用。"
else
  echo "[2/3] 正在拉起会话 '$SESSION_NAME' (目标算力: $REQUESTED_ACCEL)..."
  EXTRA_FLAGS=()
  if [[ "$HIGH_MEM" == "--high-mem" ]]; then
    EXTRA_FLAGS+=(--high-mem)
  fi

  REQ_UPPER="$(echo "$REQUESTED_ACCEL" | tr '[:lower:]' '[:upper:]')"
  case "$REQ_UPPER" in
    CPU)
      CANDIDATES=("CPU")
      ;;
    AUTO)
      CANDIDATES=("G4" "A100" "L4" "T4" "CPU")
      ;;
    H100)
      CANDIDATES=("H100" "G4" "A100" "L4" "T4" "CPU")
      ;;
    G4)
      CANDIDATES=("G4" "A100" "L4" "T4" "CPU")
      ;;
    A100)
      CANDIDATES=("A100" "G4" "L4" "T4" "CPU")
      ;;
    L4)
      CANDIDATES=("L4" "T4" "CPU")
      ;;
    *)
      CANDIDATES=("$REQ_UPPER" "L4" "T4" "CPU")
      ;;
  esac

  CREATED=0
  for ACCEL in "${CANDIDATES[@]}"; do
    echo "  -> 尝试申请算力: $ACCEL ${EXTRA_FLAGS[*]:-} ..."
    if [[ "$ACCEL" == "CPU" ]]; then
      if "$COLAB_BIN" new -s "$SESSION_NAME" "${EXTRA_FLAGS[@]}"; then
        CREATED=1
        break
      elif [[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && "$COLAB_BIN" new -s "$SESSION_NAME"; then
        CREATED=1
        break
      fi
    else
      if "$COLAB_BIN" new -s "$SESSION_NAME" --gpu "$ACCEL" "${EXTRA_FLAGS[@]}"; then
        CREATED=1
        break
      elif [[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && "$COLAB_BIN" new -s "$SESSION_NAME" --gpu "$ACCEL"; then
        CREATED=1
        break
      fi
    fi
    echo "  -> $ACCEL 申请失败或无配额，尝试下一个选项..." >&2
  done

  if [[ "$CREATED" -ne 1 ]]; then
    echo "错误：所有算力申请均失败，请检查 'colab sessions' 或 'colab usage'。" >&2
    exit 1
  fi
fi

echo "[3/3] 获取会话状态与网页访问地址..."
"$COLAB_BIN" status -s "$SESSION_NAME" || true
WEB_URL="$("$COLAB_BIN" url -s "$SESSION_NAME")"
echo "=========================================="
echo "SESSION: $SESSION_NAME"
echo "COLAB_WEB_URL: $WEB_URL"
echo "=========================================="
