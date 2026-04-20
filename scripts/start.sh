#!/usr/bin/env bash
#
# One-stop control script for the miniforge native-Windows LLM stack.
#
# Single entry point for every maintenance action: bootstrap the venv, install
# engine binaries, download GGUFs, launch the stack, stop it, show status, or
# run benchmarks.
#
# USAGE:
#   ./scripts/start.sh [ACTION] [FLAGS]
#
# ACTIONS:
#   setup              Create .venv, install deps, install llama.cpp Vulkan (default)
#   install-vulkan     (Re)install llama.cpp Vulkan binaries only
#   install-ik-llama   Install ik_llama.cpp (download or build)
#   pull               Download the REAP GGUF + Qwen3-30B control
#   up                 Launch engine_manager + LiteLLM + Open WebUI
#   down               Stop everything the stack started
#   status             Live status of every service and port
#   bench              Run the benchmark harness across engines
#   obs-up             Launch Prometheus + Grafana (downloads if missing)
#   all                setup -> install-vulkan -> up (skips pull; opt-in)
#
# FLAGS:
#   --force                   Rebuild/redownload even if current version matches
#   --with-airllm             During setup, also install the [airllm] extra
#   --with-ik-llama           During setup, also install the ik_llama.cpp binaries
#   --skip-venv               (setup) Reuse an existing .venv without recreating
#   --tag <tag>               (install-vulkan) Pin a specific llama.cpp release tag
#   --skip-reap               (pull) Skip the 56 GB MiniMax-REAP download
#   --skip-qwen               (pull) Skip the 18 GB Qwen3-30B-A3B download
#   --reap-quant <quant>      (pull) Override REAP quant. Default Q2_K
#   --reap-model-dir <dir>    (pull) Override REAP model directory
#   --model-dir <dir>         (pull) Override model directory
#   --convert-ik              (install-ik-llama) After installing, requantize Q2_K -> IQ2_KS
#   --source-model <path>     (install-ik-llama --convert-ik) Path to source GGUF for requantization
#   --target-quant <quant>    (install-ik-llama) Target quantization. Default iq2_ks
#   --no-ui                   (up) Don't launch Open WebUI
#   --no-litellm              (up) Don't launch LiteLLM; clients hit engine_manager directly
#   --no-obs                  (up) Don't launch Prometheus/Grafana
#   --watch <seconds>         (status) Refresh every N seconds
#   --bench-aliases <list>    (bench) Model aliases to benchmark, comma-separated
#   --bench-target <url>      (bench) OpenAI-compatible base URL
#   --bench-api-key <key>     (bench) API key for benchmarks
#   --bench-max-prompts <n>   (bench) Use only first N prompts. 0 = all
#
# EXAMPLES:
#   ./scripts/start.sh                        # default: setup only
#   ./scripts/start.sh all                    # setup + install Vulkan + launch stack
#   ./scripts/start.sh pull
#   ./scripts/start.sh up
#   ./scripts/start.sh status --watch 5
#   ./scripts/start.sh bench --bench-aliases qwen3-30b-vulkan --bench-max-prompts 3
#   ./scripts/start.sh down

set -euo pipefail

# ===========================================================================
# globals / defaults
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
ENGINES_DIR="$REPO_ROOT/engines"
VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

# Default action
ACTION="setup"

# Shared flags
FORCE=0

# Setup flags
WITH_AIRLLM=0
WITH_IK_LLAMA=0
SKIP_VENV=0

# install-vulkan flags
TAG=""

# pull flags
SKIP_REAP=0
SKIP_QWEN=0
REAP_QUANT="Q2_K"
REAP_MODEL_DIR=""
MODEL_DIR=""

# install-ik-llama flags
CONVERT_IK=0
SOURCE_MODEL=""
TARGET_QUANT="iq2_ks"

# up flags
NO_UI=0
NO_LITELLM=0
NO_OBS=0

# status flags
WATCH=0

# bench flags
BENCH_ALIASES=("minimax-reap-vulkan" "minimax-reap-ikllama" "minimax-reap-miniforge" "minimax-reap-airllm")
BENCH_TARGET="http://127.0.0.1:4000"
BENCH_API_KEY="sk-local-dev-key"
BENCH_MAX_PROMPTS=0

# ===========================================================================
# helpers
# ===========================================================================

log_section() {
    echo ""
    echo "==> $1"
}

log_warn() {
    echo "WARNING: $1" >&2
}

ensure_dir() {
    [[ -d "$1" ]] || mkdir -p "$1"
}

load_dotenv() {
    local env_file="$REPO_ROOT/.env"
    [[ -f "$env_file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^\s*# ]] && continue
        [[ "$line" =~ ^\s*$ ]] && continue
        if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            export "$key=$value"
        fi
    done < "$env_file"
}

activate_venv() {
    if [[ -f "$VENV_ACTIVATE" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_ACTIVATE"
    fi
}

get_python_exe() {
    if [[ -x "$VENV_PY" ]]; then
        echo "$VENV_PY"
    else
        echo "python3"
    fi
}

get_hf_command() {
    if command -v hf &>/dev/null; then
        echo "hf"
    elif command -v huggingface-cli &>/dev/null; then
        echo "huggingface-cli"
    else
        echo "ERROR: Neither 'hf' nor 'huggingface-cli' available. Run './scripts/start.sh setup' first." >&2
        exit 1
    fi
}

github_headers() {
    local headers=(-H "User-Agent: miniforge-installer")
    [[ -n "${GITHUB_TOKEN:-}" ]] && headers+=(-H "Authorization: Bearer $GITHUB_TOKEN")
    printf '%s\n' "${headers[@]}"
}

# ===========================================================================
# argument parsing
# ===========================================================================

parse_args() {
    # First argument may be an action
    if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
        ACTION="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force)
                FORCE=1
                shift
                ;;
            --with-airllm)
                WITH_AIRLLM=1
                shift
                ;;
            --with-ik-llama)
                WITH_IK_LLAMA=1
                shift
                ;;
            --skip-venv)
                SKIP_VENV=1
                shift
                ;;
            --tag)
                TAG="$2"
                shift 2
                ;;
            --skip-reap)
                SKIP_REAP=1
                shift
                ;;
            --skip-qwen)
                SKIP_QWEN=1
                shift
                ;;
            --reap-quant)
                REAP_QUANT="$2"
                shift 2
                ;;
            --reap-model-dir)
                REAP_MODEL_DIR="$2"
                shift 2
                ;;
            --model-dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            --convert-ik)
                CONVERT_IK=1
                shift
                ;;
            --source-model)
                SOURCE_MODEL="$2"
                shift 2
                ;;
            --target-quant)
                TARGET_QUANT="$2"
                shift 2
                ;;
            --no-ui)
                NO_UI=1
                shift
                ;;
            --no-litellm)
                NO_LITELLM=1
                shift
                ;;
            --no-obs)
                NO_OBS=1
                shift
                ;;
            --watch)
                WATCH="$2"
                shift 2
                ;;
            --bench-aliases)
                IFS=',' read -ra BENCH_ALIASES <<< "$2"
                shift 2
                ;;
            --bench-target)
                BENCH_TARGET="$2"
                shift 2
                ;;
            --bench-api-key)
                BENCH_API_KEY="$2"
                shift 2
                ;;
            --bench-max-prompts)
                BENCH_MAX_PROMPTS="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done

    # Validate action
    case "$ACTION" in
        setup|install-vulkan|install-ik-llama|pull|up|down|status|bench|obs-up|all)
            ;;
        *)
            echo "Unknown action: $ACTION" >&2
            exit 1
            ;;
    esac
}

show_help() {
    head -n 66 "$0" | tail -n 64
}

# ===========================================================================
# action: install-vulkan
# ===========================================================================

install_vulkan() {
    log_section "Installing llama.cpp Vulkan binaries"
    local engine_dir="$ENGINES_DIR/llama-cpp-vulkan"
    local version_file="$engine_dir/VERSION"

    ensure_dir "$engine_dir"

    local release_json
    if [[ -n "$TAG" ]]; then
        echo "Fetching release $TAG..."
        release_json=$(curl -sL "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$TAG" \
            $(github_headers))
    else
        echo "Fetching latest llama.cpp release..."
        release_json=$(curl -sL "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" \
            $(github_headers))
    fi

    local tag_name
    tag_name=$(echo "$release_json" | grep -o '"tag_name": "[^"]*"' | cut -d'"' -f4)
    echo "Release: $tag_name"

    # Check version if not forced
    if [[ $FORCE -eq 0 && -f "$version_file" ]]; then
        local current
        current=$(<"$version_file")
        if [[ "$current" == "$tag_name" ]]; then
            echo "Already at $tag_name. Use --force to reinstall."
            return 0
        fi
    fi

    # Find Linux asset (ubuntu-x64 or linux-x64)
    local asset_url
    asset_url=$(echo "$release_json" | grep -o '"browser_download_url": "[^"]*bin-ubuntu-x64\.zip"' | cut -d'"' -f4)
    if [[ -z "$asset_url" ]]; then
        asset_url=$(echo "$release_json" | grep -o '"browser_download_url": "[^"]*bin-linux-x64\.zip"' | cut -d'"' -f4)
    fi
    if [[ -z "$asset_url" ]]; then
        echo "ERROR: No Linux x64 asset in release $tag_name" >&2
        exit 1
    fi

    local asset_name
    asset_name=$(basename "$asset_url")
    echo "Downloading $asset_name..."

    local tmp_zip="/tmp/$asset_name"
    curl -sL "$asset_url" -o "$tmp_zip"

    # Clean and extract
    rm -rf "$engine_dir"/*
    ensure_dir "$engine_dir"

    echo "Extracting to $engine_dir..."
    unzip -q "$tmp_zip" -d "$engine_dir"
    rm -f "$tmp_zip"

    # Handle nested directory structure
    if [[ -d "$engine_dir/build/bin" ]]; then
        mv "$engine_dir/build/bin"/* "$engine_dir/" 2>/dev/null || true
        rm -rf "$engine_dir/build"
    fi

    echo "$tag_name" > "$version_file"

    local binary="$engine_dir/llama-server"
    if [[ ! -x "$binary" ]]; then
        echo "ERROR: llama-server not found at $binary" >&2
        exit 1
    fi

    chmod +x "$engine_dir"/*

    echo "Installed llama.cpp Vulkan $tag_name"
    echo "  Verify: $engine_dir/llama-server --version"
    echo "  List devices: $engine_dir/llama-server --list-devices"
}

# ===========================================================================
# action: install-ik-llama
# ===========================================================================

install_ik_llama() {
    log_section "Installing ik_llama.cpp"
    local engine_dir="$ENGINES_DIR/ik-llama"
    local version_file="$engine_dir/VERSION"
    local build_dir="/tmp/ik_llama_build"

    ensure_dir "$engine_dir"

    if [[ $FORCE -eq 0 && -x "$engine_dir/llama-server" ]]; then
        echo "ik_llama already installed. Use --force to rebuild."
    else
        local prebuilt_url=""
        local release_json
        release_json=$(curl -sL "https://api.github.com/repos/ikawrakow/ik_llama.cpp/releases" \
            $(github_headers))

        # Look for linux asset
        prebuilt_url=$(echo "$release_json" | grep -o '"browser_download_url": "[^"]*linux[^"]*\.zip"' | head -1 | cut -d'"' -f4)

        if [[ -n "$prebuilt_url" ]]; then
            local asset_name
            asset_name=$(basename "$prebuilt_url")
            echo "Downloading $asset_name..."

            local tmp_zip="/tmp/$asset_name"
            curl -sL "$prebuilt_url" -o "$tmp_zip"

            rm -rf "$engine_dir"/*
            ensure_dir "$engine_dir"
            unzip -q "$tmp_zip" -d "$engine_dir"
            rm -f "$tmp_zip"

            # Handle nested directories
            for nested in "$engine_dir/build/bin/Release" "$engine_dir/build/bin" "$engine_dir/bin"; do
                if [[ -d "$nested" ]]; then
                    mv "$nested"/* "$engine_dir/" 2>/dev/null || true
                    rm -rf "${nested%%/*}" 2>/dev/null || true
                    break
                fi
            done

            local tag_name
            tag_name=$(echo "$release_json" | grep -o '"tag_name": "[^"]*"' | head -1 | cut -d'"' -f4)
            echo "$tag_name" > "$version_file"
        else
            echo "No prebuilt Linux binary found. Building from source."

            if ! command -v git &>/dev/null; then
                echo "ERROR: git not found. Install git first." >&2
                exit 1
            fi
            if ! command -v cmake &>/dev/null; then
                echo "ERROR: cmake not found. Install cmake first." >&2
                exit 1
            fi
            if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
                echo "ERROR: C++ compiler not found. Install build-essential or clang." >&2
                exit 1
            fi

            rm -rf "$build_dir"
            echo "Cloning ik_llama.cpp..."
            git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp "$build_dir"

            pushd "$build_dir" &>/dev/null
            local commit
            commit=$(git rev-parse HEAD | cut -c1-7)
            echo "Configuring (Release, AVX2 CPU)..."
            cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
            echo "Building (10-20 min)..."
            cmake --build build --config Release --parallel

            local bin_src="$build_dir/build/bin/Release"
            [[ -d "$bin_src" ]] || bin_src="$build_dir/build/bin"

            if [[ ! -d "$bin_src" ]]; then
                echo "ERROR: Build output missing under $build_dir/build/bin" >&2
                exit 1
            fi

            rm -rf "$engine_dir"/*
            ensure_dir "$engine_dir"
            cp "$bin_src"/* "$engine_dir/"
            echo "git-$commit" > "$version_file"
            popd &>/dev/null
            rm -rf "$build_dir"
        fi

        chmod +x "$engine_dir"/*
        echo "Installed ik_llama.cpp"
    fi

    # Handle requantization
    if [[ $CONVERT_IK -eq 1 ]]; then
        if [[ -z "$SOURCE_MODEL" ]]; then
            echo "ERROR: --source-model required with --convert-ik" >&2
            exit 1
        fi
        if [[ ! -f "$SOURCE_MODEL" ]]; then
            echo "ERROR: Source GGUF not found: $SOURCE_MODEL" >&2
            exit 1
        fi

        local qbin="$engine_dir/llama-quantize"
        if [[ ! -x "$qbin" ]]; then
            echo "ERROR: llama-quantize missing; required for --convert-ik" >&2
            exit 1
        fi

        local src_dir
        src_dir=$(dirname "$SOURCE_MODEL")
        local src_name
        src_name=$(basename "$SOURCE_MODEL" .gguf)
        local dst_name="${src_name//Q2_K/${TARGET_QUANT^^}}"
        local dst_path="$src_dir/$dst_name.gguf"

        echo "Requantizing $SOURCE_MODEL -> $dst_path ($TARGET_QUANT) (20-60 min)..."
        "$qbin" "$SOURCE_MODEL" "$dst_path" "$TARGET_QUANT"
        echo "Wrote $dst_path"
    fi
}

# ===========================================================================
# action: setup
# ===========================================================================

setup() {
    log_section "miniforge bootstrap"

    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 not on PATH. Install Python 3.11+ and restart your shell." >&2
        exit 1
    fi

    local py_ver
    py_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$py_ver" | cut -d. -f1) -lt 3 ]] || \
       ([[ $(echo "$py_ver" | cut -d. -f1) -eq 3 ]] && [[ $(echo "$py_ver" | cut -d. -f2) -lt 11 ]]); then
        echo "ERROR: Python $py_ver is too old. Need 3.11+." >&2
        exit 1
    fi
    echo "python $py_ver OK"

    if ! command -v uv &>/dev/null; then
        echo "Installing uv via pip..."
        python3 -m pip install --user --upgrade uv
    fi
    echo "uv OK"

    if [[ $SKIP_VENV -eq 0 ]]; then
        if [[ ! -d ".venv" ]]; then
            echo "Creating .venv..."
            uv venv --python "$py_ver"
        else
            echo ".venv exists, skipping (delete it to recreate)"
        fi
    fi

    echo "Installing miniforge[all]..."
    uv pip install -e ".[all]" --index-strategy unsafe-best-match

    if [[ $WITH_AIRLLM -eq 1 ]]; then
        echo "Installing [airllm]..."
        uv pip install -e ".[airllm]" --index-strategy unsafe-best-match || \
            log_warn "[airllm] install failed; AirLLM engine will be unavailable."
    fi

    install_vulkan

    echo "Installing Open WebUI as isolated uv tool..."
    uv tool install open-webui || log_warn "uv tool install open-webui failed; up will skip the UI."

    echo "Installing LiteLLM proxy as isolated uv tool..."
    uv tool install 'litellm[proxy]' || log_warn "uv tool install litellm failed; up will skip LiteLLM."

    if [[ $WITH_IK_LLAMA -eq 1 ]]; then
        install_ik_llama
    fi

    for d in bench/results data/openwebui engines models logs; do
        ensure_dir "$REPO_ROOT/$d"
    done

    local env_example="$REPO_ROOT/.env.example"
    local env_file="$REPO_ROOT/.env"
    if [[ -f "$env_example" && ! -f "$env_file" ]]; then
        cp "$env_example" "$env_file"
        echo ""
        echo "Created .env from .env.example. Edit HF_TOKEN / MODEL_DIR / MODEL_DIR_REAP."
    fi

    log_section "Setup complete"
    echo "Next:"
    echo "  ./scripts/start.sh pull        # download REAP GGUF (~56 GB)"
    echo "  ./scripts/start.sh up          # launch the stack"
    echo "  ./scripts/start.sh all         # setup + up together"
}

# ===========================================================================
# action: pull
# ===========================================================================

pull() {
    log_section "Downloading model GGUFs"
    activate_venv

    if [[ -z "$REAP_MODEL_DIR" ]]; then
        REAP_MODEL_DIR="${MODEL_DIR_REAP:-$HOME/miniforge/models}"
    fi
    if [[ -z "$MODEL_DIR" ]]; then
        MODEL_DIR="${MODEL_DIR:-$HOME/miniforge/gguf}"
    fi

    ensure_dir "$REAP_MODEL_DIR"
    ensure_dir "$MODEL_DIR"

    local hf
    hf=$(get_hf_command)

    pull_repo() {
        local repo="$1"
        local pattern="$2"
        local local_dir="$3"
        echo ""
        echo "==> $repo"
        echo "    filter : $pattern"
        echo "    target : $local_dir"
        local args=(download "$repo" --include "$pattern" --local-dir "$local_dir")
        [[ -n "${HF_TOKEN:-}" ]] && args+=(--token "$HF_TOKEN")
        "$hf" "${args[@]}"
    }

    if [[ $SKIP_REAP -eq 0 ]]; then
        local target="$REAP_MODEL_DIR/"
        echo "Pulling MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10 (~99 GB)."
        pull_repo 'saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10' "*.safetensors" "$target"
    else
        echo "Skipping REAP (--skip-reap)."
    fi

    if [[ $SKIP_QWEN -eq 0 ]]; then
        local target="$MODEL_DIR/Qwen3-30B-A3B-Instruct-GGUF"
        echo "Pulling Qwen3-30B-A3B UD-Q4_K_M (~18 GB)."
        pull_repo 'unsloth/Qwen3-30B-A3B-Instruct-GGUF' '*UD-Q4_K_M*.gguf' "$target"
    else
        echo "Skipping Qwen3-30B (--skip-qwen)."
    fi

    echo ""
    echo "Done."
    echo "Env hints for .env:"
    echo "  MODEL_DIR=$MODEL_DIR"
    echo "  MODEL_DIR_REAP=$REAP_MODEL_DIR"
}

# ===========================================================================
# action: up
# ===========================================================================

start_service_background() {
    local name="$1"
    local file_path="$2"
    shift 2
    local arguments=("$@")

    local log_path="$LOG_DIR/$name.log"
    local pid_path="$LOG_DIR/$name.pid"

    ensure_dir "$LOG_DIR"

    # Check if already running
    if [[ -f "$pid_path" ]]; then
        local existing_pid
        existing_pid=$(<"$pid_path")
        if kill -0 "$existing_pid" 2>/dev/null; then
            log_warn "$name already running (pid $existing_pid). Run 'down' first."
            return 0
        fi
    fi

    echo "Starting $name..."
    nohup "$file_path" "${arguments[@]}" > "$log_path" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_path"
    echo "  $name pid=$pid, log=$log_path"
}

up() {
    log_section "Launching miniforge stack"
    activate_venv
    load_dotenv
    ensure_dir "$LOG_DIR"

    # Set UTF-8 for Python tools
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8

    local python
    python=$(get_python_exe)

    start_service_background 'engine_manager' "$python" \
        -u -m miniforge.launcher.engine_manager \
        --config config/engines.yaml

    if [[ $NO_LITELLM -eq 0 ]]; then
        local master="${LITELLM_MASTER_KEY:-sk-local-dev-key}"
        export LITELLM_MASTER_KEY="$master"
        local uv_exe
        uv_exe=$(command -v uv)
        start_service_background 'litellm' "$uv_exe" \
            tool run --from 'litellm[proxy]' litellm \
            --config config/litellm.yaml \
            --host 127.0.0.1 --port 4000
    fi

    if [[ $NO_UI -eq 0 ]]; then
        local owui_env="$REPO_ROOT/config/openwebui.env"
        if [[ -f "$owui_env" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ "$line" =~ ^\s*# ]] && continue
                [[ "$line" =~ ^\s*$ ]] && continue
                if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$ ]]; then
                    local key="${BASH_REMATCH[1]}"
                    local value="${BASH_REMATCH[2]}"
                    export "$key=$value"
                fi
            done < "$owui_env"
        fi

        if [[ $NO_LITELLM -eq 0 ]]; then
            local ow_key="${LITELLM_MASTER_KEY:-sk-local-dev-key}"
            export OPENAI_API_KEY="$ow_key"
        fi

        local uv_exe
        uv_exe=$(command -v uv)
        start_service_background 'openwebui' "$uv_exe" \
            tool run open-webui \
            serve --host 127.0.0.1 --port 8080
    fi

    if [[ $NO_OBS -eq 0 ]]; then
        start_obs_services
    fi

    echo ""
    echo "Stack starting. URLs:"
    echo "  Open WebUI      http://127.0.0.1:8080"
    echo "  LiteLLM proxy   http://127.0.0.1:4000"
    echo "  engine_manager  http://127.0.0.1:9292"
    if [[ $NO_OBS -eq 0 ]]; then
        echo "  Prometheus      http://127.0.0.1:9090"
        echo "  Grafana         http://127.0.0.1:3000  (admin/admin)"
    fi
    echo ""
    echo "Tail logs:   tail -f ./logs/engine_manager.log"
    echo "Status:      ./scripts/start.sh status"
    echo "Stop:        ./scripts/start.sh down"
}

# ===========================================================================
# action: down
# ===========================================================================

stop_by_pidfile() {
    local name="$1"
    local pid_file="$LOG_DIR/$name.pid"

    [[ -f "$pid_file" ]] || return 0

    local proc_id
    proc_id=$(<"$pid_file")
    if [[ -n "$proc_id" ]] && kill -0 "$proc_id" 2>/dev/null; then
        echo "Stopping $name (pid $proc_id)..."
        kill -TERM "$proc_id" 2>/dev/null || true
        sleep 1
        kill -KILL "$proc_id" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

down() {
    log_section "Stopping miniforge stack"
    ensure_dir "$LOG_DIR"

    stop_by_pidfile 'grafana'
    stop_by_pidfile 'prometheus'
    stop_by_pidfile 'openwebui'
    stop_by_pidfile 'litellm'
    stop_by_pidfile 'engine_manager'

    # Kill any remaining processes by name
    for name in llama-server open-webui open_webui litellm prometheus grafana-server; do
        pkill -9 -f "$name" 2>/dev/null || true
    done

    # Kill any processes on our ports
    for port in 8001 8002 8003 8004 4000 8080 9292 9090 3000; do
        local pids
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -TERM 2>/dev/null || true
            sleep 1
            echo "$pids" | xargs kill -KILL 2>/dev/null || true
        fi
    done

    echo "Stopped."
}

# ===========================================================================
# action: status
# ===========================================================================

status() {
    local services=(
        "engine_manager:9292:/health"
        "litellm:4000:/health/readiness"
        "openwebui:8080:/health"
        "prometheus:9090:/-/ready"
        "grafana:3000:/api/health"
        "miniforge-server:8003:/health"
        "miniforge-airllm:8004:/health"
        "llama_vulkan:8001:/health"
        "ik_llama:8002:/health"
    )

    show_once() {
        local now
        now=$(date '+%H:%M:%S')
        echo ""
        echo "miniforge status @ $now"
        printf '%-22s %-10s %-10s %s\n' 'service' 'port' 'listening' 'details'
        printf '%.0s-' {1..80}; echo

        for svc in "${services[@]}"; do
            IFS=':' read -r name port health <<< "$svc"
            local is_up=0
            local detail=""

            if lsof -ti :"$port" &>/dev/null; then
                is_up=1
                # Try health check
                local resp
                if resp=$(curl -s "http://127.0.0.1:$port$health" --max-time 2 2>/dev/null); then
                    detail="${resp:0:57}"
                    [[ ${#resp} -gt 57 ]] && detail="${detail}..."
                else
                    detail="(no /health)"
                fi
            fi

            local up_text="no"
            if [[ $is_up -eq 1 ]]; then
                up_text="yes"
                printf '%-22s %-10s %-10s %s\n' "$name" "$port" "$up_text" "$detail"
            else
                printf '%-22s %-10s %-10s %s\n' "$name" "$port" "$up_text" "$detail"
            fi
        done

        # Try to get active engine
        local mx
        if mx=$(curl -s 'http://127.0.0.1:9292/engine' --max-time 2 2>/dev/null); then
            local alias
            alias=$(echo "$mx" | grep -o '"alias":"[^"]*"' | cut -d'"' -f4)
            [[ -n "$alias" ]] && echo "" && echo "active engine alias: $alias"
        fi

        # Try to get TPS
        local metrics
        if metrics=$(curl -s 'http://127.0.0.1:9292/metrics' --max-time 2 2>/dev/null); then
            local tps_line
            tps_line=$(echo "$metrics" | grep '^miniforge_tps ' | head -1)
            if [[ -n "$tps_line" ]]; then
                local tps
                tps=$(echo "$tps_line" | awk '{print $2}')
                echo "last generation TPS: $tps"
            fi
        fi
    }

    if [[ $WATCH -le 0 ]]; then
        show_once
    else
        while true; do
            clear
            show_once
            sleep "$WATCH"
        done
    fi
}

# ===========================================================================
# action: bench
# ===========================================================================

bench() {
    log_section "Running benchmark sweep"
    activate_venv

    local results_dir="$REPO_ROOT/bench/results"
    local prompts_file="$REPO_ROOT/bench/prompts.jsonl"

    if [[ ! -f "$prompts_file" ]]; then
        echo "ERROR: Missing $prompts_file" >&2
        exit 1
    fi

    ensure_dir "$results_dir"

    local python
    python=$(get_python_exe)
    local stamp
    stamp=$(date '+%Y%m%d-%H%M%S')

    for alias in "${BENCH_ALIASES[@]}"; do
        local out="$results_dir/${alias}-${stamp}.json"
        echo ""
        echo "==> $alias"
        local py_args=(
            "$REPO_ROOT/bench/runner.py"
            --alias "$alias"
            --target "$BENCH_TARGET"
            --api-key "$BENCH_API_KEY"
            --prompts "$prompts_file"
            --output "$out"
        )
        [[ $BENCH_MAX_PROMPTS -gt 0 ]] && py_args+=(--max-prompts "$BENCH_MAX_PROMPTS")

        "$python" "${py_args[@]}" || log_warn "$alias bench failed. Continuing."
    done

    echo ""
    echo "==> Compiling SUMMARY.md"
    "$python" "$REPO_ROOT/bench/compare.py" "$results_dir" --out "$results_dir/SUMMARY.md"
}

# ===========================================================================
# action: obs-up
# ===========================================================================

get_obs_paths() {
    local obs_dir="$ENGINES_DIR/_obs"
    echo "$obs_dir"
}

ensure_obs_installed() {
    local obs_dir
    obs_dir=$(get_obs_paths)
    local prom_dir="$obs_dir/prometheus"
    local graf_dir="$obs_dir/grafana"

    ensure_dir "$obs_dir"

    if [[ $FORCE -eq 1 ]] || [[ ! -x "$prom_dir/prometheus" ]]; then
        echo "Downloading Prometheus..."
        local release_json
        release_json=$(curl -sL "https://api.github.com/repos/prometheus/prometheus/releases/latest" \
            $(github_headers))
        local asset_url
        asset_url=$(echo "$release_json" | grep -o '"browser_download_url": "[^"]*linux-amd64\.tar\.gz"' | head -1 | cut -d'"' -f4)

        if [[ -z "$asset_url" ]]; then
            echo "ERROR: no linux-amd64 prometheus asset" >&2
            exit 1
        fi

        local tmp_tar="/tmp/prometheus.tar.gz"
        curl -sL "$asset_url" -o "$tmp_tar"
        rm -rf "$prom_dir"
        ensure_dir "$prom_dir"
        tar -xzf "$tmp_tar" -C "$obs_dir" --strip-components=1
        rm -f "$tmp_tar"
    fi

    if [[ $FORCE -eq 1 ]] || [[ ! -x "$graf_dir/bin/grafana-server" ]]; then
        echo "Downloading Grafana OSS..."
        # Get latest Grafana OSS for Linux
        local graf_url="https://dl.grafana.com/oss/release/grafana-11.3.0.linux-amd64.tar.gz"
        local tmp_tar="/tmp/grafana.tar.gz"
        curl -sL "$graf_url" -o "$tmp_tar"
        rm -rf "$graf_dir"
        ensure_dir "$graf_dir"
        tar -xzf "$tmp_tar" -C "$obs_dir" --strip-components=1
        rm -f "$tmp_tar"
    fi

    local prov_dir="$graf_dir/conf/provisioning"
    local dash_prov_dir="$prov_dir/dashboards"
    local ds_prov_dir="$prov_dir/datasources"
    ensure_dir "$dash_prov_dir"
    ensure_dir "$ds_prov_dir"

    cat > "$ds_prov_dir/miniforge.yaml" <<'EOF'
apiVersion: 1
datasources:
  - name: prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    uid: prometheus
    isDefault: true
EOF

    local dashes_root="$graf_dir/data/dashboards/miniforge"
    ensure_dir "$dashes_root"
    local dash_src="$REPO_ROOT/config/grafana/dashboards/llm-engines.json"
    [[ -f "$dash_src" ]] && cp -f "$dash_src" "$dashes_root/"

    cat > "$dash_prov_dir/miniforge.yaml" <<EOF
apiVersion: 1
providers:
  - name: miniforge
    folder: miniforge
    type: file
    options:
      path: $dashes_root
EOF
}

start_obs_services() {
    local obs_dir
    obs_dir=$(get_obs_paths)
    local prom_dir="$obs_dir/prometheus"
    local graf_dir="$obs_dir/grafana"

    if [[ ! -x "$prom_dir/prometheus" ]] || [[ ! -x "$graf_dir/bin/grafana-server" ]]; then
        ensure_obs_installed
    fi

    local prom_cfg="$REPO_ROOT/config/prometheus.yml"
    start_service_background 'prometheus' "$prom_dir/prometheus" \
        --config.file="$prom_cfg" \
        --web.listen-address=127.0.0.1:9090 \
        --storage.tsdb.path="$obs_dir/prometheus-data"

    start_service_background 'grafana' "$graf_dir/bin/grafana-server" \
        --homepath="$graf_dir"
}

obs_up() {
    log_section "Launching Prometheus + Grafana"
    ensure_dir "$LOG_DIR"
    ensure_obs_installed
    start_obs_services
    echo "Prometheus http://127.0.0.1:9090"
    echo "Grafana    http://127.0.0.1:3000 (admin/admin)"
}

# ===========================================================================
# main dispatch
# ===========================================================================

main() {
    parse_args "$@"

    case "$ACTION" in
        setup)
            setup
            ;;
        install-vulkan)
            install_vulkan
            ;;
        install-ik-llama)
            install_ik_llama
            ;;
        pull)
            pull
            ;;
        up)
            up
            ;;
        down)
            down
            ;;
        status)
            status
            ;;
        bench)
            bench
            ;;
        obs-up)
            obs_up
            ;;
        all)
            setup
            up
            ;;
        *)
            echo "Unknown action: $ACTION" >&2
            exit 1
            ;;
    esac
}

main "$@"
