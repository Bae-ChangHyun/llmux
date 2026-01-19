#!/bin/bash

# vLLM Container Management Script
# Usage: ./run.sh <profile> <action>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="$SCRIPT_DIR/profiles"
COMMON_ENV="$SCRIPT_DIR/.env.common"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}vLLM Container Management${NC}"
    echo ""
    echo "Usage: ./run.sh <profile> <action> [options]"
    echo ""
    echo "Actions:"
    echo "  up              - Start container"
    echo "  up --dev        - Start container with dev build (from source)"
    echo "  up --dev --tag TAG  - Start with specific dev image version"
    echo "  down            - Stop container"
    echo "  logs            - Show container logs (follow mode)"
    echo "  status          - Show container status"
    echo ""
    echo "Build commands:"
    echo "  ./run.sh build [branch]            # Build with auto date tag (branch-YYYYMMDD)"
    echo "  ./run.sh build [branch] --tag TAG  # Build with custom tag"
    echo "  ./run.sh build --official          # Build for all GPU architectures (slow)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh vlm up                        # Start VLM (official image)"
    echo "  ./run.sh vlm up --dev                  # Start VLM (latest dev build)"
    echo "  ./run.sh vlm up --dev --tag main-20260116  # Start with specific version"
    echo "  ./run.sh build                         # Build from main (tag: main-20260116)"
    echo "  ./run.sh build fix-lora --tag v1.0     # Build with custom tag"
    echo "  ./run.sh llm down                      # Stop LLM container"
    echo "  ./run.sh clova logs                    # Show CLOVA logs"
    echo ""
    echo "Other commands:"
    echo "  ./run.sh list        # List available profiles"
    echo "  ./run.sh images      # List built vllm-dev images"
    echo "  ./run.sh ps          # Show all running vLLM containers"
    echo "  ./run.sh gpu         # Show GPU usage"
}

list_profiles() {
    echo -e "${BLUE}Available Profiles:${NC}"
    echo ""
    printf "%-12s %-6s %-6s %-8s %s\n" "PROFILE" "GPU" "PORT" "LORA" "MODEL"
    echo "--------------------------------------------------------------"

    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            # Use filename (without .env) as profile name
            profile_name=$(basename "$profile" .env)

            # Read values from profile
            gpu_id=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
            port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
            config=$(grep "^CONFIG_NAME=" "$profile" | cut -d'=' -f2)
            enable_lora=$(grep "^ENABLE_LORA=" "$profile" | cut -d'=' -f2)

            # Get actual model name from config file
            config_file="$SCRIPT_DIR/config/${config}.yaml"
            if [[ -f "$config_file" ]]; then
                model_name=$(grep "^model:" "$config_file" | cut -d':' -f2- | sed 's/^ *//')
            else
                model_name="$config"
            fi

            if [[ "$enable_lora" == "true" ]]; then
                lora_status="ON"
            else
                lora_status="-"
            fi

            printf "%-15s %-6s %-6s %-8s %s\n" "$profile_name" "$gpu_id" "$port" "$lora_status" "$model_name"
        fi
    done
    echo ""
}

find_profile() {
    local name=$1
    local profile_path="$PROFILES_DIR/$name.env"

    # Check if profile file exists
    if [[ -f "$profile_path" ]]; then
        echo "$profile_path"
        return 0
    fi

    return 1
}

check_conflict() {
    local profile=$1
    local gpu_id=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
    local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)

    # Check for port conflict
    local port_conflict=$(docker ps --format '{{.Names}}:{{.Ports}}' 2>/dev/null | grep ":$port->")
    if [[ -n "$port_conflict" ]]; then
        local conflict_container=$(echo "$port_conflict" | cut -d':' -f1)
        echo -e "${RED}Error: Port $port is already in use by container '$conflict_container'${NC}"
        return 1
    fi

    return 0
}

build_lora_options() {
    local profile_path=$1
    local lora_options=""

    local enable_lora=$(grep "^ENABLE_LORA=" "$profile_path" | cut -d'=' -f2)

    if [[ "$enable_lora" == "true" ]]; then
        lora_options="--enable-lora"

        local max_loras=$(grep "^MAX_LORAS=" "$profile_path" | cut -d'=' -f2)
        local max_lora_rank=$(grep "^MAX_LORA_RANK=" "$profile_path" | cut -d'=' -f2)
        local lora_modules=$(grep "^LORA_MODULES=" "$profile_path" | cut -d'=' -f2-)

        if [[ -n "$max_loras" ]]; then
            lora_options="$lora_options --max-loras $max_loras"
        fi

        if [[ -n "$max_lora_rank" ]]; then
            lora_options="$lora_options --max-lora-rank $max_lora_rank"
        fi

        if [[ -n "$lora_modules" ]]; then
            # Replace comma with space to pass all modules to single --lora-modules option
            local modules_formatted="${lora_modules//,/ }"
            lora_options="$lora_options --lora-modules $modules_formatted"
        fi
    fi

    echo "$lora_options"
}

# Detect GPU compute capability
detect_gpu_arch() {
    local arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$arch" ]]; then
        echo ""
        return 1
    fi
    echo "$arch"
}

run_build() {
    local branch="main"
    local use_official=false
    local custom_tag=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --official)
                use_official=true
                shift
                ;;
            --tag)
                custom_tag="$2"
                shift 2
                ;;
            *)
                branch="$1"
                shift
                ;;
        esac
    done

    if [[ "$use_official" == "true" ]]; then
        run_build_official "$branch" "$custom_tag"
    else
        run_build_fast "$branch" "$custom_tag"
    fi
}

# Clone or update vLLM repository
clone_or_update_vllm() {
    local branch=$1
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    if [[ -d "$vllm_src_dir/.git" ]]; then
        echo -e "${BLUE}Updating existing vLLM source...${NC}"
        cd "$vllm_src_dir"
        git fetch origin
        git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"
        git pull origin "$branch" 2>/dev/null || true
    else
        echo -e "${BLUE}Cloning vLLM repository...${NC}"
        rm -rf "$vllm_src_dir"
        git clone https://github.com/vllm-project/vllm.git "$vllm_src_dir"
        cd "$vllm_src_dir"
        git checkout "$branch"
    fi
}

# Fast local build - uses official Dockerfile with YOUR GPU only
run_build_fast() {
    local branch=${1:-main}
    local custom_tag=${2:-}
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    # Auto-detect GPU architecture
    local gpu_arch=$(detect_gpu_arch)
    if [[ -z "$gpu_arch" ]]; then
        echo -e "${RED}Error: Could not detect GPU. Make sure nvidia-smi works.${NC}"
        echo -e "${YELLOW}Tip: Use './run.sh build --official' for official build${NC}"
        return 1
    fi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

    # Generate tags
    local date_tag="${branch}-$(date +%Y%m%d)"
    local main_tag="${custom_tag:-$date_tag}"

    echo -e "${BLUE}Building vLLM from source (branch: $branch)${NC}"
    echo -e "${GREEN}Detected GPU: $gpu_name (sm_$gpu_arch)${NC}"
    echo -e "${GREEN}Building for your GPU only - MUCH faster!${NC}"
    echo -e "${YELLOW}Tag: vllm-dev:$main_tag${NC}"
    echo ""

    clone_or_update_vllm "$branch"

    echo ""
    echo -e "${BLUE}Building with official Dockerfile (GPU: $gpu_arch only)...${NC}"
    echo ""

    # Build using official Dockerfile with local GPU arch only
    docker build \
        -f docker/Dockerfile \
        --build-arg torch_cuda_arch_list="$gpu_arch" \
        --build-arg RUN_WHEEL_CHECK=false \
        --target vllm-openai \
        -t "vllm-dev:$main_tag" \
        -t "vllm-dev:$branch" \
        .

    local result=$?

    cd "$SCRIPT_DIR"

    if [[ $result -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Build completed successfully!${NC}"
        echo "Images tagged as:"
        echo "  - vllm-dev:$main_tag"
        echo "  - vllm-dev:$branch (latest for this branch)"
        echo ""
        echo "Usage:"
        echo "  ./run.sh <profile> up --dev              # Use latest ($branch)"
        echo "  ./run.sh <profile> up --dev --tag $main_tag  # Use specific version"
    else
        echo -e "${RED}Build failed!${NC}"
    fi

    return $result
}

# Official build - builds for ALL GPU architectures (slow)
run_build_official() {
    local branch=${1:-main}
    local custom_tag=${2:-}
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    # Generate tags
    local date_tag="${branch}-$(date +%Y%m%d)"
    local main_tag="${custom_tag:-$date_tag}"

    echo -e "${BLUE}Building vLLM from source (branch: $branch)...${NC}"
    echo -e "${YELLOW}Using official vLLM Dockerfile (ALL architectures)${NC}"
    echo -e "${YELLOW}This may take several HOURS on first build.${NC}"
    echo -e "${YELLOW}Tag: vllm-dev:$main_tag${NC}"
    echo ""

    clone_or_update_vllm "$branch"

    echo ""
    echo -e "${BLUE}Building Docker image using official Dockerfile...${NC}"
    echo ""

    # Build using official Dockerfile (all architectures)
    # Skip wheel size check for official builds (builds for all GPUs are large)
    docker build \
        -f docker/Dockerfile \
        --build-arg RUN_WHEEL_CHECK=false \
        --target vllm-openai \
        -t "vllm-dev:$main_tag" \
        -t "vllm-dev:$branch" \
        .

    local result=$?

    cd "$SCRIPT_DIR"

    if [[ $result -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Build completed successfully!${NC}"
        echo "Images tagged as:"
        echo "  - vllm-dev:$main_tag"
        echo "  - vllm-dev:$branch (latest for this branch)"
        echo ""
        echo "Usage:"
        echo "  ./run.sh <profile> up --dev              # Use latest ($branch)"
        echo "  ./run.sh <profile> up --dev --tag $main_tag  # Use specific version"
    else
        echo -e "${RED}Build failed!${NC}"
    fi

    return $result
}

run_up() {
    local profile_path=$1
    local profile_name=$2
    local use_dev=$3
    local custom_tag=$4

    # Check for conflicts
    if ! check_conflict "$profile_path"; then
        return 1
    fi

    # Build LoRA options and export as environment variable
    export LORA_OPTIONS=$(build_lora_options "$profile_path")

    cd "$SCRIPT_DIR"

    if [[ "$use_dev" == "true" ]]; then
        echo -e "${GREEN}Starting $profile_name (dev build)...${NC}"

        # Check if dev image exists
        local branch=$(grep "^VLLM_BRANCH=" "$COMMON_ENV" 2>/dev/null | cut -d'=' -f2)
        branch=${branch:-main}

        local image_tag="${custom_tag:-$branch}"

        if ! docker image inspect "vllm-dev:$image_tag" &>/dev/null; then
            if [[ -n "$custom_tag" ]]; then
                echo -e "${RED}Error: Image vllm-dev:$image_tag not found${NC}"
                echo -e "${YELLOW}Available images:${NC}"
                docker images vllm-dev --format "  {{.Tag}}"
                return 1
            else
                echo -e "${YELLOW}Dev image not found. Building first...${NC}"
                run_build "$branch"
                if [[ $? -ne 0 ]]; then
                    return 1
                fi
            fi
        fi

        export VLLM_DEV_TAG="$image_tag"
        docker compose -f docker-compose.dev.yaml --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d
    else
        echo -e "${GREEN}Starting $profile_name...${NC}"
        docker compose --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d
    fi

    local result=$?

    if [[ $result -eq 0 ]]; then
        echo -e "${GREEN}$profile_name started successfully!${NC}"
        echo ""
        echo "Container: $(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)"
        echo "Port: $(grep "^VLLM_PORT=" "$profile_path" | cut -d'=' -f2)"
        echo "GPU: $(grep "^GPU_ID=" "$profile_path" | cut -d'=' -f2)"

        local enable_lora=$(grep "^ENABLE_LORA=" "$profile_path" | cut -d'=' -f2)
        if [[ "$enable_lora" == "true" ]]; then
            echo -e "LoRA: ${GREEN}Enabled${NC}"
            local lora_modules=$(grep "^LORA_MODULES=" "$profile_path" | cut -d'=' -f2-)
            if [[ -n "$lora_modules" ]]; then
                echo "LoRA Modules: $lora_modules"
            fi
        fi

        if [[ "$use_dev" == "true" ]]; then
            echo -e "Build: ${YELLOW}Dev (from source) - vllm-dev:$image_tag${NC}"
        fi
    fi

    return $result
}

run_down() {
    local profile_path=$1
    local profile_name=$2
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    # Check if container exists
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo -e "${YELLOW}$profile_name is not running${NC}"
        return 0
    fi

    echo -e "${YELLOW}Stopping $profile_name...${NC}"

    docker stop "$container_name" >/dev/null 2>&1
    docker rm "$container_name" >/dev/null 2>&1

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}$profile_name stopped successfully!${NC}"
    else
        echo -e "${RED}Failed to stop $profile_name${NC}"
        return 1
    fi
}

run_logs() {
    local profile_path=$1
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    echo -e "${BLUE}Showing logs for $container_name...${NC}"
    docker logs -f "$container_name"
}

run_status() {
    local profile_path=$1
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    echo -e "${BLUE}Status for $container_name:${NC}"
    docker ps -a --filter "name=^${container_name}$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

show_ps() {
    echo -e "${BLUE}Running vLLM Containers:${NC}"
    echo ""
    docker ps --filter "ancestor=vllm/vllm-openai:v0.13.0" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

show_gpu() {
    echo -e "${BLUE}GPU Usage:${NC}"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name mem_used mem_total util; do
        echo "GPU $idx: $name"
        echo "  Memory: ${mem_used}MB / ${mem_total}MB"
        echo "  Utilization: ${util}%"
        echo ""
    done
}

show_images() {
    echo -e "${BLUE}vLLM Development Images:${NC}"
    echo ""

    if ! docker images vllm-dev --format "{{.Tag}}" 2>/dev/null | grep -q .; then
        echo -e "${YELLOW}No vllm-dev images found.${NC}"
        echo ""
        echo "Build one with: ./run.sh build [branch]"
        return 0
    fi

    printf "%-30s %-15s %-20s\n" "TAG" "SIZE" "CREATED"
    echo "---------------------------------------------------------------------"

    docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | \
    while IFS=$'\t' read -r tag size created; do
        # Parse created date (format: 2026-01-16 17:47:23 +0900 KST)
        created_short=$(echo "$created" | cut -d' ' -f1,2)
        printf "%-30s %-15s %-20s\n" "$tag" "$size" "$created_short"
    done

    echo ""
    echo "Usage: ./run.sh <profile> up --dev --tag <TAG>"
}

# Main logic
case "$1" in
    ""|"-h"|"--help"|"help")
        show_help
        ;;
    "list")
        list_profiles
        ;;
    "images")
        show_images
        ;;
    "ps")
        show_ps
        ;;
    "gpu")
        show_gpu
        ;;
    "build")
        shift
        run_build "$@"
        ;;
    *)
        PROFILE_NAME=$1
        ACTION=$2
        shift 2

        PROFILE_PATH=$(find_profile "$PROFILE_NAME")

        if [[ -z "$PROFILE_PATH" ]]; then
            echo -e "${RED}Error: Profile '$PROFILE_NAME' not found${NC}"
            echo ""
            list_profiles
            exit 1
        fi

        case "$ACTION" in
            "up")
                USE_DEV="false"
                CUSTOM_TAG=""

                # Parse options
                while [[ $# -gt 0 ]]; do
                    case $1 in
                        --dev)
                            USE_DEV="true"
                            shift
                            ;;
                        --tag)
                            CUSTOM_TAG="$2"
                            shift 2
                            ;;
                        *)
                            shift
                            ;;
                    esac
                done

                run_up "$PROFILE_PATH" "$PROFILE_NAME" "$USE_DEV" "$CUSTOM_TAG"
                ;;
            "down")
                run_down "$PROFILE_PATH" "$PROFILE_NAME"
                ;;
            "logs")
                run_logs "$PROFILE_PATH"
                ;;
            "status")
                run_status "$PROFILE_PATH"
                ;;
            *)
                echo -e "${RED}Error: Unknown action '$ACTION'${NC}"
                echo "Available actions: up, down, logs, status"
                exit 1
                ;;
        esac
        ;;
esac
