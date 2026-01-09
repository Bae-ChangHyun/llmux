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
    echo "Usage: ./run.sh <profile> <action>"
    echo ""
    echo "Actions:"
    echo "  up      - Start container"
    echo "  down    - Stop container"
    echo "  logs    - Show container logs (follow mode)"
    echo "  status  - Show container status"
    echo ""
    echo "Examples:"
    echo "  ./run.sh vlm up      # Start VLM container"
    echo "  ./run.sh llm down    # Stop LLM container"
    echo "  ./run.sh clova logs  # Show CLOVA logs"
    echo ""
    echo "Other commands:"
    echo "  ./run.sh list        # List available profiles"
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

            if [[ "$enable_lora" == "true" ]]; then
                lora_status="ON"
            else
                lora_status="-"
            fi

            printf "%-15s %-6s %-6s %-8s %s\n" "$profile_name" "$gpu_id" "$port" "$lora_status" "$config"
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

run_up() {
    local profile_path=$1
    local profile_name=$2

    # Check for conflicts
    if ! check_conflict "$profile_path"; then
        return 1
    fi

    echo -e "${GREEN}Starting $profile_name...${NC}"

    # Build LoRA options and export as environment variable
    export LORA_OPTIONS=$(build_lora_options "$profile_path")

    cd "$SCRIPT_DIR"
    docker compose --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d
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
    fi

    return $result
}

run_down() {
    local profile_name=$1

    echo -e "${YELLOW}Stopping $profile_name...${NC}"

    cd "$SCRIPT_DIR"
    docker compose -p "$profile_name" down

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}$profile_name stopped successfully!${NC}"
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

# Main logic
case "$1" in
    ""|"-h"|"--help"|"help")
        show_help
        ;;
    "list")
        list_profiles
        ;;
    "ps")
        show_ps
        ;;
    "gpu")
        show_gpu
        ;;
    *)
        PROFILE_NAME=$1
        ACTION=$2

        PROFILE_PATH=$(find_profile "$PROFILE_NAME")

        if [[ -z "$PROFILE_PATH" ]]; then
            echo -e "${RED}Error: Profile '$PROFILE_NAME' not found${NC}"
            echo ""
            list_profiles
            exit 1
        fi

        case "$ACTION" in
            "up")
                run_up "$PROFILE_PATH" "$PROFILE_NAME"
                ;;
            "down")
                run_down "$PROFILE_NAME"
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
