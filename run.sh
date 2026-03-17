#!/bin/bash

# vLLM Container Management Script
# Usage: ./run.sh <profile> <action>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="$SCRIPT_DIR/profiles"
COMMON_ENV="$SCRIPT_DIR/.env.common"

# Load library modules
source "$SCRIPT_DIR/lib/colors.sh"
source "$SCRIPT_DIR/lib/validation.sh"
source "$SCRIPT_DIR/lib/tui.sh"

# Temp file tracking for automatic cleanup on exit
TEMP_FILES=()
cleanup_temp_files() {
    for f in "${TEMP_FILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup_temp_files EXIT

# Create a tracked temp file (auto-cleaned on exit)
create_temp() {
    local tmp=$(mktemp)
    TEMP_FILES+=("$tmp")
    echo "$tmp"
}

# Cache for version info (valid for current execution only)
declare -A VERSION_CACHE

# Load feature modules (after shared variables are set)
source "$SCRIPT_DIR/lib/version.sh"
source "$SCRIPT_DIR/lib/build.sh"
source "$SCRIPT_DIR/lib/container.sh"
source "$SCRIPT_DIR/lib/menus.sh"

# Interactive mode entry point
run_interactive() {
    # Try Textual TUI first (Python), fallback to whiptail/dialog
    if command -v python3 &> /dev/null; then
        local tui_dir="$SCRIPT_DIR/tui"
        if [[ -f "$tui_dir/app.py" ]]; then
            # Check if textual is available
            if ! python3 -c "import textual" 2>/dev/null; then
                echo -e "${YELLOW}Textual not found. Installing...${NC}"
                if command -v uv &> /dev/null && [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
                    (cd "$SCRIPT_DIR" && uv sync 2>/dev/null)
                    if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
                        PYTHONPATH="$SCRIPT_DIR" "$SCRIPT_DIR/.venv/bin/python" -m tui.app
                        return $?
                    fi
                fi
                pip install textual 2>/dev/null
            fi
            if python3 -c "import textual" 2>/dev/null; then
                if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
                    PYTHONPATH="$SCRIPT_DIR" "$SCRIPT_DIR/.venv/bin/python" -m tui.app
                else
                    PYTHONPATH="$SCRIPT_DIR" python3 -m tui.app
                fi
                return $?
            fi
            echo -e "${YELLOW}Textual not installed. Install with: pip install textual${NC}"
            echo -e "${YELLOW}Falling back to whiptail/dialog TUI...${NC}"
            echo ""
        fi
    fi

    # Fallback: legacy whiptail/dialog TUI
    if ! check_tui_tool; then
        exit 1
    fi
    show_main_menu
}

#=============================================================================
# CLI FUNCTIONS
#=============================================================================

show_help() {
    echo -e "${BLUE}vLLM Container Management${NC}"
    echo ""
    echo "Usage: ./run.sh [command] | <profile> <action> [options]"
    echo ""
    echo "Interactive Mode:"
    echo "  ./run.sh             - Launch interactive TUI menu"
    echo "  ./run.sh -i          - Launch interactive TUI menu"
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
    echo "  ./run.sh build [branch]                      # Build with auto date tag"
    echo "  ./run.sh build [branch] --tag TAG            # Build with custom tag"
    echo "  ./run.sh build [branch] --repo <repo-url>    # Build from custom repo"
    echo "  ./run.sh build --official                    # Build for all GPU architectures"
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
    echo "  ./run.sh version     # Show version information"
    echo "  ./run.sh images      # List built vllm-dev images"
    echo "  ./run.sh ps          # Show all running vLLM containers"
    echo "  ./run.sh gpu         # Show GPU usage"
}

list_profiles() {
    echo -e "${BLUE}Available Profiles:${NC}"
    echo ""
    printf "%-15s %-8s %-6s %-6s %-8s %s\n" "PROFILE" "STATUS" "GPU" "PORT" "LORA" "MODEL"
    echo "-------------------------------------------------------------------------------"

    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local profile_name=$(basename "$profile" .env)
            local container_name=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            local gpu_id=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
            local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
            local config=$(grep "^CONFIG_NAME=" "$profile" | cut -d'=' -f2)
            local enable_lora=$(grep "^ENABLE_LORA=" "$profile" | cut -d'=' -f2)

            local status="stopped"
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container_name}$"; then
                status="running"
            fi

            local config_file="$SCRIPT_DIR/config/${config}.yaml"
            local model_name=""
            if [[ -f "$config_file" ]]; then
                model_name=$(grep "^model:" "$config_file" | cut -d':' -f2- | sed 's/^ *//')
            else
                model_name="$config"
            fi

            local lora_status="-"
            if [[ "$enable_lora" == "true" ]]; then
                lora_status="ON"
            fi

            printf "%-15s %-8s %-6s %-6s %-8s %s\n" "$profile_name" "$status" "$gpu_id" "$port" "$lora_status" "$model_name"
        fi
    done
    echo ""

    # Show local latest image
    local local_latest=$(get_local_latest_image)
    echo -e "${GREEN}Local latest vLLM image:${NC} $local_latest"
    echo -e "${YELLOW}Tip: Run './run.sh version' to see available versions${NC}"
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

# Main logic
case "$1" in
    "")
        # No arguments - run interactive TUI mode
        run_interactive
        ;;
    "-h"|"--help"|"help")
        show_help
        ;;
    "-i"|"--interactive"|"interactive")
        run_interactive
        ;;
    "list")
        list_profiles
        ;;
    "version")
        show_version_info
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
        ACTION=${2:-}

        if [[ -z "$ACTION" ]]; then
            echo -e "${RED}Error: No action specified${NC}"
            echo "Usage: ./run.sh <profile> <up|down|logs|status>"
            exit 1
        fi

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
