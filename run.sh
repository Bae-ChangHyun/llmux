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

#=============================================================================
# VERSION MANAGEMENT FUNCTIONS
#=============================================================================

# Fetch version info from Docker Hub
fetch_docker_hub_version() {
    local tag=$1
    local cache_key="dockerhub_${tag}"

    # Return cached value if exists
    if [[ -n "${VERSION_CACHE[$cache_key]}" ]]; then
        echo "${VERSION_CACHE[$cache_key]}"
        return 0
    fi

    # Query Docker Hub API
    local api_url="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/${tag}"
    local response=$(curl -s --connect-timeout 5 --max-time 10 "$api_url" 2>/dev/null)

    if [[ -z "$response" ]] || echo "$response" | grep -q "not found"; then
        echo "unknown"
        return 1
    fi

    # Extract last updated date
    local last_updated=$(echo "$response" | grep -o '"last_updated":"[^"]*"' | cut -d'"' -f4 | cut -d'T' -f1)

    # Cache and return
    VERSION_CACHE[$cache_key]="$last_updated"
    echo "$last_updated"
}

# Get latest release version tag
get_latest_release_version() {
    local cache_key="latest_release"

    if [[ -n "${VERSION_CACHE[$cache_key]}" ]]; then
        echo "${VERSION_CACHE[$cache_key]}"
        return 0
    fi

    # Get tags from Docker Hub
    local api_url="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100"
    local response=$(curl -s --connect-timeout 5 --max-time 10 "$api_url" 2>/dev/null)

    # Extract version tags (v0.x.x format)
    local version=$(echo "$response" | grep -o '"name":"v[0-9]\+\.[0-9]\+\.[0-9]\+"' | head -1 | cut -d'"' -f4)

    if [[ -z "$version" ]]; then
        version="unknown"
    fi

    VERSION_CACHE[$cache_key]="$version"
    echo "$version"
}

# Get current container version (detailed info)
get_current_container_version() {
    local container_name=$1

    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container_name}$"; then
        echo "Not running"
        return 1
    fi

    local image=$(docker inspect "$container_name" --format='{{.Config.Image}}' 2>/dev/null)
    local image_id=$(docker inspect "$container_name" --format='{{.Image}}' 2>/dev/null | cut -c8-19)

    # Get image creation date
    local created=$(docker inspect "$image" --format='{{.Created}}' 2>/dev/null | cut -d'T' -f1)

    echo "$image (ID: $image_id, Created: $created)"
}

# Get latest locally available vllm/vllm-openai image
# Usage: get_local_latest_image [--tag-only]
get_local_latest_image() {
    local tag_only="false"
    [[ "$1" == "--tag-only" ]] && tag_only="true"

    # List local images sorted by creation date (newest first)
    local latest_line=$(docker images vllm/vllm-openai --format "{{.Tag}}\t{{.CreatedAt}}" 2>/dev/null | head -1)

    if [[ -z "$latest_line" ]]; then
        if [[ "$tag_only" == "true" ]]; then
            echo "none"
        else
            echo "No local images"
        fi
        return 1
    fi

    local tag=$(echo "$latest_line" | cut -f1)
    local created=$(echo "$latest_line" | cut -f2 | cut -d' ' -f1)

    if [[ "$tag_only" == "true" ]]; then
        echo "$tag"
    else
        echo "$tag (Created: $created)"
    fi
}

# Get current profile's container version
get_profile_container_version() {
    local profile_path=$1
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    get_current_container_version "$container_name"
}

# Cleanup unused vLLM images
cleanup_unused_vllm_images() {
    local image_pattern=$1  # "nightly" or "v*" for official releases
    local exclude_tag=$2    # Tag to exclude from cleanup (e.g., just pulled tag)

    echo -e "${YELLOW}Cleaning up unused vLLM images (pattern: $image_pattern)...${NC}"

    # Get all vllm images matching pattern
    local all_images=$(docker images "vllm/vllm-openai" --format "{{.Repository}}:{{.Tag}}" | grep "$image_pattern")

    # Get images in use by containers
    local used_images=$(docker ps -a --format "{{.Image}}" | grep "vllm/vllm-openai" | grep "$image_pattern" | sort -u)

    # Delete unused images
    local deleted_count=0
    for img in $all_images; do
        # Skip excluded tag
        if [[ -n "$exclude_tag" && "$img" == "vllm/vllm-openai:$exclude_tag" ]]; then
            continue
        fi
        if ! echo "$used_images" | grep -q "^${img}$"; then
            echo "  Removing unused image: $img"
            docker rmi "$img" 2>/dev/null && ((deleted_count++))
        fi
    done

    if [[ $deleted_count -gt 0 ]]; then
        echo -e "${GREEN}Cleaned up $deleted_count unused image(s)${NC}"
    else
        echo -e "${BLUE}No unused images to clean up${NC}"
    fi
}

# Pull latest image and cleanup old ones
pull_and_cleanup() {
    local tag=$1
    local cleanup_pattern=$2

    echo -e "${BLUE}Pulling vllm/vllm-openai:$tag...${NC}"
    docker pull "vllm/vllm-openai:$tag"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Successfully pulled vllm/vllm-openai:$tag${NC}"
        # Cleanup old images, excluding the just-pulled tag
        cleanup_unused_vllm_images "$cleanup_pattern" "$tag"
        return 0
    else
        echo -e "${RED}Failed to pull image${NC}"
        return 1
    fi
}

# Show version information
show_version_info() {
    echo -e "${BLUE}=== vLLM Version Information ===${NC}"
    echo ""

    # Local latest image
    local local_latest=$(get_local_latest_image)
    echo -e "${GREEN}Local Latest:${NC} vllm/vllm-openai:${local_latest}"
    echo ""

    # Running containers and their versions
    echo -e "${BLUE}Running Containers:${NC}"
    local found_running=false
    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local container_name=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container_name}$"; then
                local version=$(get_current_container_version "$container_name")
                echo "  - $container_name: $version"
                found_running=true
            fi
        fi
    done
    if [[ "$found_running" == "false" ]]; then
        echo "  (No running containers)"
    fi
    echo ""

    # Latest release
    echo -n -e "${YELLOW}Fetching Docker Hub info...${NC}"
    local latest_version=$(get_latest_release_version)
    local nightly_date=$(fetch_docker_hub_version "nightly")
    echo -e "\r                                      \r"  # Clear line

    echo -e "${GREEN}Available Versions:${NC}"
    echo "  - Official Latest: ${latest_version}"
    echo "  - Nightly: Updated on ${nightly_date}"

    # Dev builds
    local dev_count=$(docker images vllm-dev --format "{{.Tag}}" 2>/dev/null | wc -l)
    echo "  - Dev Builds: ${dev_count} local build(s)"
    if [[ $dev_count -gt 0 ]]; then
        docker images vllm-dev --format "    • {{.Tag}} ({{.Size}}, {{.CreatedSince}})" 2>/dev/null
    fi

    echo ""
}

#=============================================================================
# TUI MENU FUNCTIONS
#=============================================================================

# Quick Setup - Create Profile + Config at once
quick_setup_menu() {
    # Model name input
    local model=$(tui_inputbox "Quick Setup" "Enter HuggingFace model name:\n(e.g., lightonai/LightOnOCR-2-1B)" "")
    [[ -z "$model" ]] && return

    # Extract name from model (part after last /)
    local name=$(echo "$model" | rev | cut -d'/' -f1 | rev)

    # Convert to lowercase and replace invalid chars for container name
    local safe_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-' | sed 's/-*$//')

    # Check if already exists
    if [[ -f "$PROFILES_DIR/$safe_name.env" ]]; then
        tui_msgbox "Error" "Profile '$safe_name' already exists."
        return
    fi
    if [[ -f "$SCRIPT_DIR/config/$safe_name.yaml" ]]; then
        tui_msgbox "Error" "Config '$safe_name' already exists."
        return
    fi

    # GPU ID
    local gpu=$(tui_inputbox "GPU" "Enter GPU ID (e.g., 0 or 0,1):" "0")
    [[ -z "$gpu" ]] && return
    if ! validate_gpu_id "$gpu"; then
        tui_msgbox "Invalid Input" "GPU ID must be a number or comma-separated numbers (e.g., 0 or 0,1)"
        return
    fi

    # Port
    local port=$(tui_inputbox "Port" "Enter vLLM port:" "8000")
    [[ -z "$port" ]] && return
    if ! validate_port "$port"; then
        tui_msgbox "Invalid Input" "Port must be between 1024 and 65535"
        return
    fi

    # GPU memory utilization
    local gpu_util=$(tui_inputbox "GPU Memory" "GPU memory utilization (0.0-1.0):" "0.9")
    [[ -z "$gpu_util" ]] && return
    if ! validate_gpu_memory "$gpu_util"; then
        tui_msgbox "Invalid Input" "GPU memory utilization must be between 0.0 and 1.0"
        return
    fi

    # Tensor parallel (auto-detect from GPU count)
    local gpu_count=$(echo "$gpu" | tr ',' '\n' | wc -l)
    local tp="$gpu_count"

    # Confirmation
    local msg="Model: $model\n"
    msg="${msg}Name: $safe_name\n"
    msg="${msg}GPU: $gpu (TP=$tp)\n"
    msg="${msg}Port: $port\n"
    msg="${msg}GPU Util: $gpu_util"

    if ! tui_yesno "Confirm Quick Setup" "$msg\n\nCreate profile and config?"; then
        return
    fi

    # Create files to temp first, then move (atomic)
    mkdir -p "$SCRIPT_DIR/config" "$PROFILES_DIR"

    local config_path="$SCRIPT_DIR/config/$safe_name.yaml"
    local profile_path="$PROFILES_DIR/$safe_name.env"
    local tmp_config="/tmp/vllm-$safe_name.yaml"
    local tmp_profile="/tmp/vllm-$safe_name.env"

    printf 'model: %s\ngpu-memory-utilization: %s\n' "$model" "$gpu_util" > "$tmp_config"

    cat > "$tmp_profile" << EOF
# Profile: $safe_name
# Model: $model
# GPU: $gpu, Port: $port

CONTAINER_NAME=$safe_name
VLLM_PORT=$port
CONFIG_NAME=$safe_name

# GPU Configuration
GPU_ID=$gpu
TENSOR_PARALLEL_SIZE=$tp

# LoRA Configuration (optional)
ENABLE_LORA=false
#MAX_LORAS=2
#MAX_LORA_RANK=16
#LORA_MODULES=adapter1=/app/lora/path1
EOF

    # Move both files - if either fails, clean up both
    local err=""
    cp "$tmp_config" "$config_path" 2>/tmp/vllm-setup-err.log || err="config"
    if [[ -z "$err" ]]; then
        cp "$tmp_profile" "$profile_path" 2>>/tmp/vllm-setup-err.log || err="profile"
    fi

    rm -f "$tmp_config" "$tmp_profile"

    if [[ -n "$err" ]]; then
        # Rollback: remove both if either failed
        rm -f "$config_path" "$profile_path"
        local err_detail=$(cat /tmp/vllm-setup-err.log 2>/dev/null)
        tui_msgbox "Error" "Failed to create $err file.\n\n${err_detail:-Check directory permissions.}\n\nTry: sudo chown \$USER:$USER $SCRIPT_DIR/config/"
    else
        tui_msgbox "Success" "Created:\n- Config: config/$safe_name.yaml\n- Profile: profiles/$safe_name.env\n\nStart with: ./run.sh $safe_name up"
    fi
}

# Main Menu
show_main_menu() {
    while true; do
        local choice=$(tui_menu "vLLM Container Management" "Select an option:" \
            "Q" "Quick Setup (Profile + Config)" \
            "1" "Container Management (up/down/logs/status)" \
            "2" "Profile Management (create/edit/delete)" \
            "3" "Config Management (create/edit)" \
            "4" "Build Management (build/images)" \
            "5" "System Info (GPU/PS)" \
            "X" "Exit")

        case "$choice" in
            Q) quick_setup_menu ;;
            1) container_menu ;;
            2) profile_menu ;;
            3) config_menu ;;
            4) build_menu ;;
            5) system_menu ;;
            X|"") break ;;
        esac
    done
}

# Container Management Menu
container_menu() {
    while true; do
        local choice=$(tui_menu "Container Management" "Select an option:" \
            "1" "Start Container (up)" \
            "2" "Stop Container (down)" \
            "3" "View Logs" \
            "4" "Check Status" \
            "B" "Back to Main Menu")

        case "$choice" in
            1) container_up_menu ;;
            2) container_down_menu ;;
            3) container_logs_menu ;;
            4) container_status_menu ;;
            B|"") break ;;
        esac
    done
}

# Container Up Menu
container_up_menu() {
    build_all_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No profiles found. Create a profile first."
        return
    fi

    local profile=$(tui_menu "Start Container" "Select profile:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"

    # Get local latest image info
    local local_latest=$(get_local_latest_image)

    # Fetch version info from Docker Hub
    echo "Fetching version information..."
    local latest_release=$(get_latest_release_version)
    local nightly_date=$(fetch_docker_hub_version "nightly")

    # Build menu options
    local menu_options=(
        "local" "Local Latest: $local_latest (no pull)"
        "official" "Official Latest: $latest_release (pull)"
        "nightly" "Nightly: $nightly_date (pull)"
        "dev" "Dev build (local source builds)"
        "custom" "Custom tag (specify exact version)"
    )

    # Select version type
    local version_choice=$(tui_menu "Select vLLM Version" "Profile: $profile" "${menu_options[@]}")
    [[ -z "$version_choice" ]] && return

    local use_dev="false"
    local version_tag=""
    local custom_tag=""
    local need_pull="false"

    case "$version_choice" in
        local)
            local local_tag=$(get_local_latest_image --tag-only)
            if [[ "$local_tag" == "none" ]]; then
                tui_msgbox "Error" "No local vllm/vllm-openai images found.\nSelect Official or Nightly to pull an image first."
                return
            fi
            version_tag="$local_tag"
            ;;
        official)
            version_tag="$latest_release"
            if tui_yesno "Pull Official Latest" "Pull vllm/vllm-openai:$latest_release?"; then
                clear
                if ! pull_and_cleanup "$latest_release" "v[0-9]*"; then
                    echo -e "${YELLOW}Press Enter to continue...${NC}"
                    read -r
                    return
                fi
                echo ""
                echo -e "${YELLOW}Press Enter to continue...${NC}"
                read -r
            else
                return
            fi
            ;;
        nightly)
            version_tag="nightly"
            if tui_yesno "Pull Nightly Build" "Pull latest nightly build ($nightly_date)?"; then
                clear
                if ! pull_and_cleanup "nightly" "nightly"; then
                    echo -e "${YELLOW}Press Enter to continue...${NC}"
                    read -r
                    return
                fi
                echo ""
                echo -e "${YELLOW}Press Enter to continue...${NC}"
                read -r
            else
                return
            fi
            ;;
        dev)
            use_dev="true"
            build_dev_image_items
            if [[ ${#MENU_ITEMS[@]} -gt 0 ]]; then
                local tag_choice=$(tui_menu "Select Dev Image" "Choose dev image tag:" "${MENU_ITEMS[@]}")
                if [[ -n "$tag_choice" ]]; then
                    custom_tag="$tag_choice"
                else
                    return
                fi
            else
                tui_msgbox "Error" "No dev images found. Build one first with: ./run.sh build"
                return
            fi
            ;;
        custom)
            version_tag=$(tui_inputbox "Custom Tag" "Enter custom tag (e.g., v0.14.0):" "")
            [[ -z "$version_tag" ]] && return
            if ! [[ "$version_tag" =~ ^[a-zA-Z0-9._-]+$ ]]; then
                tui_msgbox "Invalid Input" "Tag must contain only letters, numbers, dots, dashes, and underscores"
                return
            fi
            need_pull="auto"
            ;;
    esac

    # Confirmation
    local msg="Profile: $profile\n"
    if [[ "$use_dev" == "true" ]]; then
        msg="${msg}Version: Dev Build (vllm-dev:$custom_tag)"
    else
        msg="${msg}Version: vllm/vllm-openai:$version_tag"
    fi

    if tui_yesno "Confirm Start" "$msg\n\nStart this container?"; then
        clear
        run_up "$profile_path" "$profile" "$use_dev" "$custom_tag" "$version_tag" "$need_pull"
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    fi
}

# Container Down Menu
container_down_menu() {
    build_running_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Info" "No running containers found."
        return
    fi

    local profile=$(tui_menu "Stop Container" "Select container to stop:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"

    if tui_yesno "Confirm Stop" "Stop container for profile '$profile'?"; then
        clear
        run_down "$profile_path" "$profile"
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    fi
}

# Container Logs Menu
container_logs_menu() {
    build_running_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Info" "No running containers found."
        return
    fi

    local profile=$(tui_menu "View Logs" "Select container:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    clear
    echo -e "${BLUE}Showing logs for $container_name (Ctrl+C to exit)...${NC}"
    echo ""
    docker logs -f "$container_name" 2>&1 || echo -e "${YELLOW}Container not running or not found.${NC}"
    echo ""
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read -r
}

# Container Status Menu
container_status_menu() {
    build_all_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(tui_menu "Check Status" "Select profile:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"
    local container_name=$(grep "^CONTAINER_NAME=" "$profile_path" | cut -d'=' -f2)

    local status=$(docker ps -a --filter "name=^${container_name}$" --format "Status: {{.Status}}\nPorts: {{.Ports}}" 2>/dev/null)

    if [[ -z "$status" ]]; then
        status="Container not found"
    fi

    tui_msgbox "Status: $container_name" "$status"
}

# Profile Management Menu
profile_menu() {
    while true; do
        local choice=$(tui_menu "Profile Management" "Select an option:" \
            "1" "Create New Profile" \
            "2" "Edit Profile" \
            "3" "Delete Profile" \
            "4" "View Profile Details" \
            "B" "Back to Main Menu")

        case "$choice" in
            1) profile_create_menu ;;
            2) profile_edit_menu ;;
            3) profile_delete_menu ;;
            4) profile_view_menu ;;
            B|"") break ;;
        esac
    done
}

# Profile Create Menu
profile_create_menu() {
    # Profile name
    local name=$(tui_inputbox "Create Profile" "Enter profile name:" "")
    [[ -z "$name" ]] && return

    if ! validate_name "$name"; then
        tui_msgbox "Invalid Name" "Name must contain only letters, numbers, dash, and underscore"
        return
    fi

    # Check if exists
    if [[ -f "$PROFILES_DIR/$name.env" ]]; then
        tui_msgbox "Error" "Profile '$name' already exists."
        return
    fi

    # Container name
    local container=$(tui_inputbox "Container Name" "Enter container name:" "$name")
    [[ -z "$container" ]] && return

    # Port
    local port=$(tui_inputbox "Port" "Enter vLLM port:" "8000")
    [[ -z "$port" ]] && return

    # GPU ID
    local gpu=$(tui_inputbox "GPU ID" "Enter GPU ID (e.g., 0 or 0,1):" "0")
    [[ -z "$gpu" ]] && return

    # Tensor parallel size
    local tp=$(tui_inputbox "Tensor Parallel" "Enter tensor parallel size:" "1")
    [[ -z "$tp" ]] && return

    build_config_menu_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No configs found. Create a config first."
        return
    fi

    local config=$(tui_menu "Select Config" "Choose model config:" "${MENU_ITEMS[@]}")
    [[ -z "$config" ]] && return

    # LoRA settings
    local enable_lora="false"
    if tui_yesno "LoRA" "Enable LoRA support?"; then
        enable_lora="true"
    fi

    # Create profile file
    cat > "$PROFILES_DIR/$name.env" << EOF
# Profile: $name
# GPU: $gpu, Port: $port

CONTAINER_NAME=$container
VLLM_PORT=$port
CONFIG_NAME=$config

# GPU Configuration
GPU_ID=$gpu
TENSOR_PARALLEL_SIZE=$tp

# LoRA Configuration (optional)
ENABLE_LORA=$enable_lora
#MAX_LORAS=2
#MAX_LORA_RANK=16
#LORA_MODULES=adapter1=/app/lora/path1
EOF

    tui_msgbox "Success" "Profile '$name' created successfully!"
}

# Profile Edit Menu
profile_edit_menu() {
    build_all_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(tui_menu "Edit Profile" "Select profile to edit:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"

    while true; do
        local current_port=$(grep "^VLLM_PORT=" "$profile_path" | cut -d'=' -f2)
        local current_gpu=$(grep "^GPU_ID=" "$profile_path" | cut -d'=' -f2)
        local current_tp=$(grep "^TENSOR_PARALLEL_SIZE=" "$profile_path" | cut -d'=' -f2)
        local current_config=$(grep "^CONFIG_NAME=" "$profile_path" | cut -d'=' -f2)
        local current_lora=$(grep "^ENABLE_LORA=" "$profile_path" | cut -d'=' -f2)

        local choice=$(tui_menu "Edit: $profile" "Current: Port=$current_port GPU=$current_gpu" \
            "1" "Change Port (current: $current_port)" \
            "2" "Change GPU ID (current: $current_gpu)" \
            "3" "Change Tensor Parallel (current: $current_tp)" \
            "4" "Change Config (current: $current_config)" \
            "5" "Toggle LoRA (current: $current_lora)" \
            "B" "Back")

        case "$choice" in
            1)
                local new_port=$(tui_inputbox "Change Port" "Enter new port:" "$current_port")
                if [[ -n "$new_port" ]]; then
                    if validate_port "$new_port"; then
                        sed -i "s/^VLLM_PORT=.*/VLLM_PORT=$new_port/" "$profile_path"
                        tui_msgbox "Updated" "Port changed to $new_port"
                    else
                        tui_msgbox "Invalid Input" "Port must be between 1024 and 65535"
                    fi
                fi
                ;;
            2)
                local new_gpu=$(tui_inputbox "Change GPU" "Enter GPU ID(s):" "$current_gpu")
                if [[ -n "$new_gpu" ]]; then
                    if validate_gpu_id "$new_gpu"; then
                        sed -i "s/^GPU_ID=.*/GPU_ID=$new_gpu/" "$profile_path"
                        tui_msgbox "Updated" "GPU ID changed to $new_gpu"
                    else
                        tui_msgbox "Invalid Input" "GPU ID must be a number or comma-separated numbers"
                    fi
                fi
                ;;
            3)
                local new_tp=$(tui_inputbox "Tensor Parallel" "Enter tensor parallel size:" "$current_tp")
                if [[ -n "$new_tp" ]] && [[ "$new_tp" =~ ^[0-9]+$ ]]; then
                    sed -i "s/^TENSOR_PARALLEL_SIZE=.*/TENSOR_PARALLEL_SIZE=$new_tp/" "$profile_path"
                    tui_msgbox "Updated" "Tensor parallel changed to $new_tp"
                elif [[ -n "$new_tp" ]]; then
                    tui_msgbox "Invalid Input" "Tensor parallel size must be a positive integer"
                fi
                ;;
            4)
                build_config_menu_items
                local new_config=$(tui_menu "Select Config" "Choose config:" "${MENU_ITEMS[@]}")
                if [[ -n "$new_config" ]]; then
                    sed -i "s/^CONFIG_NAME=.*/CONFIG_NAME=$new_config/" "$profile_path"
                    tui_msgbox "Updated" "Config changed to $new_config"
                fi
                ;;
            5)
                if [[ "$current_lora" == "true" ]]; then
                    sed -i "s/^ENABLE_LORA=.*/ENABLE_LORA=false/" "$profile_path"
                    tui_msgbox "Updated" "LoRA disabled"
                else
                    sed -i "s/^ENABLE_LORA=.*/ENABLE_LORA=true/" "$profile_path"
                    tui_msgbox "Updated" "LoRA enabled"
                fi
                ;;
            B|"") break ;;
        esac
    done
}

# Profile Delete Menu
profile_delete_menu() {
    build_all_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(tui_menu "Delete Profile" "Select profile to delete:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    if tui_yesno "Confirm Delete" "Are you sure you want to delete profile '$profile'?"; then
        rm -f "$PROFILES_DIR/$profile.env"
        tui_msgbox "Deleted" "Profile '$profile' has been deleted."
    fi
}

# Profile View Menu
profile_view_menu() {
    build_all_profile_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(tui_menu "View Profile" "Select profile:" "${MENU_ITEMS[@]}")
    [[ -z "$profile" ]] && return

    tui_textbox "Profile: $profile" "$PROFILES_DIR/$profile.env"
}

# Config Management Menu
config_menu() {
    while true; do
        local choice=$(tui_menu "Config Management" "Select an option:" \
            "1" "Create New Config" \
            "2" "Edit Config" \
            "3" "Delete Config" \
            "4" "View Config" \
            "B" "Back to Main Menu")

        case "$choice" in
            1) config_create_menu ;;
            2) config_edit_menu ;;
            3) config_delete_menu ;;
            4) config_view_menu ;;
            B|"") break ;;
        esac
    done
}

# Config Create Menu
config_create_menu() {
    local name=$(tui_inputbox "Create Config" "Enter config name:" "")
    [[ -z "$name" ]] && return

    if ! validate_name "$name"; then
        tui_msgbox "Invalid Name" "Name must contain only letters, numbers, dash, and underscore"
        return
    fi

    if [[ -f "$SCRIPT_DIR/config/$name.yaml" ]]; then
        tui_msgbox "Error" "Config '$name' already exists."
        return
    fi

    local model=$(tui_inputbox "Model" "Enter model name (HuggingFace):" "")
    [[ -z "$model" ]] && return

    local gpu_util=$(tui_inputbox "GPU Memory" "GPU memory utilization (0.0-1.0):" "0.9")
    [[ -z "$gpu_util" ]] && return

    mkdir -p "$SCRIPT_DIR/config"
    cat > "$SCRIPT_DIR/config/$name.yaml" << EOF
model: $model
gpu-memory-utilization: $gpu_util
EOF

    tui_msgbox "Success" "Config '$name' created successfully!"
}

# Config Edit Menu
config_edit_menu() {
    build_config_menu_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(tui_menu "Edit Config" "Select config:" "${MENU_ITEMS[@]}")
    [[ -z "$config" ]] && return

    local config_path="$SCRIPT_DIR/config/$config.yaml"

    while true; do
        local current_model=$(grep "^model:" "$config_path" | cut -d':' -f2- | sed 's/^ *//')
        local current_util=$(grep "^gpu-memory-utilization:" "$config_path" | cut -d':' -f2- | sed 's/^ *//')

        local choice=$(tui_menu "Edit: $config" "Model: $current_model" \
            "1" "Change Model (current: $current_model)" \
            "2" "Change GPU Utilization (current: $current_util)" \
            "3" "Add Custom Parameter" \
            "4" "Edit Custom Parameter" \
            "5" "Delete Custom Parameter" \
            "B" "Back")

        case "$choice" in
            1)
                local new_model=$(tui_inputbox "Model" "Enter model name:" "$current_model")
                if [[ -n "$new_model" ]]; then
                    # Sanitize model name for sed (using | delimiter)
                    local safe_model=$(sanitize_for_sed "$new_model" "|")
                    sed -i "s|^model:.*|model: $safe_model|" "$config_path"
                    tui_msgbox "Updated" "Model changed"
                fi
                ;;
            2)
                local new_util=$(tui_inputbox "GPU Utilization" "Enter value (0.0-1.0):" "$current_util")
                if [[ -n "$new_util" ]]; then
                    if validate_gpu_memory "$new_util"; then
                        if grep -q "^gpu-memory-utilization:" "$config_path"; then
                            sed -i "s|^gpu-memory-utilization:.*|gpu-memory-utilization: $new_util|" "$config_path"
                        else
                            echo "gpu-memory-utilization: $new_util" >> "$config_path"
                        fi
                        tui_msgbox "Updated" "GPU utilization changed"
                    else
                        tui_msgbox "Invalid Input" "Value must be between 0.0 and 1.0"
                    fi
                fi
                ;;
            3)
                local param=$(tui_inputbox "Parameter" "Enter parameter name:" "")
                if [[ -n "$param" ]]; then
                    local value=$(tui_inputbox "Value" "Enter value for '$param':" "")
                    if [[ -n "$value" ]]; then
                        echo "$param: $value" >> "$config_path"
                        tui_msgbox "Added" "Parameter '$param' added"
                    fi
                fi
                ;;
            4)
                build_custom_param_items "$config_path"
                if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
                    tui_msgbox "Info" "No custom parameters found."
                else
                    local param=$(tui_menu "Edit Parameter" "Select parameter to edit:" "${MENU_ITEMS[@]}")
                    if [[ -n "$param" ]]; then
                        local current_val=$(grep "^${param}:" "$config_path" | cut -d':' -f2- | sed 's/^ *//')
                        local new_val=$(tui_inputbox "Edit: $param" "Enter new value for '$param':" "$current_val")
                        if [[ -n "$new_val" ]]; then
                            local safe_val=$(sanitize_for_sed "$new_val" "|")
                            sed -i "s|^${param}:.*|${param}: ${safe_val}|" "$config_path"
                            tui_msgbox "Updated" "Parameter '$param' updated"
                        fi
                    fi
                fi
                ;;
            5)
                build_custom_param_items "$config_path"
                if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
                    tui_msgbox "Info" "No custom parameters found."
                else
                    local param=$(tui_menu "Delete Parameter" "Select parameter to delete:" "${MENU_ITEMS[@]}")
                    if [[ -n "$param" ]]; then
                        if tui_yesno "Confirm" "Delete parameter '$param'?"; then
                            sed -i "/^${param}:/d" "$config_path"
                            tui_msgbox "Deleted" "Parameter '$param' deleted"
                        fi
                    fi
                fi
                ;;
            B|"") break ;;
        esac
    done
}

# Config Delete Menu
config_delete_menu() {
    build_config_menu_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(tui_menu "Delete Config" "Select config:" "${MENU_ITEMS[@]}")
    [[ -z "$config" ]] && return

    if tui_yesno "Confirm Delete" "Are you sure you want to delete config '$config'?"; then
        rm -f "$SCRIPT_DIR/config/$config.yaml"
        tui_msgbox "Deleted" "Config '$config' has been deleted."
    fi
}

# Config View Menu
config_view_menu() {
    build_config_menu_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(tui_menu "View Config" "Select config:" "${MENU_ITEMS[@]}")
    [[ -z "$config" ]] && return

    tui_textbox "Config: $config" "$SCRIPT_DIR/config/$config.yaml"
}

# Build Management Menu
build_menu() {
    while true; do
        local choice=$(tui_menu "Build Management" "Select an option:" \
            "1" "Build vLLM (from source)" \
            "2" "List Built Images" \
            "3" "Delete Image" \
            "B" "Back to Main Menu")

        case "$choice" in
            1) build_vllm_menu ;;
            2) build_list_images ;;
            3) build_delete_image ;;
            B|"") break ;;
        esac
    done
}

# Build vLLM Menu
build_vllm_menu() {
    # Select repo
    local repo_choice=$(tui_menu "Repository" "Select vLLM repository:" \
        "official" "vllm-project/vllm (Official)" \
        "custom" "Custom fork/repository")
    [[ -z "$repo_choice" ]] && return

    local repo_url="https://github.com/vllm-project/vllm.git"
    if [[ "$repo_choice" == "custom" ]]; then
        repo_url=$(tui_inputbox "Custom Repository" "Enter repository URL:" "https://github.com/username/vllm.git")
        [[ -z "$repo_url" ]] && return
    fi

    local branch=$(tui_inputbox "Branch" "Enter branch name:" "main")
    [[ -z "$branch" ]] && return

    local build_type=$(tui_menu "Build Type" "Select build type:" \
        "fast" "Fast build (your GPU only) - Recommended" \
        "official" "Official build (all GPUs) - Very slow")
    [[ -z "$build_type" ]] && return

    local custom_tag=""
    if tui_yesno "Custom Tag" "Use custom tag? (No = auto-generated date tag)"; then
        custom_tag=$(tui_inputbox "Tag" "Enter custom tag:" "")
    fi

    # Extract repo name for display
    local repo_name=$(echo "$repo_url" | sed 's|https://github.com/||' | sed 's|\.git||')

    local msg="Repository: $repo_name\nBranch: $branch\nBuild type: $build_type"
    [[ -n "$custom_tag" ]] && msg="$msg\nTag: $custom_tag"

    if tui_yesno "Confirm Build" "$msg\n\nStart build? This may take a while."; then
        clear
        if [[ "$build_type" == "official" ]]; then
            run_build_official "$repo_url" "$branch" "$custom_tag"
        else
            run_build_fast "$repo_url" "$branch" "$custom_tag"
        fi
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    fi
}

# List Built Images
build_list_images() {
    local tmp_file=$(create_temp)

    echo "vLLM Development Images" > "$tmp_file"
    echo "=======================" >> "$tmp_file"
    echo "" >> "$tmp_file"

    if ! docker images vllm-dev --format "{{.Tag}}" 2>/dev/null | grep -q .; then
        echo "No vllm-dev images found." >> "$tmp_file"
        echo "" >> "$tmp_file"
        echo "Build one with: ./run.sh build [branch]" >> "$tmp_file"
        echo "           or: ./run.sh build [branch] --repo <url>" >> "$tmp_file"
    else
        docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | \
        while IFS=$'\t' read -r tag size created; do
            # Get label info
            local repo_url=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.repo.url"}}' 2>/dev/null)
            local branch=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.repo.branch"}}' 2>/dev/null)
            local commit=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.commit.hash"}}' 2>/dev/null)
            local build_date=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.build.date"}}' 2>/dev/null)

            echo "Tag: vllm-dev:$tag" >> "$tmp_file"
            echo "  Size: $size | Created: $created" >> "$tmp_file"

            if [[ -n "$repo_url" && "$repo_url" != "<no value>" ]]; then
                # Extract repo name from URL
                local repo_name=$(echo "$repo_url" | sed 's|https://github.com/||' | sed 's|\.git||')
                echo "  Repository: $repo_name" >> "$tmp_file"
                echo "  Branch: $branch | Commit: $commit" >> "$tmp_file"
                if [[ -n "$build_date" && "$build_date" != "<no value>" ]]; then
                    local build_date_short=$(echo "$build_date" | cut -d'T' -f1)
                    echo "  Built: $build_date_short" >> "$tmp_file"
                fi
            else
                echo "  (Legacy build - no metadata)" >> "$tmp_file"
            fi

            echo "---------------------------------------------------------------" >> "$tmp_file"
        done
    fi

    tui_textbox "Built Images" "$tmp_file"
}

# Delete Image
build_delete_image() {
    build_dev_image_items
    if [[ ${#MENU_ITEMS[@]} -eq 0 ]]; then
        tui_msgbox "Info" "No vllm-dev images found."
        return
    fi

    local tag=$(tui_menu "Delete Image" "Select image to delete:" "${MENU_ITEMS[@]}")
    [[ -z "$tag" ]] && return

    if tui_yesno "Confirm Delete" "Delete image 'vllm-dev:$tag'?"; then
        docker rmi "vllm-dev:$tag" 2>/dev/null
        tui_msgbox "Deleted" "Image 'vllm-dev:$tag' has been deleted."
    fi
}

# System Info Menu
system_menu() {
    while true; do
        local choice=$(tui_menu "System Info" "Select an option:" \
            "1" "GPU Usage" \
            "2" "Running Containers" \
            "3" "All Profiles Status" \
            "4" "Version Information" \
            "B" "Back to Main Menu")

        case "$choice" in
            1) system_gpu_info ;;
            2) system_containers ;;
            3) system_profiles_status ;;
            4) system_version_info ;;
            B|"") break ;;
        esac
    done
}

# System Version Info
system_version_info() {
    local tmp_file=$(create_temp)

    echo "vLLM Version Information" > "$tmp_file"
    echo "========================" >> "$tmp_file"
    echo "" >> "$tmp_file"

    # Local latest image
    local local_latest=$(get_local_latest_image)
    echo "Local Latest: vllm/vllm-openai:$local_latest" >> "$tmp_file"
    echo "" >> "$tmp_file"

    # Fetch latest release
    echo "Fetching version information from Docker Hub..." >> "$tmp_file"
    local latest_release=$(get_latest_release_version)
    local nightly_date=$(fetch_docker_hub_version "nightly")

    # Clear the "Fetching" message
    sed -i '/^Fetching version information/d' "$tmp_file"

    echo "Latest Release:  vllm/vllm-openai:$latest_release" >> "$tmp_file"
    echo "Nightly Build:   Updated on $nightly_date" >> "$tmp_file"
    echo "" >> "$tmp_file"

    # Dev builds
    local dev_count=$(docker images vllm-dev --format "{{.Tag}}" 2>/dev/null | wc -l)
    echo "Dev Builds:      $dev_count local build(s)" >> "$tmp_file"
    if [[ $dev_count -gt 0 ]]; then
        echo "" >> "$tmp_file"
        echo "Available dev images:" >> "$tmp_file"
        docker images vllm-dev --format "  - {{.Tag}} ({{.Size}}, created {{.CreatedSince}})" >> "$tmp_file"
    fi

    tui_textbox "Version Information" "$tmp_file"
}

# System GPU Info
system_gpu_info() {
    local tmp_file=$(create_temp)

    echo "GPU Usage" > "$tmp_file"
    echo "=========" >> "$tmp_file"
    echo "" >> "$tmp_file"

    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx name mem_used mem_total util; do
        echo "GPU $idx: $name" >> "$tmp_file"
        echo "  Memory: ${mem_used}MB / ${mem_total}MB" >> "$tmp_file"
        echo "  Utilization: ${util}%" >> "$tmp_file"
        echo "" >> "$tmp_file"
    done

    tui_textbox "GPU Usage" "$tmp_file"
}

# System Containers
system_containers() {
    local tmp_file=$(create_temp)

    echo "Running vLLM Containers" > "$tmp_file"
    echo "=======================" >> "$tmp_file"
    echo "" >> "$tmp_file"

    # Get all containers that might be vLLM related
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null >> "$tmp_file"

    tui_textbox "Running Containers" "$tmp_file"
}

# System Profiles Status
system_profiles_status() {
    local tmp_file=$(create_temp)

    echo "All Profiles Status" > "$tmp_file"
    echo "===================" >> "$tmp_file"
    echo "" >> "$tmp_file"
    printf "%-15s %-10s %-8s %-6s %s\n" "PROFILE" "STATUS" "PORT" "GPU" "MODEL" >> "$tmp_file"
    echo "--------------------------------------------------------------" >> "$tmp_file"

    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local name=$(basename "$profile" .env)
            local container=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
            local gpu=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
            local config=$(grep "^CONFIG_NAME=" "$profile" | cut -d'=' -f2)

            local status="stopped"
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                status="running"
            fi

            printf "%-15s %-10s %-8s %-6s %s\n" "$name" "$status" "$port" "$gpu" "$config" >> "$tmp_file"
        fi
    done

    tui_textbox "Profiles Status" "$tmp_file"
}

# Interactive mode entry point
run_interactive() {
    if ! check_tui_tool; then
        exit 1
    fi
    show_main_menu
}

#=============================================================================
# CLI FUNCTIONS (Original)
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

check_conflict() {
    local profile=$1
    local gpu_id=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
    local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
    local container_name=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)

    # Check for port conflict (exclude own container)
    local port_conflict=$(docker ps --format '{{.Names}}:{{.Ports}}' 2>/dev/null | grep -v "^${container_name}:" | grep ":$port->")
    if [[ -n "$port_conflict" ]]; then
        local conflict_container=$(echo "$port_conflict" | cut -d':' -f1)
        echo -e "${RED}Error: Port $port is already in use by container '$conflict_container'${NC}"
        return 1
    fi

    # Check for GPU conflict (warning only - models can share a GPU with lower memory)
    for other_profile in "$PROFILES_DIR"/*.env; do
        [[ ! -f "$other_profile" ]] && continue
        [[ "$other_profile" == "$profile" ]] && continue

        local other_container=$(grep "^CONTAINER_NAME=" "$other_profile" | cut -d'=' -f2)
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${other_container}$"; then
            local other_gpu=$(grep "^GPU_ID=" "$other_profile" | cut -d'=' -f2)
            local gid ogid
            for gid in $(echo "$gpu_id" | tr ',' '\n'); do
                for ogid in $(echo "$other_gpu" | tr ',' '\n'); do
                    if [[ "$gid" == "$ogid" ]]; then
                        echo -e "${YELLOW}Warning: GPU $gid is also used by running container '$other_container'${NC}"
                    fi
                done
            done
        fi
    done

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
    local repo_url="https://github.com/vllm-project/vllm.git"
    local branch="main"
    local use_official=false
    local custom_tag=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --repo)
                repo_url="$2"
                shift 2
                ;;
            --official)
                use_official=true
                shift
                ;;
            --tag)
                custom_tag="$2"
                shift 2
                ;;
            -*)
                echo -e "${RED}Error: Unknown option '$1'${NC}"
                echo "Usage: ./run.sh build [branch] [--repo URL] [--official] [--tag TAG]"
                return 1
                ;;
            *)
                branch="$1"
                shift
                ;;
        esac
    done

    if [[ "$use_official" == "true" ]]; then
        run_build_official "$repo_url" "$branch" "$custom_tag"
    else
        run_build_fast "$repo_url" "$branch" "$custom_tag"
    fi
}

# Clone or update vLLM repository
# After calling this function, caller should cd to $SCRIPT_DIR/.vllm-src for build
clone_or_update_vllm() {
    local repo_url=${1:-https://github.com/vllm-project/vllm.git}
    local branch=${2:-main}
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    if [[ -d "$vllm_src_dir/.git" ]]; then
        echo -e "${BLUE}Updating existing vLLM source...${NC}"
        pushd "$vllm_src_dir" > /dev/null

        # Check if remote URL matches
        local current_remote=$(git remote get-url origin)
        if [[ "$current_remote" != "$repo_url" ]]; then
            echo -e "${YELLOW}Repository URL changed. Re-cloning...${NC}"
            popd > /dev/null
            rm -rf "$vllm_src_dir"
            git clone "$repo_url" "$vllm_src_dir"
            pushd "$vllm_src_dir" > /dev/null
            git checkout "$branch"
        else
            git fetch origin
            git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"
            git pull origin "$branch" 2>/dev/null || true
        fi

        local hash=$(git rev-parse --short HEAD)
        popd > /dev/null
    else
        echo -e "${BLUE}Cloning vLLM repository...${NC}"
        rm -rf "$vllm_src_dir"
        git clone "$repo_url" "$vllm_src_dir"
        pushd "$vllm_src_dir" > /dev/null
        git checkout "$branch"
        local hash=$(git rev-parse --short HEAD)
        popd > /dev/null
    fi

    echo "$hash"
}

# Fast local build - uses official Dockerfile with YOUR GPU only
run_build_fast() {
    local repo_url=${1:-https://github.com/vllm-project/vllm.git}
    local branch=${2:-main}
    local custom_tag=${3:-}
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

    echo -e "${BLUE}Building vLLM from source${NC}"
    echo -e "${YELLOW}Repository: $repo_url${NC}"
    echo -e "${YELLOW}Branch: $branch${NC}"
    echo -e "${GREEN}Detected GPU: $gpu_name (sm_$gpu_arch)${NC}"
    echo -e "${GREEN}Building for your GPU only - MUCH faster!${NC}"
    echo -e "${YELLOW}Tag: vllm-dev:$main_tag${NC}"
    echo ""

    local commit_hash=$(clone_or_update_vllm "$repo_url" "$branch")
    local build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo ""
    echo -e "${BLUE}Building with official Dockerfile (GPU: $gpu_arch only)...${NC}"
    echo ""

    # Build using official Dockerfile with local GPU arch only + labels
    docker build \
        -f "$vllm_src_dir/docker/Dockerfile" \
        --build-arg torch_cuda_arch_list="$gpu_arch" \
        --build-arg RUN_WHEEL_CHECK=false \
        --target vllm-openai \
        --label "vllm.repo.url=$repo_url" \
        --label "vllm.repo.branch=$branch" \
        --label "vllm.commit.hash=$commit_hash" \
        --label "vllm.build.date=$build_date" \
        --label "vllm.build.type=fast" \
        -t "vllm-dev:$main_tag" \
        -t "vllm-dev:$branch" \
        "$vllm_src_dir"

    local result=$?

    if [[ $result -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Build completed successfully!${NC}"
        echo "Images tagged as:"
        echo "  - vllm-dev:$main_tag"
        echo "  - vllm-dev:$branch (latest for this branch)"
        echo ""
        echo "Build info:"
        echo "  - Repository: $repo_url"
        echo "  - Branch: $branch"
        echo "  - Commit: $commit_hash"
        echo "  - Build date: $build_date"
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
    local repo_url=${1:-https://github.com/vllm-project/vllm.git}
    local branch=${2:-main}
    local custom_tag=${3:-}
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    # Generate tags
    local date_tag="${branch}-$(date +%Y%m%d)"
    local main_tag="${custom_tag:-$date_tag}"

    echo -e "${BLUE}Building vLLM from source${NC}"
    echo -e "${YELLOW}Repository: $repo_url${NC}"
    echo -e "${YELLOW}Branch: $branch${NC}"
    echo -e "${YELLOW}Using official vLLM Dockerfile (ALL architectures)${NC}"
    echo -e "${YELLOW}This may take several HOURS on first build.${NC}"
    echo -e "${YELLOW}Tag: vllm-dev:$main_tag${NC}"
    echo ""

    local commit_hash=$(clone_or_update_vllm "$repo_url" "$branch")
    local build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo ""
    echo -e "${BLUE}Building Docker image using official Dockerfile...${NC}"
    echo ""

    # Build using official Dockerfile (all architectures) + labels
    # Skip wheel size check for official builds (builds for all GPUs are large)
    docker build \
        -f "$vllm_src_dir/docker/Dockerfile" \
        --build-arg RUN_WHEEL_CHECK=false \
        --target vllm-openai \
        --label "vllm.repo.url=$repo_url" \
        --label "vllm.repo.branch=$branch" \
        --label "vllm.commit.hash=$commit_hash" \
        --label "vllm.build.date=$build_date" \
        --label "vllm.build.type=official" \
        -t "vllm-dev:$main_tag" \
        -t "vllm-dev:$branch" \
        "$vllm_src_dir"

    local result=$?

    if [[ $result -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Build completed successfully!${NC}"
        echo "Images tagged as:"
        echo "  - vllm-dev:$main_tag"
        echo "  - vllm-dev:$branch (latest for this branch)"
        echo ""
        echo "Build info:"
        echo "  - Repository: $repo_url"
        echo "  - Branch: $branch"
        echo "  - Commit: $commit_hash"
        echo "  - Build date: $build_date"
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
    local version_override=${5:-}
    local should_pull=${6:-"auto"}

    # Check for conflicts
    if ! check_conflict "$profile_path"; then
        return 1
    fi

    # Build LoRA options and export as environment variable
    export LORA_OPTIONS=$(build_lora_options "$profile_path")

    # Apply version override if specified (does not modify .env.common)
    if [[ -n "$version_override" ]]; then
        export VLLM_VERSION="$version_override"
    fi

    cd "$SCRIPT_DIR"

    # Export profile path for container env injection via overrides
    export PROFILE_PATH="$profile_path"

    local extra_packages=$(grep "^EXTRA_PIP_PACKAGES=" "$profile_path" | cut -d'=' -f2)
    if [[ -n "$extra_packages" ]]; then
        echo -e "${BLUE}Extra pip packages:${NC} $extra_packages"
    fi

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

        echo -e "${BLUE}Using image:${NC} vllm-dev:$image_tag"
        export VLLM_DEV_TAG="$image_tag"
        docker compose -f docker-compose.dev.yaml -f docker-compose.overrides.yaml --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d
    else
        local vllm_version
        if [[ -n "$version_override" ]]; then
            vllm_version="$version_override"
        else
            # Fallback: use latest local image
            vllm_version=$(get_local_latest_image --tag-only)
            if [[ "$vllm_version" == "none" ]]; then
                echo -e "${RED}Error: No local vllm/vllm-openai images found.${NC}"
                echo -e "${YELLOW}Pull an image first: docker pull vllm/vllm-openai:latest${NC}"
                return 1
            fi
        fi
        local config_name=$(grep "^CONFIG_NAME=" "$profile_path" | cut -d'=' -f2)

        echo -e "${GREEN}Starting $profile_name...${NC}"

        # Determine pull behavior
        local pull_opt=""
        if [[ "$should_pull" == "true" ]]; then
            pull_opt="--pull always"
        elif [[ "$should_pull" == "auto" ]]; then
            # Fallback for CLI/direct calls: pull if nightly or latest
            if [[ "$vllm_version" == "nightly" || "$vllm_version" == "latest" ]]; then
                pull_opt="--pull always"
            fi
        fi
        # should_pull == "false": no pull (use local image as-is)

        echo -e "${BLUE}Using image:${NC} vllm/vllm-openai:$vllm_version"
        docker compose -f docker-compose.yaml -f docker-compose.overrides.yaml --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d $pull_opt
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

    # Determine which compose file was used based on container image
    local image=$(docker inspect "$container_name" --format='{{.Config.Image}}' 2>/dev/null)
    local compose_file="docker-compose.yaml"

    if [[ "$image" == vllm-dev:* ]]; then
        compose_file="docker-compose.dev.yaml"
        export VLLM_DEV_TAG="${image#vllm-dev:}"
    fi

    cd "$SCRIPT_DIR"

    export PROFILE_PATH="$profile_path"

    # Use docker compose down for proper cleanup (networks, etc.)
    if docker compose -f "$compose_file" -f docker-compose.overrides.yaml --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" down 2>/dev/null; then
        echo -e "${GREEN}$profile_name stopped successfully!${NC}"
    else
        # Fallback to direct docker stop/rm if compose down fails
        echo -e "${YELLOW}Compose down failed, falling back to direct stop...${NC}"
        docker stop "$container_name" 2>/dev/null
        if docker rm "$container_name" 2>/dev/null; then
            echo -e "${GREEN}$profile_name stopped successfully!${NC}"
        else
            echo -e "${RED}Failed to remove container $container_name${NC}"
            return 1
        fi
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
    # Collect container names from all profiles
    local found=false
    local names=()
    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local container=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                names+=("$container")
                found=true
            fi
        fi
    done

    if [[ "$found" == "true" ]]; then
        printf "%-20s %-40s %-20s %s\n" "NAME" "IMAGE" "STATUS" "PORTS"
        for name in "${names[@]}"; do
            docker ps --filter "name=^${name}$" --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" | \
                while IFS=$'\t' read -r n i s p; do
                    printf "%-20s %-40s %-20s %s\n" "$n" "$i" "$s" "$p"
                done
        done
    else
        echo "(No running containers)"
    fi
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
        echo "           or: ./run.sh build [branch] --repo <repo-url>"
        return 0
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    docker images vllm-dev --format "{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedAt}}" | \
    while IFS=$'\t' read -r tag image_id size created; do
        local created_short=$(echo "$created" | cut -d' ' -f1,2)

        # Get label info
        local repo_url=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.repo.url"}}' 2>/dev/null)
        local branch=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.repo.branch"}}' 2>/dev/null)
        local commit=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.commit.hash"}}' 2>/dev/null)
        local build_date=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.build.date"}}' 2>/dev/null)
        local build_type=$(docker inspect "vllm-dev:$tag" --format='{{index .Config.Labels "vllm.build.type"}}' 2>/dev/null)

        echo -e "${GREEN}Tag:${NC} vllm-dev:$tag"
        echo "  Size: $size | Created: $created_short"

        if [[ -n "$repo_url" && "$repo_url" != "<no value>" ]]; then
            # Extract repo name from URL
            local repo_name=$(echo "$repo_url" | sed 's|https://github.com/||' | sed 's|\.git||')
            echo "  Repository: $repo_name"
            echo "  Branch: $branch | Commit: $commit"
            if [[ -n "$build_date" && "$build_date" != "<no value>" ]]; then
                local build_date_short=$(echo "$build_date" | cut -d'T' -f1)
                echo "  Built: $build_date_short | Type: $build_type"
            fi
        else
            echo -e "  ${YELLOW}(Legacy build - no metadata)${NC}"
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    done

    echo ""
    echo "Usage: ./run.sh <profile> up --dev --tag <TAG>"
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
