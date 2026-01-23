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

# TUI Tool (whiptail or dialog)
TUI_TOOL=""

# Check for TUI tool availability
check_tui_tool() {
    if command -v whiptail &> /dev/null; then
        TUI_TOOL="whiptail"
        return 0
    elif command -v dialog &> /dev/null; then
        TUI_TOOL="dialog"
        return 0
    else
        echo -e "${RED}Error: Neither whiptail nor dialog is installed.${NC}"
        echo "Install with: apt-get install whiptail"
        return 1
    fi
}

# TUI Helper Functions
tui_menu() {
    local title="$1"
    local text="$2"
    shift 2
    $TUI_TOOL --title "$title" --menu "$text" 20 70 12 "$@" 3>&1 1>&2 2>&3
}

tui_checklist() {
    local title="$1"
    local text="$2"
    shift 2
    $TUI_TOOL --title "$title" --checklist "$text" 20 70 12 "$@" 3>&1 1>&2 2>&3
}

tui_inputbox() {
    local title="$1"
    local text="$2"
    local default="$3"
    $TUI_TOOL --title "$title" --inputbox "$text" 10 60 "$default" 3>&1 1>&2 2>&3
}

tui_yesno() {
    local title="$1"
    local text="$2"
    $TUI_TOOL --title "$title" --yesno "$text" 10 60
}

tui_msgbox() {
    local title="$1"
    local text="$2"
    $TUI_TOOL --title "$title" --msgbox "$text" 15 70
}

tui_textbox() {
    local title="$1"
    local file="$2"
    $TUI_TOOL --title "$title" --textbox "$file" 25 80
}

# Get profile list for menu
get_profile_menu_items() {
    local items=""
    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local name=$(basename "$profile" .env)
            local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
            local gpu=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
            local config=$(grep "^CONFIG_NAME=" "$profile" | cut -d'=' -f2)
            items="$items $name \"GPU:$gpu Port:$port Config:$config\""
        fi
    done
    echo "$items"
}

# Get config list for menu
get_config_menu_items() {
    local items=""
    for config in "$SCRIPT_DIR/config"/*.yaml; do
        if [[ -f "$config" ]]; then
            local name=$(basename "$config" .yaml)
            local model=$(grep "^model:" "$config" | cut -d':' -f2- | sed 's/^ *//')
            items="$items $name \"$model\""
        fi
    done
    echo "$items"
}

# Get image list for menu
get_image_menu_items() {
    local items=""
    while IFS=$'\t' read -r tag size created; do
        items="$items $tag \"Size:$size\""
    done < <(docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null)
    echo "$items"
}

# Get running container list for menu (matches with profiles)
get_running_container_menu_items() {
    local items=""
    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local name=$(basename "$profile" .env)
            local container=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            # Check if container is running
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
                local gpu=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
                items="$items $name \"Running - GPU:$gpu Port:$port\""
            fi
        fi
    done
    echo "$items"
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

    # Port
    local port=$(tui_inputbox "Port" "Enter vLLM port:" "8000")
    [[ -z "$port" ]] && return

    # GPU memory utilization
    local gpu_util=$(tui_inputbox "GPU Memory" "GPU memory utilization (0.0-1.0):" "0.9")
    [[ -z "$gpu_util" ]] && return

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

    # Create config file
    cat > "$SCRIPT_DIR/config/$safe_name.yaml" << EOF
model: $model
gpu-memory-utilization: $gpu_util
EOF

    # Create profile file
    cat > "$PROFILES_DIR/$safe_name.env" << EOF
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

    tui_msgbox "Success" "Created:\n- Config: config/$safe_name.yaml\n- Profile: profiles/$safe_name.env\n\nStart with: ./run.sh $safe_name up"
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
    # Select profile
    local items=$(get_profile_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No profiles found. Create a profile first."
        return
    fi

    local profile=$(eval "tui_menu \"Start Container\" \"Select profile:\" $items")
    [[ -z "$profile" ]] && return

    local profile_path="$PROFILES_DIR/$profile.env"

    # Select mode
    local mode=$(tui_menu "Start Mode" "Select start mode:" \
        "official" "Use official vLLM image" \
        "dev" "Use dev build (from source)")
    [[ -z "$mode" ]] && return

    local use_dev="false"
    local custom_tag=""

    if [[ "$mode" == "dev" ]]; then
        use_dev="true"

        # Check for available images
        local img_items=$(get_image_menu_items)
        if [[ -n "$img_items" ]]; then
            local tag_choice=$(tui_menu "Select Image" "Choose dev image tag:" \
                "latest" "Use latest branch tag" \
                $img_items)

            if [[ -n "$tag_choice" && "$tag_choice" != "latest" ]]; then
                custom_tag="$tag_choice"
            fi
        fi
    fi

    # Confirmation
    local msg="Profile: $profile\nMode: $mode"
    [[ -n "$custom_tag" ]] && msg="$msg\nTag: $custom_tag"

    if tui_yesno "Confirm Start" "$msg\n\nStart this container?"; then
        clear
        run_up "$profile_path" "$profile" "$use_dev" "$custom_tag"
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    fi
}

# Container Down Menu
container_down_menu() {
    local items=$(get_running_container_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Info" "No running containers found."
        return
    fi

    local profile=$(eval "tui_menu \"Stop Container\" \"Select container to stop:\" $items")
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
    local items=$(get_running_container_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Info" "No running containers found."
        return
    fi

    local profile=$(eval "tui_menu \"View Logs\" \"Select container:\" $items")
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
    local items=$(get_profile_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(eval "tui_menu \"Check Status\" \"Select profile:\" $items")
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

    # Select config
    local config_items=$(get_config_menu_items)
    if [[ -z "$config_items" ]]; then
        tui_msgbox "Error" "No configs found. Create a config first."
        return
    fi

    local config=$(eval "tui_menu \"Select Config\" \"Choose model config:\" $config_items")
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
    local items=$(get_profile_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(eval "tui_menu \"Edit Profile\" \"Select profile to edit:\" $items")
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
                    sed -i "s/^VLLM_PORT=.*/VLLM_PORT=$new_port/" "$profile_path"
                    tui_msgbox "Updated" "Port changed to $new_port"
                fi
                ;;
            2)
                local new_gpu=$(tui_inputbox "Change GPU" "Enter GPU ID(s):" "$current_gpu")
                if [[ -n "$new_gpu" ]]; then
                    sed -i "s/^GPU_ID=.*/GPU_ID=$new_gpu/" "$profile_path"
                    tui_msgbox "Updated" "GPU ID changed to $new_gpu"
                fi
                ;;
            3)
                local new_tp=$(tui_inputbox "Tensor Parallel" "Enter tensor parallel size:" "$current_tp")
                if [[ -n "$new_tp" ]]; then
                    sed -i "s/^TENSOR_PARALLEL_SIZE=.*/TENSOR_PARALLEL_SIZE=$new_tp/" "$profile_path"
                    tui_msgbox "Updated" "Tensor parallel changed to $new_tp"
                fi
                ;;
            4)
                local config_items=$(get_config_menu_items)
                local new_config=$(eval "tui_menu \"Select Config\" \"Choose config:\" $config_items")
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
    local items=$(get_profile_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(eval "tui_menu \"Delete Profile\" \"Select profile to delete:\" $items")
    [[ -z "$profile" ]] && return

    if tui_yesno "Confirm Delete" "Are you sure you want to delete profile '$profile'?"; then
        rm -f "$PROFILES_DIR/$profile.env"
        tui_msgbox "Deleted" "Profile '$profile' has been deleted."
    fi
}

# Profile View Menu
profile_view_menu() {
    local items=$(get_profile_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No profiles found."
        return
    fi

    local profile=$(eval "tui_menu \"View Profile\" \"Select profile:\" $items")
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

    if [[ -f "$SCRIPT_DIR/config/$name.yaml" ]]; then
        tui_msgbox "Error" "Config '$name' already exists."
        return
    fi

    local model=$(tui_inputbox "Model" "Enter model name (HuggingFace):" "")
    [[ -z "$model" ]] && return

    local gpu_util=$(tui_inputbox "GPU Memory" "GPU memory utilization (0.0-1.0):" "0.9")
    [[ -z "$gpu_util" ]] && return

    cat > "$SCRIPT_DIR/config/$name.yaml" << EOF
model: $model
gpu-memory-utilization: $gpu_util
EOF

    tui_msgbox "Success" "Config '$name' created successfully!"
}

# Config Edit Menu
config_edit_menu() {
    local items=$(get_config_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(eval "tui_menu \"Edit Config\" \"Select config:\" $items")
    [[ -z "$config" ]] && return

    local config_path="$SCRIPT_DIR/config/$config.yaml"

    while true; do
        local current_model=$(grep "^model:" "$config_path" | cut -d':' -f2- | sed 's/^ *//')
        local current_util=$(grep "^gpu-memory-utilization:" "$config_path" | cut -d':' -f2- | sed 's/^ *//')

        local choice=$(tui_menu "Edit: $config" "Model: $current_model" \
            "1" "Change Model (current: $current_model)" \
            "2" "Change GPU Utilization (current: $current_util)" \
            "3" "Add Custom Parameter" \
            "B" "Back")

        case "$choice" in
            1)
                local new_model=$(tui_inputbox "Model" "Enter model name:" "$current_model")
                if [[ -n "$new_model" ]]; then
                    sed -i "s|^model:.*|model: $new_model|" "$config_path"
                    tui_msgbox "Updated" "Model changed"
                fi
                ;;
            2)
                local new_util=$(tui_inputbox "GPU Utilization" "Enter value (0.0-1.0):" "$current_util")
                if [[ -n "$new_util" ]]; then
                    if grep -q "^gpu-memory-utilization:" "$config_path"; then
                        sed -i "s|^gpu-memory-utilization:.*|gpu-memory-utilization: $new_util|" "$config_path"
                    else
                        echo "gpu-memory-utilization: $new_util" >> "$config_path"
                    fi
                    tui_msgbox "Updated" "GPU utilization changed"
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
            B|"") break ;;
        esac
    done
}

# Config Delete Menu
config_delete_menu() {
    local items=$(get_config_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(eval "tui_menu \"Delete Config\" \"Select config:\" $items")
    [[ -z "$config" ]] && return

    if tui_yesno "Confirm Delete" "Are you sure you want to delete config '$config'?"; then
        rm -f "$SCRIPT_DIR/config/$config.yaml"
        tui_msgbox "Deleted" "Config '$config' has been deleted."
    fi
}

# Config View Menu
config_view_menu() {
    local items=$(get_config_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Error" "No configs found."
        return
    fi

    local config=$(eval "tui_menu \"View Config\" \"Select config:\" $items")
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

    local msg="Branch: $branch\nBuild type: $build_type"
    [[ -n "$custom_tag" ]] && msg="$msg\nTag: $custom_tag"

    if tui_yesno "Confirm Build" "$msg\n\nStart build? This may take a while."; then
        clear
        if [[ "$build_type" == "official" ]]; then
            run_build_official "$branch" "$custom_tag"
        else
            run_build_fast "$branch" "$custom_tag"
        fi
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    fi
}

# List Built Images
build_list_images() {
    local tmp_file=$(mktemp)

    echo "vLLM Development Images" > "$tmp_file"
    echo "========================" >> "$tmp_file"
    echo "" >> "$tmp_file"

    if ! docker images vllm-dev --format "{{.Tag}}" 2>/dev/null | grep -q .; then
        echo "No vllm-dev images found." >> "$tmp_file"
        echo "" >> "$tmp_file"
        echo "Build one with: ./run.sh build [branch]" >> "$tmp_file"
    else
        printf "%-25s %-12s %-20s\n" "TAG" "SIZE" "CREATED" >> "$tmp_file"
        echo "-----------------------------------------------------------" >> "$tmp_file"
        docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | \
        while IFS=$'\t' read -r tag size created; do
            printf "%-25s %-12s %-20s\n" "$tag" "$size" "$created" >> "$tmp_file"
        done
    fi

    tui_textbox "Built Images" "$tmp_file"
    rm -f "$tmp_file"
}

# Delete Image
build_delete_image() {
    local items=$(get_image_menu_items)
    if [[ -z "$items" ]]; then
        tui_msgbox "Info" "No vllm-dev images found."
        return
    fi

    local tag=$(eval "tui_menu \"Delete Image\" \"Select image to delete:\" $items")
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
            "B" "Back to Main Menu")

        case "$choice" in
            1) system_gpu_info ;;
            2) system_containers ;;
            3) system_profiles_status ;;
            B|"") break ;;
        esac
    done
}

# System GPU Info
system_gpu_info() {
    local tmp_file=$(mktemp)

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
    rm -f "$tmp_file"
}

# System Containers
system_containers() {
    local tmp_file=$(mktemp)

    echo "Running vLLM Containers" > "$tmp_file"
    echo "=======================" >> "$tmp_file"
    echo "" >> "$tmp_file"

    # Get all containers that might be vLLM related
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null >> "$tmp_file"

    tui_textbox "Running Containers" "$tmp_file"
    rm -f "$tmp_file"
}

# System Profiles Status
system_profiles_status() {
    local tmp_file=$(mktemp)

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
    rm -f "$tmp_file"
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
