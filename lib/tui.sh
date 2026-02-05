#!/bin/bash
# TUI (Text User Interface) helper functions

# TUI Tool (whiptail or dialog)
TUI_TOOL=""

# Global array populated by build_*_items() helper functions
declare -a MENU_ITEMS

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

# TUI wrapper functions
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

#=============================================================================
# Menu item builder helpers
# These populate the global MENU_ITEMS array to avoid code duplication.
# Usage: build_all_profile_items && tui_menu "Title" "Text" "${MENU_ITEMS[@]}"
#=============================================================================

# Build menu items for ALL profiles (name + GPU/Port/Config info)
build_all_profile_items() {
    MENU_ITEMS=()
    for profile_file in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile_file" ]]; then
            local name=$(basename "$profile_file" .env)
            local port=$(grep "^VLLM_PORT=" "$profile_file" | cut -d'=' -f2)
            local gpu=$(grep "^GPU_ID=" "$profile_file" | cut -d'=' -f2)
            local config=$(grep "^CONFIG_NAME=" "$profile_file" | cut -d'=' -f2)
            MENU_ITEMS+=("$name" "GPU:$gpu Port:$port Config:$config")
        fi
    done
}

# Build menu items for RUNNING containers only
build_running_profile_items() {
    MENU_ITEMS=()
    for profile_file in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile_file" ]]; then
            local name=$(basename "$profile_file" .env)
            local container=$(grep "^CONTAINER_NAME=" "$profile_file" | cut -d'=' -f2)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                local port=$(grep "^VLLM_PORT=" "$profile_file" | cut -d'=' -f2)
                local gpu=$(grep "^GPU_ID=" "$profile_file" | cut -d'=' -f2)
                MENU_ITEMS+=("$name" "Running - GPU:$gpu Port:$port")
            fi
        fi
    done
}

# Build menu items for config files
build_config_menu_items() {
    MENU_ITEMS=()
    for config_file in "$SCRIPT_DIR/config"/*.yaml; do
        if [[ -f "$config_file" ]]; then
            local name=$(basename "$config_file" .yaml)
            local model=$(grep "^model:" "$config_file" | cut -d':' -f2- | sed 's/^ *//')
            MENU_ITEMS+=("$name" "$model")
        fi
    done
}

# Build menu items for custom parameters in a config file
# Excludes model and gpu-memory-utilization (managed by dedicated options)
build_custom_param_items() {
    local config_path="$1"
    MENU_ITEMS=()
    while IFS=': ' read -r key value; do
        [[ -z "$key" ]] && continue
        MENU_ITEMS+=("$key" "$value")
    done < <(grep -v "^model:\|^gpu-memory-utilization:\|^#\|^$" "$config_path")
}

# Build menu items for dev images
build_dev_image_items() {
    MENU_ITEMS=()
    while IFS=$'\t' read -r tag size created; do
        local created_date=$(echo "$created" | cut -d' ' -f1)
        local branch=$(echo "$tag" | sed 's/-[0-9]\{8\}$//')
        MENU_ITEMS+=("$tag" "Branch:$branch Size:$size Date:$created_date")
    done < <(docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null)
}
