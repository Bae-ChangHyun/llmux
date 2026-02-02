#!/bin/bash
# TUI (Text User Interface) helper functions

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
        local created_date=$(echo "$created" | cut -d' ' -f1)
        local branch=$(echo "$tag" | sed 's/-[0-9]\{8\}$//')
        items="$items $tag \"Branch:$branch Size:$size Date:$created_date\""
    done < <(docker images vllm-dev --format "{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null)
    echo "$items"
}

# Get running container list for menu
get_running_container_menu_items() {
    local items=""
    for profile in "$PROFILES_DIR"/*.env; do
        if [[ -f "$profile" ]]; then
            local name=$(basename "$profile" .env)
            local container=$(grep "^CONTAINER_NAME=" "$profile" | cut -d'=' -f2)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
                local port=$(grep "^VLLM_PORT=" "$profile" | cut -d'=' -f2)
                local gpu=$(grep "^GPU_ID=" "$profile" | cut -d'=' -f2)
                items="$items $name \"Running - GPU:$gpu Port:$port\""
            fi
        fi
    done
    echo "$items"
}
