#!/bin/bash
# Input validation functions

# Validate port number (1024-65535)
validate_port() {
    local port=$1
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Port must be a number${NC}"
        return 1
    fi
    if [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
        echo -e "${RED}Error: Port must be between 1024 and 65535${NC}"
        return 1
    fi
    return 0
}

# Validate GPU ID (single or comma-separated numbers, supports multi-digit IDs)
validate_gpu_id() {
    local gpu=$1
    if ! [[ "$gpu" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo -e "${RED}Error: GPU ID must be a number or comma-separated numbers (e.g., 0 or 0,1)${NC}"
        return 1
    fi
    return 0
}

# Validate GPU memory utilization (0.0-1.0)
validate_gpu_memory() {
    local mem=$1
    if ! [[ "$mem" =~ ^0?\.[0-9]+$ || "$mem" =~ ^1(\.0+)?$ ]]; then
        echo -e "${RED}Error: GPU memory utilization must be between 0.1 and 1.0${NC}"
        return 1
    fi
    return 0
}

# Validate profile/config name (alphanumeric, dash, underscore only)
validate_name() {
    local name=$1
    if ! [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo -e "${RED}Error: Name must contain only letters, numbers, dash, and underscore${NC}"
        return 1
    fi
    return 0
}

# Sanitize string for sed replacement (escape special characters)
sanitize_for_sed() {
    local str=$1
    local delimiter=${2:-/}
    # Escape backslash first, then the delimiter and &
    str=$(echo "$str" | sed 's/\\/\\\\/g')
    if [[ "$delimiter" == "/" ]]; then
        echo "$str" | sed 's/[/&]/\\&/g'
    elif [[ "$delimiter" == "|" ]]; then
        echo "$str" | sed 's/[|&]/\\&/g'
    else
        echo "$str" | sed "s/[${delimiter}&]/\\\\&/g"
    fi
}
