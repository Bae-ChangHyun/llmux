#!/bin/bash
# Version management functions for vLLM images and containers

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
