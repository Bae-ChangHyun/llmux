#!/bin/bash
# Container operation functions (up, down, logs, status, etc.)

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

    cd "$SCRIPT_DIR"

    # Ensure network exists (external: true in docker-compose.yaml)
    docker network create vllm-network 2>/dev/null || true

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
        local -a pull_args=()
        if [[ "$should_pull" == "true" ]]; then
            pull_args=(--pull always)
        elif [[ "$should_pull" == "auto" ]]; then
            # Fallback for CLI/direct calls: pull if nightly or latest
            if [[ "$vllm_version" == "nightly" || "$vllm_version" == "latest" ]]; then
                pull_args=(--pull always)
            fi
        fi
        # should_pull == "false": no pull (use local image as-is)

        echo -e "${BLUE}Using image:${NC} vllm/vllm-openai:$vllm_version"
        export VLLM_VERSION="$vllm_version"
        docker compose -f docker-compose.yaml -f docker-compose.overrides.yaml --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d "${pull_args[@]}"
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

    # Export LORA_OPTIONS to suppress docker compose warning
    export LORA_OPTIONS=$(build_lora_options "$profile_path")

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
