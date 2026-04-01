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

profile_env_get() {
    local key=$1
    local profile_path=$2
    grep "^${key}=" "$profile_path" | cut -d'=' -f2-
}

profile_env_set() {
    local key=$1
    local value=$2
    local profile_path=$3

    if grep -q "^${key}=" "$profile_path"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$profile_path"
    else
        printf '\n%s=%s\n' "$key" "$value" >> "$profile_path"
    fi
}

check_common_env() {
    local profile_path=$1

    if [[ ! -f "$COMMON_ENV" ]]; then
        echo -e "${RED}Error: .env.common not found.${NC}"
        echo -e "${YELLOW}Create it from .env.common.example before starting containers.${NC}"
        return 1
    fi

    local hf_cache_path=$(grep "^HF_CACHE_PATH=" "$COMMON_ENV" | cut -d'=' -f2-)
    if [[ -z "$hf_cache_path" ]]; then
        echo -e "${RED}Error: HF_CACHE_PATH is not set in .env.common${NC}"
        return 1
    fi
    if [[ "$hf_cache_path" != /* ]]; then
        echo -e "${RED}Error: HF_CACHE_PATH must be an absolute path. Current value: $hf_cache_path${NC}"
        return 1
    fi

    if is_lora_enabled "$profile_path"; then
        local lora_base_path=$(grep "^LORA_BASE_PATH=" "$COMMON_ENV" | cut -d'=' -f2-)
        if [[ -z "$lora_base_path" ]]; then
            echo -e "${RED}Error: ENABLE_LORA=true but LORA_BASE_PATH is not set in .env.common${NC}"
            return 1
        fi
        if [[ "$lora_base_path" != /* ]]; then
            echo -e "${RED}Error: LORA_BASE_PATH must be an absolute path. Current value: $lora_base_path${NC}"
            return 1
        fi
    fi

    return 0
}

is_lora_enabled() {
    local profile_path=$1
    local enable_lora=$(profile_env_get "ENABLE_LORA" "$profile_path")
    [[ "$enable_lora" == "true" ]]
}

ensure_profile_config() {
    local profile_path=$1
    local profile_name=$2
    local config_name=$(profile_env_get "CONFIG_NAME" "$profile_path")
    local model_id=$(profile_env_get "MODEL_ID" "$profile_path")

    if [[ -z "$config_name" ]]; then
        config_name="$profile_name"
        profile_env_set "CONFIG_NAME" "$config_name" "$profile_path"
        echo -e "${YELLOW}No config linked for '$profile_name'. Auto-linked default config '${config_name}'.${NC}"
    fi

    local config_path="$SCRIPT_DIR/config/${config_name}.yaml"
    if [[ -f "$config_path" ]]; then
        export CONFIG_NAME="$config_name"
        return 0
    fi

    mkdir -p "$SCRIPT_DIR/config"

    if [[ -n "$model_id" ]]; then
        cat > "$config_path" << EOF
model: $model_id
gpu-memory-utilization: 0.55
EOF
        echo -e "${YELLOW}Created default config: config/${config_name}.yaml${NC}"
        export CONFIG_NAME="$config_name"
        return 0
    fi

    cat > "$config_path" << EOF
# Auto-generated default config for profile: $profile_name
# Set a valid Hugging Face model ID below, then start again.
model: your-org/your-model
gpu-memory-utilization: 0.55
EOF
    echo -e "${RED}Created config/${config_name}.yaml but MODEL_ID is not set for profile '$profile_name'.${NC}"
    echo -e "${YELLOW}Edit the config model field or set MODEL_ID in profiles/${profile_name}.env, then start again.${NC}"
    export CONFIG_NAME="$config_name"
    return 1
}

get_compose_files() {
    local profile_path=$1
    local use_dev=$2
    local compose_files=()

    if [[ "$use_dev" == "true" ]]; then
        compose_files+=("-f" "docker-compose.dev.yaml")
    else
        compose_files+=("-f" "docker-compose.yaml")
    fi

    if is_lora_enabled "$profile_path"; then
        compose_files+=("-f" "docker-compose.lora.yaml")
    fi

    compose_files+=("-f" "docker-compose.overrides.yaml")
    printf '%s\n' "${compose_files[@]}"
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

    if ! check_common_env "$profile_path"; then
        return 1
    fi

    if ! ensure_profile_config "$profile_path" "$profile_name"; then
        return 1
    fi

    # Export profile path for container env injection via overrides
    export PROFILE_PATH="$profile_path"
    mapfile -t compose_files < <(get_compose_files "$profile_path" "$use_dev") || return 1

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
        docker compose "${compose_files[@]}" --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d
    else
        local vllm_version
        if [[ -n "$version_override" ]]; then
            # Resolve "latest" to actual version tag (e.g., v0.18.0)
            if [[ "$version_override" == "latest" ]]; then
                local resolved=$(get_latest_release_version)
                if [[ -n "$resolved" && "$resolved" != "unknown" ]]; then
                    vllm_version="$resolved"
                else
                    vllm_version="$version_override"
                fi
            else
                vllm_version="$version_override"
            fi
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
        docker compose "${compose_files[@]}" --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" up -d "${pull_args[@]}"
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
    local use_dev="false"

    if [[ "$image" == vllm-dev:* ]]; then
        use_dev="true"
        export VLLM_DEV_TAG="${image#vllm-dev:}"
    fi

    cd "$SCRIPT_DIR"

    export PROFILE_PATH="$profile_path"
    mapfile -t compose_files < <(get_compose_files "$profile_path" "$use_dev") || return 1

    # Use docker compose down for proper cleanup (networks, etc.)
    if docker compose "${compose_files[@]}" --env-file "$COMMON_ENV" --env-file "$profile_path" -p "$profile_name" down 2>/dev/null; then
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
