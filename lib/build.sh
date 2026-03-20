#!/bin/bash
# Build-related functions for vLLM from-source builds

# Detect GPU compute capability
detect_gpu_arch() {
    local arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$arch" ]]; then
        echo ""
        return 1
    fi
    echo "$arch"
}

# Clone or update vLLM repository
# After calling this function, caller should cd to $SCRIPT_DIR/.vllm-src for build
clone_or_update_vllm() {
    local repo_url=${1:-https://github.com/vllm-project/vllm.git}
    local branch=${2:-main}
    local vllm_src_dir="$SCRIPT_DIR/.vllm-src"

    # Guard against empty path in rm -rf
    if [[ -z "$vllm_src_dir" || "$vllm_src_dir" == "/" ]]; then
        echo -e "${RED}Error: Invalid vLLM source directory${NC}"
        return 1
    fi

    if [[ -d "$vllm_src_dir/.git" ]]; then
        echo -e "${BLUE}Updating existing vLLM source...${NC}"

        # Check if remote URL matches
        local current_remote=$(cd "$vllm_src_dir" && git remote get-url origin)
        if [[ "$current_remote" != "$repo_url" ]]; then
            echo -e "${YELLOW}Repository URL changed. Re-cloning...${NC}"
            rm -rf "$vllm_src_dir"
            git clone "$repo_url" "$vllm_src_dir"
            (cd "$vllm_src_dir" && git checkout "$branch")
        else
            (cd "$vllm_src_dir" && git fetch origin && { git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"; } && git pull origin "$branch" 2>/dev/null || true)
        fi

        local hash=$(cd "$vllm_src_dir" && git rev-parse --short HEAD)
    else
        echo -e "${BLUE}Cloning vLLM repository...${NC}"
        rm -rf "$vllm_src_dir"
        git clone "$repo_url" "$vllm_src_dir"
        (cd "$vllm_src_dir" && git checkout "$branch")
        local hash=$(cd "$vllm_src_dir" && git rev-parse --short HEAD)
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

# CLI entry point for build command
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
