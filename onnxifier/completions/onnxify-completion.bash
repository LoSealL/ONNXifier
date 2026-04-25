# shellcheck shell=bash
# bash completion for onnxify
# Source this file in your shell to enable tab completion for the onnxify CLI:
#   source /path/to/onnxify-completion.bash
#
# To make it permanent, add the above line to your ~/.bashrc

_onnxify_get_passes() {
    local cache_file="${TMPDIR:-/tmp}/.onnxify_passes_$(whoami).cache"
    local python_cmd="${ONNXIFY_PYTHON:-python3}"

    if [[ ! -f "$cache_file" ]] || [[ -n $(find "$cache_file" -mmin +60 2>/dev/null) ]]; then
        "$python_cmd" -c "from onnxifier.passes import PASSES; print('\n'.join(PASSES))" >"$cache_file" 2>/dev/null || true
    fi

    if [[ -f "$cache_file" ]]; then
        cat "$cache_file"
    fi
}

_onnxify_complete() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD - 1]}"

    # Look backward to find the most recent option flag.
    # This handles space-separated multi-pass inputs like:
    #   onnxify model.onnx -a infer_shape fold_const<TAB>
    local _last_opt=""
    local i
    for ((i = COMP_CWORD - 1; i >= 0; i--)); do
        local w="${COMP_WORDS[$i]}"
        if [[ "$w" == -* ]]; then
            _last_opt="$w"
            break
        fi
    done

    case "$_last_opt" in
    -a | --activate | -r | --remove)
        # If the current word looks like a new option (and is not empty),
        # fall through to option completion instead of pass completion.
        if [[ "$cur" != -* || "$cur" == "" ]]; then
            local prefix="$cur"
            local base=""

            # Handle comma-separated passes like "fuse_gelu,fuse_mish"
            if [[ "$cur" == *,* ]]; then
                base="${cur%,*},"
                prefix="${cur##*,}"
            fi

            local passes
            passes=$(_onnxify_get_passes)

            local matches=()
            local p
            for p in $passes; do
                if [[ "$p" == "$prefix"* ]]; then
                    matches+=("$base$p")
                fi
            done

            if [[ ${#matches[@]} -gt 0 ]]; then
                COMPREPLY=("${matches[@]}")
            fi
            return 0
        fi
        ;;
    esac

    case "$prev" in
    --format)
        COMPREPLY=($(compgen -W "protobuf textproto json onnxtxt" -- "$cur"))
        return 0
        ;;
    --checker-backend)
        COMPREPLY=($(compgen -W "onnx openvino onnxruntime" -- "$cur"))
        return 0
        ;;
    -vv | --log-level)
        COMPREPLY=($(compgen -W "DEBUG INFO WARNING ERROR CRITICAL" -- "$cur"))
        return 0
        ;;
    --print)
        local passes
        passes=$(_onnxify_get_passes)
        local matches=()
        local p
        for p in all l1 l2 l3 $passes; do
            if [[ "$p" == "$cur"* ]]; then
                matches+=("$p")
            fi
        done
        if [[ ${#matches[@]} -gt 0 ]]; then
            COMPREPLY=("${matches[@]}")
        fi
        return 0
        ;;
    esac

    # Complete options
    if [[ "$cur" == -* ]]; then
        local opts="-a --activate -r --remove -n --no-passes --print --format -s --infer-shapes -c --config-file -u --uncheck --check -d --dry-run --checker-backend -v --opset-version -vv --log-level -R --recursive --nodes -h --help"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return 0
    fi

    # Default to file completion
    COMPREPLY=($(compgen -f -- "$cur"))
}

complete -o default -F _onnxify_complete onnxify
