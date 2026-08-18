#!/usr/bin/env bash

cwam_require_trusted_executable() {
    local executable="$1"
    local trusted_root="$2"
    local resolved_executable
    local resolved_root

    [[ -d "$trusted_root" && ! -L "$trusted_root" ]] || {
        printf 'ERROR: trusted executable root is not a real directory: %s\n' "$trusted_root" >&2
        return 1
    }
    [[ -x "$executable" ]] || {
        printf 'ERROR: executable is missing, broken, or not executable: %s\n' "$executable" >&2
        return 1
    }
    resolved_root="$(realpath -e -- "$trusted_root")" || return 1
    resolved_executable="$(realpath -e -- "$executable")" || return 1
    [[ -f "$resolved_executable" && ! -L "$resolved_executable" ]] || {
        printf 'ERROR: executable target is not a regular file: %s\n' "$executable" >&2
        return 1
    }
    case "$resolved_executable" in
        "$resolved_root"/*) ;;
        *)
            printf 'ERROR: executable target escapes trusted root: %s -> %s\n' \
                "$executable" "$resolved_executable" >&2
            return 1
            ;;
    esac
}
