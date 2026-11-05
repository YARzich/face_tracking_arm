#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build an isolated, pinned Fast DDS 2.14.7 overlay with upstream PR #6559.
set -euo pipefail

fail() { echo "Fast DDS SHM overlay: $*" >&2; exit 1; }
tree_checksum() {
  (
    cd -- "$1"
    while IFS= read -r -d '' path; do
      [[ "$path" == "$2" ]] && continue
      if [[ -L "$path" ]]; then
        printf 'L %s -> %s\n' "$path" "$(readlink -- "$path")"
      elif [[ -f "$path" ]]; then
        printf 'F %s %s\n' "$path" "$(sha256sum "$path" | cut -d' ' -f1)"
      elif [[ -d "$path" ]]; then
        printf 'D %s\n' "$path"
      fi
    done < <(find . -mindepth 1 -print0 | LC_ALL=C sort -z)
  ) | sha256sum | cut -d' ' -f1
}
cache_option_matches() {
  local option_name=$1 expected=$2 existing
  existing=$(grep "^${option_name}:" "$build_root/CMakeCache.txt" || true)
  [[ -z "$existing" || "$existing" == *"=$expected" ]] ||
    fail "build cache option $option_name differs from the pin"
}
require_config_define() {
  local macro=$1 value=$2 config=$3
  grep -Eq "^#define ${macro} ${value}$" "$config" ||
    fail "generated config does not contain ${macro} ${value}"
}

tool_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$tool_dir/.." && pwd)
workspace_dir=$(cd -- "$repo_dir/../.." && pwd)
lock="$tool_dir/fastdds/fastdds_shm.lock"
patch="$tool_dir/fastdds/fastdds_shm.patch"
source "$lock"
set +u
source /opt/ros/jazzy/setup.bash
set -u

cache_dir="$workspace_dir/.cache/fastdds_shm_$UPSTREAM_VERSION"
source_root="$cache_dir/source"
build_root="$cache_dir/build"
install_root="$cache_dir/install"
deps_root="$cache_dir/deps"
asio_root="$deps_root/asio"
asio_archive="$deps_root/asio-$ASIO_COMMIT.tar.gz"
source_marker="$source_root/.fastdds_shm_source.lock"
asio_marker="$asio_root/.fastdds_shm_asio.lock"
build_marker="$build_root/.fastdds_shm_build.lock"
install_marker="$install_root/.fastdds_shm_install.lock"

[[ -f "$patch" ]] || fail "patch not found: $patch"
[[ "$(sha256sum "$patch" | cut -d' ' -f1)" == "$PATCH_SHA256" ]] ||
  fail 'tracked patch checksum mismatch'
command -v cmake >/dev/null || fail 'cmake is required'
command -v ninja >/dev/null || fail 'ninja is required'
command -v git >/dev/null || fail 'git is required'
command -v curl >/dev/null || fail 'curl is required'
command -v tar >/dev/null || fail 'tar is required'
command -v sha256sum >/dev/null || fail 'sha256sum is required'

mkdir -p -- "$cache_dir" "$deps_root"

if [[ ! -e "$source_root" ]]; then
  git clone --quiet --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_REPO" "$source_root"
  [[ "$(git -C "$source_root" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] ||
    fail 'cloned source is not the pinned release commit'
  printf '%s\n' "$UPSTREAM_TAG $UPSTREAM_COMMIT" > "$source_marker"
fi
[[ -d "$source_root/.git" && -f "$source_marker" ]] ||
  fail "refusing non-pinned or incomplete source path: $source_root"
[[ "$(cat "$source_marker")" == "$UPSTREAM_TAG $UPSTREAM_COMMIT" ]] ||
  fail 'source marker does not match the pinned release'
[[ "$(git -C "$source_root" remote get-url origin)" == "$UPSTREAM_REPO" ]] ||
  fail 'source cache origin differs from the pinned repository'
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] ||
  fail 'source HEAD is not the pinned upstream commit'
[[ -z "$(git -C "$source_root" ls-files --others --exclude-standard | grep -vxF .fastdds_shm_source.lock || true)" ]] ||
  fail 'source cache contains untracked files'
package_version=$(sed -n 's/^project(fastrtps VERSION "\([^"]*\)".*/\1/p' "$source_root/CMakeLists.txt" | head -n1)
[[ "$package_version" == "$UPSTREAM_VERSION" ]] || fail 'source CMake version differs from the pin'
source_asio_commit=$(git -C "$source_root" ls-tree HEAD thirdparty/asio | awk '{print $3}')
[[ "$source_asio_commit" == "$ASIO_COMMIT" ]] || fail 'Fast DDS tag has a different Asio submodule pin'

# Pin and verify the exact Asio submodule content separately from Fast DDS.
if [[ -e "$asio_archive" ]]; then
  [[ "$(sha256sum "$asio_archive" | cut -d' ' -f1)" == "$ASIO_ARCHIVE_SHA256" ]] ||
    fail "cached Asio archive checksum mismatch: $asio_archive"
else
  partial="$asio_archive.part.$$"
  trap 'rm -f -- "$partial"' EXIT
  curl --fail --location --silent --show-error "$ASIO_ARCHIVE_URL" -o "$partial" ||
    fail 'unable to download the pinned Asio archive'
  [[ "$(sha256sum "$partial" | cut -d' ' -f1)" == "$ASIO_ARCHIVE_SHA256" ]] ||
    fail 'downloaded Asio archive checksum mismatch'
  mv -- "$partial" "$asio_archive"
  trap - EXIT
fi
asio_marker_text="$ASIO_COMMIT $ASIO_VERSION $ASIO_ARCHIVE_SHA256 $ASIO_LICENSE $ASIO_TREE_SHA256"
if [[ ! -e "$asio_root" ]]; then
  asio_staging="$deps_root/asio.extract.$$"
  [[ ! -e "$asio_staging" ]] || fail "refusing existing extraction path: $asio_staging"
  mkdir -- "$asio_staging"
  tar -xzf "$asio_archive" -C "$asio_staging" --strip-components=1
  [[ -f "$asio_staging/asio/include/asio.hpp" && -f "$asio_staging/asio/include/asio/version.hpp" ]] ||
    fail 'pinned Asio archive has an unexpected include layout'
  mv -- "$asio_staging" "$asio_root"
  printf '%s\n' "$asio_marker_text" > "$asio_marker"
fi
[[ -f "$asio_marker" && "$(cat "$asio_marker")" == "$asio_marker_text" ]] ||
  fail 'Asio cache marker does not match the pin'
[[ "$(tree_checksum "$asio_root" './.fastdds_shm_asio.lock')" == "$ASIO_TREE_SHA256" ]] ||
  fail 'Asio source cache is dirty'
grep -Fq '#define ASIO_VERSION 103402' "$asio_root/asio/include/asio/version.hpp" ||
  fail "Asio version is not $ASIO_VERSION"

# Apply only the pinned two-file regression fix, idempotently.
if git -C "$source_root" apply --unidiff-zero --reverse --check "$patch" >/dev/null 2>&1; then
  [[ "$(git -C "$source_root" diff --name-only | LC_ALL=C sort)" == \
    $'src/cpp/rtps/transport/shared_mem/SharedMemGlobal.hpp\ntest/unittest/transport/SharedMemTests.cpp' ]] ||
    fail 'patched source has unexpected tracked edits'
  actual_diff=$(git -C "$source_root" diff --binary --unified=0 | sha256sum | cut -d' ' -f1)
  [[ "$actual_diff" == "$APPLIED_DIFF_SHA256" ]] || fail 'cached patch differs from the pinned diff'
else
  [[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all | grep -v '^?? .fastdds_shm_source.lock$' || true)" ]] ||
    fail 'source cache is dirty and the pinned patch is not already applied'
  git -C "$source_root" apply --unidiff-zero --check "$patch" || fail 'pinned patch does not apply'
  git -C "$source_root" apply --unidiff-zero "$patch"
  actual_diff=$(git -C "$source_root" diff --binary --unified=0 | sha256sum | cut -d' ' -f1)
  [[ "$actual_diff" == "$APPLIED_DIFF_SHA256" ]] || fail 'applied source diff differs from the pin'
fi

# Existing build and install caches must carry this exact lock and pinned identity.
if [[ -e "$build_root" ]]; then
  [[ -f "$build_marker" ]] && cmp -s "$lock" "$build_marker" ||
    fail "refusing foreign build cache: $build_root"
  [[ -f "$build_root/CMakeCache.txt" ]] || fail 'pinned build cache is incomplete'
  grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$source_root" "$build_root/CMakeCache.txt" ||
    fail 'build cache points to a different source tree'
  grep -Fxq 'CMAKE_GENERATOR:INTERNAL=Ninja' "$build_root/CMakeCache.txt" ||
    fail 'build cache uses a different generator'
  grep -Fxq 'CMAKE_BUILD_TYPE:STRING=RelWithDebInfo' "$build_root/CMakeCache.txt" ||
    fail 'build cache uses a different build type'
  grep -Fxq "CMAKE_INSTALL_PREFIX:PATH=$install_root" "$build_root/CMakeCache.txt" ||
    fail 'build cache uses a different install prefix'
  for option in BUILD_TESTING EPROSIMA_BUILD_TESTS COMPILE_TOOLS SECURITY SQLITE3_SUPPORT NO_TLS FASTDDS_STATISTICS LOG_NO_INFO ENABLE_OLD_LOG_MACROS STRICT_REALTIME; do
    case "$option" in
      BUILD_TESTING) expected=$BUILD_TESTING ;;
      EPROSIMA_BUILD_TESTS) expected=$BUILD_EPROSIMA_TESTS ;;
      COMPILE_TOOLS) expected=$BUILD_COMPILE_TOOLS ;;
      ENABLE_OLD_LOG_MACROS) expected=$BUILD_OLD_LOG_MACROS ;;
      *) var_name="BUILD_$option"; expected=${!var_name} ;;
    esac
    cache_option_matches "$option" "$expected"
  done
  cp -- "$lock" "$build_marker"
else
  mkdir -p -- "$build_root"
  cp -- "$lock" "$build_marker"
fi
if [[ -e "$install_root" ]]; then
  [[ -f "$install_marker" ]] && cmp -s "$lock" "$install_marker" ||
    fail "refusing foreign install prefix: $install_root"
else
  mkdir -p -- "$install_root"
  cp -- "$lock" "$install_marker"
fi

cmake -S "$source_root" -B "$build_root" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$install_root" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
  -DAsio_INCLUDE_DIR:PATH="$asio_root/asio/include" \
  -DSECURITY:BOOL="$BUILD_SECURITY" \
  -DSQLITE3_SUPPORT:BOOL="$BUILD_SQLITE3_SUPPORT" \
  -DNO_TLS:BOOL="$BUILD_NO_TLS" \
  -DFASTDDS_STATISTICS:BOOL="$BUILD_FASTDDS_STATISTICS" \
  -DLOG_NO_INFO:BOOL="$BUILD_LOG_NO_INFO" \
  -DENABLE_OLD_LOG_MACROS:BOOL="$BUILD_OLD_LOG_MACROS" \
  -DSTRICT_REALTIME:BOOL="$BUILD_STRICT_REALTIME" \
  -DBUILD_TESTING:BOOL="$BUILD_TESTING" \
  -DEPROSIMA_BUILD_TESTS:BOOL="$BUILD_EPROSIMA_TESTS" \
  -DCOMPILE_TOOLS:BOOL="$BUILD_COMPILE_TOOLS" \
  -DCOMPILE_EXAMPLES=OFF \
  -DBUILD_DOCUMENTATION=OFF
for option in BUILD_TESTING EPROSIMA_BUILD_TESTS COMPILE_TOOLS SECURITY SQLITE3_SUPPORT NO_TLS FASTDDS_STATISTICS LOG_NO_INFO ENABLE_OLD_LOG_MACROS STRICT_REALTIME; do
  case "$option" in
    BUILD_TESTING) expected=$BUILD_TESTING ;;
    EPROSIMA_BUILD_TESTS) expected=$BUILD_EPROSIMA_TESTS ;;
    COMPILE_TOOLS) expected=$BUILD_COMPILE_TOOLS ;;
    ENABLE_OLD_LOG_MACROS) expected=$BUILD_OLD_LOG_MACROS ;;
    *) var_name="BUILD_$option"; expected=${!var_name} ;;
  esac
  grep -Eq "^${option}:BOOL=${expected}$" "$build_root/CMakeCache.txt" ||
    fail "CMake did not preserve pinned option $option=$expected"
done
config="$build_root/include/fastrtps/config.h"
require_config_define HAVE_SECURITY 1 "$config"
require_config_define HAVE_SQLITE3 1 "$config"
require_config_define TLS_FOUND 1 "$config"
require_config_define HAVE_STRICT_REALTIME 0 "$config"
require_config_define HAVE_LOG_NO_INFO 1 "$config"
grep -Eq '^#define FASTDDS_STATISTICS$' "$config" || fail 'FASTDDS_STATISTICS is missing from generated config'
cmake --build "$build_root" --parallel 8 --target fastrtps
cmake --install "$build_root" --component libraries

cmp -s "$lock" "$build_marker" || fail 'build pin marker changed during build'
cmp -s "$lock" "$install_marker" || fail 'install pin marker changed during build'

# Deploy only the SONAME and its real target into the already-installed package.
package_prefix="$workspace_dir/install/face_tracking_arm"
[[ -d "$package_prefix" && -f "$package_prefix/share/face_tracking_arm/package.xml" ]] ||
  fail "face_tracking_arm must already be installed at $package_prefix"
source_lib_dir="$install_root/lib"
[[ -d "$source_lib_dir" ]] || source_lib_dir="$install_root/lib64"
soname='libfastrtps.so.2.14'
[[ -L "$source_lib_dir/$soname" ]] || fail "installed SONAME is missing: $source_lib_dir/$soname"
overlay_dir="$package_prefix/lib/fastdds_shm"
chain_names=()
current="$soname"
declare -A seen_names=()
while :; do
  [[ -z "${seen_names[$current]:-}" ]] || fail 'Fast DDS library symlink cycle detected'
  seen_names[$current]=1
  chain_names+=("$current")
  if [[ -L "$source_lib_dir/$current" ]]; then
    link_target=$(readlink -- "$source_lib_dir/$current")
    [[ "$link_target" != /* && "$link_target" != */* && "$link_target" != . && "$link_target" != .. ]] ||
      fail "unexpected SONAME link target: $current -> $link_target"
    current="$link_target"
  else
    [[ -f "$source_lib_dir/$current" ]] || fail "SONAME target is missing: $source_lib_dir/$current"
    break
  fi
done
binary_sha256=$(sha256sum "$source_lib_dir/$current" | cut -d' ' -f1)
manifest=$( {
  printf 'Fast DDS %s PR #%s merge %s\n' "$UPSTREAM_VERSION" "$FIX_PR" "$FIX_MERGE_COMMIT"
  printf 'asio_commit=%s\nasio_archive_sha256=%s\npatch_sha256=%s\nbinary_sha256=%s\n' \
    "$ASIO_COMMIT" "$ASIO_ARCHIVE_SHA256" "$PATCH_SHA256" "$binary_sha256"
  for name in "${chain_names[@]}"; do
    if [[ -L "$source_lib_dir/$name" ]]; then
      printf 'symlink %s -> %s\n' "$name" "$(readlink -- "$source_lib_dir/$name")"
    else
      printf 'file %s\n' "$name"
    fi
  done
} )
if [[ -e "$overlay_dir" ]]; then
  [[ -d "$overlay_dir" && -f "$overlay_dir/.fastdds_shm_overlay.lock" ]] ||
    fail "refusing unmarked overlay directory: $overlay_dir"
  [[ "$(cat "$overlay_dir/.fastdds_shm_overlay.lock")" == "$manifest" ]] ||
    fail 'existing deployed overlay has a different pin or library checksum'
  for entry in "$overlay_dir"/* "$overlay_dir"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name=${entry##*/}
    [[ "$name" == .fastdds_shm_overlay.lock ]] && continue
    allowed=false
    for expected in "${chain_names[@]}"; do [[ "$name" == "$expected" ]] && allowed=true; done
    [[ "$allowed" == true ]] || fail "foreign file in deployed overlay: $entry"
  done
  for name in "${chain_names[@]}"; do
    if [[ -L "$source_lib_dir/$name" ]]; then
      [[ -L "$overlay_dir/$name" && "$(readlink -- "$overlay_dir/$name")" == \
        "$(readlink -- "$source_lib_dir/$name")" ]] || fail "deployed symlink differs: $name"
    else
      [[ -f "$overlay_dir/$name" ]] && cmp -s "$source_lib_dir/$name" "$overlay_dir/$name" ||
        fail "deployed library differs: $name"
    fi
  done
else
  mkdir -p -- "$(dirname -- "$overlay_dir")"
  mkdir -- "$overlay_dir"
  for name in "${chain_names[@]}"; do cp -a -- "$source_lib_dir/$name" "$overlay_dir/$name"; done
  printf '%s\n' "$manifest" > "$overlay_dir/.fastdds_shm_overlay.lock"
fi
echo "Installed pinned Fast DDS $UPSTREAM_VERSION overlay at $install_root"
echo "Deployed $soname and its target chain into $overlay_dir"
