#!/usr/bin/env bash
set -euo pipefail
fail() { echo "controller_manager clock overlay: $*" >&2; exit 1; }
tool_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$tool_dir/.." && pwd)
workspace_dir=$(cd -- "$repo_dir/../.." && pwd)
lock="$tool_dir/ros2_control/controller_manager_clock.lock"
patch="$tool_dir/ros2_control/controller_manager_clock.patch"
source "$lock"
set +u
source /opt/ros/jazzy/setup.bash
set -u
installed_deb=$(dpkg-query -W -f='${Version}' ros-jazzy-controller-manager) || fail 'ros-jazzy-controller-manager is not installed'
[[ "$installed_deb" == "$INSTALLED_DEB_VERSION" ]] || fail "expected package $INSTALLED_DEB_VERSION, found $installed_deb"
[[ "$(sha256sum "$patch" | cut -d' ' -f1)" == "$PATCH_SHA256" ]] || fail 'tracked patch checksum mismatch'
install_prefix="$workspace_dir/install/controller_manager"
marker="$install_prefix/.controller_manager_clock.lock"
active_prefix=$(ros2 pkg prefix controller_manager) || fail 'controller_manager is not discoverable'
if [[ "$active_prefix" == "$install_prefix" ]]; then
  [[ -f "$marker" ]] && cmp -s "$lock" "$marker" || fail 'workspace overlay is not this pinned build'
elif [[ "$active_prefix" != /opt/ros/jazzy ]]; then
  fail "refusing foreign controller_manager prefix: $active_prefix"
fi
if [[ -e "$install_prefix" ]]; then
  [[ -f "$marker" ]] && cmp -s "$lock" "$marker" || fail 'refusing to replace an unpinned workspace prefix'
  if [[ -f "$install_prefix/share/controller_manager/package.xml" ]]; then
    grep -q "<version>$UPSTREAM_VERSION</version>" "$install_prefix/share/controller_manager/package.xml" || fail 'workspace package version differs from the pin'
  fi
else
  mkdir -p "$install_prefix"
  cp -- "$lock" "$marker"
fi
source_root="$workspace_dir/.cache/ros2_control_controller_manager_$UPSTREAM_VERSION"
if [[ ! -d "$source_root/.git" ]]; then
  [[ ! -e "$source_root" ]] || fail "refusing non-git cache path: $source_root"
  git clone --depth 1 --filter=blob:none --sparse --branch "$UPSTREAM_TAG" \
    "$UPSTREAM_REPO" "$source_root"
  git -C "$source_root" sparse-checkout set controller_manager
fi
[[ "$(git -C "$source_root" remote get-url origin)" == "$UPSTREAM_REPO" ]] ||
  fail 'cached source has a different origin'
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] ||
  fail 'cached source is not the pinned upstream commit'
package_version=$(sed -n 's/.*<version>\([^<]*\)<\/version>.*/\1/p' \
  "$source_root/controller_manager/package.xml" | head -n1)
[[ "$package_version" == "$UPSTREAM_VERSION" ]] || fail 'upstream package.xml version differs from the pin'
build_cache="$workspace_dir/build/controller_manager/CMakeCache.txt"
if [[ -e "$workspace_dir/build/controller_manager" ]]; then grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$source_root/controller_manager" "$build_cache" && grep -Fxq 'CMAKE_GENERATOR:INTERNAL=Ninja' "$build_cache" || fail 'refusing a foreign or non-Ninja controller_manager build cache'; fi
if git -C "$source_root" apply --unidiff-zero --reverse --check "$patch" >/dev/null 2>&1; then
  [[ "$(git -C "$source_root" diff --name-only)" == controller_manager/src/controller_manager.cpp ]] ||
    fail 'cached source has extra tracked edits'
  [[ -z "$(git -C "$source_root" ls-files --others --exclude-standard)" ]] ||
    fail 'cached source has untracked files'
  git -C "$source_root" diff --binary --unified=0 -- controller_manager/src/controller_manager.cpp |
    cmp -s - "$patch" || fail 'cached production patch differs from the pinned patch'
else
  [[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] ||
    fail 'cached source is dirty and the patch is not already applied'
  git -C "$source_root" apply --unidiff-zero --check "$patch"
  git -C "$source_root" apply --unidiff-zero "$patch"
fi

(
  cd -- "$source_root/controller_manager"
  COLCON_DEFAULTS_FILE=/dev/null colcon \
    --log-base "$workspace_dir/log/controller_manager_clock_overlay" build \
    --base-paths "$source_root/controller_manager" --packages-select controller_manager \
    --build-base "$workspace_dir/build" --install-base "$workspace_dir/install" \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DBUILD_TESTING=OFF
)
cmp -s "$lock" "$marker" || fail 'workspace pin marker changed during build'
grep -q "<version>$UPSTREAM_VERSION</version>" "$install_prefix/share/controller_manager/package.xml" ||
  fail 'installed package version differs from the pin'
set +u
source "$workspace_dir/install/setup.bash"
set -u
[[ "$(ros2 pkg prefix controller_manager)" == "$install_prefix" ]] ||
  fail 'workspace setup did not select the pinned overlay'
echo "Installed pinned controller_manager overlay at $install_prefix"
