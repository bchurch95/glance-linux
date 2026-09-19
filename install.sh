#!/usr/bin/env bash
# ==============================================================================
# Glance Linux - Automated Setup & Installer Script
# Fork: https://github.com/bchurch95/glance-linux
# Supports: Acer 3D IR Depth Cameras, Intel Lunar Lake NPU, Omarchy / Hyprland
# ==============================================================================
set -euo pipefail

# ANSI color codes
BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
BLUE="\033[34m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
plugin_dir="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins"
plugin_id="io.github.ayandexyz.glance"
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}/glance"
bin_dir="$HOME/.local/bin"

log_info()    { echo -e "${BLUE}${BOLD}[*]${RESET} $*"; }
log_success() { echo -e "${GREEN}${BOLD}[✓]${RESET} $*"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[!]${RESET} $*"; }
log_err()     { echo -e "${RED}${BOLD}[✗]${RESET} $*" >&2; }

usage() {
  cat <<EOF
Usage: ./install.sh [OPTIONS]

Options:
  --no-gui        Do not install PySide6 GUI dependencies
  --no-plugin     Skip Omarchy desktop bar plugin setup
  --no-pam        Skip wiring into lock screen PAM
  --uninstall     Uninstall glancectl, systemd service, and Omarchy plugin
  -h, --help      Show this help message
EOF
  exit 0
}

want_gui=1
want_plugin=1
want_pam=1
uninstall=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gui)    want_gui=0; shift ;;
    --no-plugin) want_plugin=0; shift ;;
    --no-pam)    want_pam=0; shift ;;
    --uninstall) uninstall=1; shift ;;
    -h|--help)   usage ;;
    *) log_err "Unknown option: $1"; usage ;;
  esac
done

# ------------------------------------------------------------------------------
# UNINSTALLATION
# ------------------------------------------------------------------------------
if (( uninstall )); then
  log_info "Uninstalling Glance..."
  systemctl --user disable --now glanced.service 2>/dev/null || true
  rm -f "$unit_dir/glanced.service"
  systemctl --user daemon-reload

  rm -f "$plugin_dir/$plugin_id"
  rm -f "$bin_dir/glancectl"

  if [[ -f /usr/lib/security/pam_glance.so ]]; then
    log_info "Removing /usr/lib/security/pam_glance.so (requires sudo)..."
    sudo rm -f /usr/lib/security/pam_glance.so
  fi

  if [[ -x "$repo/.venv/bin/glancectl" ]]; then
    "$repo/.venv/bin/glancectl" setup-pam --remove 2>/dev/null || true
    "$repo/.venv/bin/glancectl" setup-lock --remove 2>/dev/null || true
  fi

  log_success "Uninstallation complete. (Enrollment store in $data_dir was preserved)."
  exit 0
fi

# ------------------------------------------------------------------------------
# BANNER
# ------------------------------------------------------------------------------
echo -e "${BOLD}${BLUE}=====================================================${RESET}"
echo -e "${BOLD}${BLUE}  Glance Linux - Setup & Installation${RESET}"
echo -e "${DIM}  Fork: bchurch95/glance-linux${RESET}"
echo -e "${DIM}  Hardware: Acer 3D IR Depth + Intel Lunar Lake NPU${RESET}"
echo -e "${BOLD}${BLUE}=====================================================${RESET}"
echo

# ------------------------------------------------------------------------------
# 1. PREREQUISITES CHECK
# ------------------------------------------------------------------------------
log_info "Checking system prerequisites..."
missing_pkgs=()

for cmd in python3 gcc make git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing_pkgs+=("$cmd")
  fi
done

if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  log_err "Missing required build tools: ${missing_pkgs[*]}"
  if command -v pacman >/dev/null 2>&1; then
    echo -e "Install them with: ${BOLD}sudo pacman -S ${missing_pkgs[*]}${RESET}"
  fi
  exit 1
fi

# ------------------------------------------------------------------------------
# 2. BUILD & INSTALL PAM MODULE
# ------------------------------------------------------------------------------
log_info "Building pam_glance.so PAM module..."
make -C "$repo/pam"

log_info "Installing pam_glance.so to /usr/lib/security/ (requires sudo)..."
sudo make -C "$repo/pam" install
log_success "pam_glance.so installed successfully."

# ------------------------------------------------------------------------------
# 3. PYTHON VIRTUAL ENVIRONMENT & DEPENDENCIES
# ------------------------------------------------------------------------------
log_info "Setting up Python virtual environment in .venv..."
if [[ ! -d "$repo/.venv" ]]; then
  python3 -m venv "$repo/.venv"
fi

"$repo/.venv/bin/pip" install --upgrade pip --quiet

log_info "Installing Glance Python package and NPU/AI dependencies..."
extras="runtime,dev"
if (( want_gui )); then
  extras="runtime,gui,dev"
fi

"$repo/.venv/bin/pip" install -e "$repo[$extras]" --quiet
log_success "Dependencies installed."

glancectl="$repo/.venv/bin/glancectl"

# ------------------------------------------------------------------------------
# 4. DOWNLOAD AI MODELS
# ------------------------------------------------------------------------------
log_info "Fetching face landmarker and ArcFace recognition models..."
"$glancectl" fetch-model
log_success "AI models downloaded."

# ------------------------------------------------------------------------------
# 5. USER BINARY & SYSTEMD SERVICE
# ------------------------------------------------------------------------------
mkdir -p "$unit_dir" "$data_dir" "$bin_dir"
chmod 700 "$data_dir"

if [[ ! -L "$bin_dir/glancectl" || "$(readlink "$bin_dir/glancectl")" != "$glancectl" ]]; then
  ln -sfn "$glancectl" "$bin_dir/glancectl"
  log_success "Linked $bin_dir/glancectl -> $glancectl"
fi

# Ensure ~/.local/bin is in PATH for future shells
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
  log_warn "$bin_dir is not currently in your PATH. Add it to your ~/.bashrc or ~/.zshrc."
fi

log_info "Configuring glanced user systemd service..."
sed "s|^ExecStart=.*|ExecStart=$glancectl daemon --mode light|" \
  "$repo/packaging/systemd/glanced.service" > "$unit_dir/glanced.service"

systemctl --user daemon-reload
systemctl --user enable --now glanced.service
log_success "glanced.service started and enabled in user systemd."

# ------------------------------------------------------------------------------
# 6. OMARCHY PLUGIN & LOCK SCREEN INTEGRATION
# ------------------------------------------------------------------------------
if (( want_plugin )) && [[ -d "${XDG_CONFIG_HOME:-$HOME/.config}/omarchy" || -d "/usr/share/omarchy" ]]; then
  log_info "Configuring Omarchy desktop integration..."
  mkdir -p "$plugin_dir"
  ln -sfn "$repo/plugin" "$plugin_dir/$plugin_id"
  log_success "Linked plugin: $plugin_dir/$plugin_id -> $repo/plugin"

  if command -v omarchy-shell >/dev/null 2>&1; then
    omarchy-shell shell rescanPlugins 2>/dev/null || true
    if command -v omarchy >/dev/null 2>&1; then
      omarchy plugin enable "$plugin_id" 2>/dev/null || true
      log_success "Omarchy plugin enabled."
    fi
  fi

  # Ensure glancectlPath is configured in shell.json so the widget finds the venv binary
  shell_json="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/shell.json"
  if [[ -f "$shell_json" ]]; then
    python3 -c '
import json, sys
path, ctl = sys.argv[1], sys.argv[2]
try:
    with open(path, "r") as f:
        data = json.load(f)
    changed = False
    for section in ("left", "center", "right"):
        for item in data.get("bar", {}).get(section, []):
            if item.get("id") == "io.github.ayandexyz.glance" and item.get("glancectlPath") != ctl:
                item["glancectlPath"] = ctl
                changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
except Exception:
    pass
' "$shell_json" "$glancectl" 2>/dev/null || true
  fi

  # Apply Face ID indicator patch on Omarchy lock screen
  log_info "Installing Face ID lock screen indicator (requires sudo)..."
  "$glancectl" setup-lock || log_warn "Could not auto-apply lock indicator; run 'glancectl setup-lock' later."

  if command -v omarchy-restart-shell >/dev/null 2>&1; then
    omarchy-restart-shell 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 7. PAM WIRING
# ------------------------------------------------------------------------------
if (( want_pam )); then
  log_info "Wiring face unlock into lock screen PAM (requires sudo)..."
  "$glancectl" setup-pam || log_warn "Could not auto-wire PAM; run 'glancectl setup-pam' later."
  log_success "Lock screen PAM wired."
fi

# ------------------------------------------------------------------------------
# 8. HARDWARE IR EMITTER CHECK
# ------------------------------------------------------------------------------
echo
log_info "Checking Acer 3D IR Depth camera hardware..."
if [[ -e /dev/video2 ]]; then
  echo -e "  ${GREEN}Found paired depth/IR camera device at /dev/video2${RESET}"
  if ! command -v linux-enable-ir-emitter >/dev/null 2>&1; then
    echo -e "  ${YELLOW}Notice: On Acer laptops, the hardware IR LED emitter requires 'linux-enable-ir-emitter'.${RESET}"
    echo -e "  To enable active infrared night illumination, run:"
    echo -e "    ${BOLD}yay -S linux-enable-ir-emitter-bin${RESET}"
    echo -e "    ${BOLD}sudo linux-enable-ir-emitter -d /dev/video2 configure -g${RESET}"
    echo -e "    ${BOLD}sudo systemctl enable --now linux-enable-ir-emitter.service${RESET}"
  else
    echo -e "  ${GREEN}linux-enable-ir-emitter is installed.${RESET}"
  fi
fi

# ------------------------------------------------------------------------------
# 9. ENROLLMENT PROMPT
# ------------------------------------------------------------------------------
echo
log_success "Glance setup finished successfully!"
echo
if "$glancectl" status | grep -q "enrolled:   True"; then
  echo -e "${GREEN}Face identity is already enrolled!${RESET}"
  echo -e "Test authentication now with: ${BOLD}glancectl authenticate${RESET}"
else
  echo -e "${YELLOW}No face enrolled yet.${RESET}"
  read -r -p "Would you like to enroll your face now? [Y/n] " answer
  answer=${answer:-Y}
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    if (( want_gui )); then
      "$glancectl" enroll --gui --name "$USER" --remember
    else
      "$glancectl" enroll --name "$USER" --remember
    fi
  else
    echo -e "You can enroll anytime using: ${BOLD}glancectl enroll --gui --name \"$USER\" --remember${RESET}"
  fi
fi

echo
echo -e "${BOLD}${GREEN}All set! You can lock your screen (Super + L) and press Enter to unlock with your face.${RESET}"
