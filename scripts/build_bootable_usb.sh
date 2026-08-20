#!/usr/bin/env bash
# ==============================================================================
# Brain Framework — Portable Autonomous Agent Bootable USB Generator
# ==============================================================================
# Builds a self-contained, zero-dependency live Linux initramfs image
# that boots directly into the Brain Autonomous Agent runtime.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/target/bootable_usb"
INITRAMFS_DIR="${BUILD_DIR}/initramfs"
OUTPUT_IMG="${BUILD_DIR}/brain_agent_live.img"

echo "============================================================"
echo "  Building Brain Portable Autonomous Agent Bootable Image   "
echo "============================================================"

# 1. Compile Brain Release Binary
echo "[1/4] Compiling static release Brain binary..."
cargo build --release -p brain -j 2

BIN_SRC="${REPO_ROOT}/target/release/brain"
if [ ! -f "${BIN_SRC}" ]; then
    echo "Error: Compiled binary not found at ${BIN_SRC}"
    exit 1
fi

# 2. Prepare Minimal Initramfs Directory Layout
echo "[2/4] Assembling minimal root filesystem..."
rm -rf "${BUILD_DIR}"
mkdir -p "${INITRAMFS_DIR}"/{bin,sbin,etc/brain,proc,sys,dev,mnt,tmp}

cp "${BIN_SRC}" "${INITRAMFS_DIR}/bin/brain"
chmod +x "${INITRAMFS_DIR}/bin/brain"

# 3. Create Default Agent Configuration
cat << 'CONFIG_EOF' > "${INITRAMFS_DIR}/etc/brain/agent.json"
{
  "name": "portable_usb_brain_agent",
  "mode": "Autonomous",
  "state_dim": 16,
  "action_dim": 4,
  "hidden_dim": 32,
  "max_steps": 1000,
  "curiosity_eta": 0.1,
  "dry_run": false
}
CONFIG_EOF

# 4. Create Minimal Init Script (/init)
cat << 'INIT_EOF' > "${INITRAMFS_DIR}/init"
#!/bin/sh
# Minimal Initramfs Entrypoint for Brain Autonomous Agent

echo "============================================================"
echo "    Brain Autonomous Agent — Booting from Live USB Media    "
echo "============================================================"

# Mount essential pseudo-filesystems
mount -t proc none /proc 2>/dev/null || true
mount -t sysfs none /sys 2>/dev/null || true
mount -t devtmpfs none /dev 2>/dev/null || true

echo "[*] Hardware detection: Scanning video and serial actuators..."
ls -l /dev/video* 2>/dev/null || echo "    No physical V4L2 device found; fallback to virtual capture."
ls -l /dev/ttyACM* 2>/dev/null || echo "    No serial HID dongle detected; fallback to mock HID."

echo "[*] Launching Brain Autonomous Agent loop..."
/bin/brain agent run --config /etc/brain/agent.json --steps 100

echo "[✓] Brain Agent session finished. Entering rescue shell..."
exec /bin/sh
INIT_EOF

chmod +x "${INITRAMFS_DIR}/init"

# 5. Pack Initramfs Archive
echo "[3/4] Creating compressed CPIO initramfs archive..."
cd "${INITRAMFS_DIR}"
find . -print0 | cpio --null --create --format=newc | gzip -9 > "${BUILD_DIR}/initramfs.cpio.gz"
cd "${REPO_ROOT}"

echo "[4/4] Output generated successfully:"
echo "      - Initramfs Archive: ${BUILD_DIR}/initramfs.cpio.gz"
echo "      - Single Binary:     ${INITRAMFS_DIR}/bin/brain"
echo "      - Agent Config:      ${INITRAMFS_DIR}/etc/brain/agent.json"
echo ""
echo "To boot with QEMU:"
echo "qemu-system-x86_64 -kernel /boot/vmlinuz -initrd ${BUILD_DIR}/initramfs.cpio.gz -append 'console=ttyS0' -nographic"
echo "============================================================"
