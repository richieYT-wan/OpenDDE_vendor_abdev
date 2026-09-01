#!/usr/bin/bash
#
# Unified installation script for the forked Protenix repository (LOCAL usage).
#
# Usage:
#   ./install.sh              # Full install
#   ./install.sh --no-db      # Skip colabfold DB
#   ./install.sh -h | --help  # Show usage
#

set -eo pipefail

# --- Argument Parsing ---
SKIP_DB=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --no-db       Skip downloading the large ColabFold database (colabfold_db.tar).
                The VHH search database is still downloaded.
  -h, --help    Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-db)  SKIP_DB=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --- Configuration ---
ENV_NAME="opendde"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export OPENDDE_ROOT_DIR="${REPO_DIR}"

# FIXED: Anchor all runtime dirs to REPO_DIR (matches Protenix's $PWD-relative lookups).
# Users running `python scripts/run.py ...` from the repo root will find files here.
INSTALL_PREFIX="$HOME/opendde_bin"          # binaries can stay in $HOME (they're on PATH)
CHECKPOINT_DIR="${REPO_DIR}/checkpoint"
COMMON_DIR="${REPO_DIR}/common"
DB_DIR="${REPO_DIR}/search_database"
GCS_BUCKET="gs://em52-ab-develop-analytics-prod-f684/data/denovo-design/snakemake/data/01_raw/opendde-inputs"
MMSEQS_VERSION="16-747c6"
MMSEQS_BIN_PATH="${INSTALL_PREFIX}/mmseqs2/bin/mmseqs"

echo "=== OpenDDE Unified Installer (local) ==="
echo "  Repo Dir:        $REPO_DIR"
echo "  Conda Env:       $ENV_NAME"
echo "  Install Prefix:  $INSTALL_PREFIX"
echo "  Checkpoint Dir:  $CHECKPOINT_DIR"
echo "  Common Dir:      $COMMON_DIR"
echo "  Database Dir:    $DB_DIR"
echo "  GCS Bucket:      $GCS_BUCKET"
echo "  Skip DB:         $([[ $SKIP_DB -eq 1 ]] && echo yes || echo no)"
echo "==========================================="

# --- Helpers: portable GCS sync/copy (defined early so all steps can use them) ---
gcs_sync() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    if command -v gcloud >/dev/null 2>&1; then
        echo ">>> Using 'gcloud storage rsync' for $src -> $dst"
        gcloud storage rsync "$src" "$dst" --recursive
        return $?
    fi
    if command -v gsutil >/dev/null 2>&1; then
        echo ">>> Using 'gsutil -m rsync' for $src -> $dst"
        gsutil -m rsync -r "$src" "$dst"
        return $?
    fi
    echo ">>> gcloud/gsutil not found, falling back to Python (google.cloud.storage)"
    python - "$src" "$dst" <<'PYEOF'
import os, sys
from urllib.parse import urlparse
from google.cloud import storage

src, dst = sys.argv[1], sys.argv[2]
if not src.endswith("/"): src += "/"
parsed = urlparse(src)
bucket_name, prefix = parsed.netloc, parsed.path.lstrip("/")
print(f"Syncing gs://{bucket_name}/{prefix} -> {dst}")

client = storage.Client()
bucket = client.bucket(bucket_name)
count = 0
for blob in bucket.list_blobs(prefix=prefix):
    if blob.name.endswith("/"): continue
    rel = os.path.relpath(blob.name, prefix)
    dest = os.path.join(dst, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) == blob.size:
        continue
    blob.download_to_filename(dest)
    count += 1
    if count % 25 == 0:
        print(f"  ... {count} files")
print(f"Done. {count} file(s) synced.")
PYEOF
}

gcs_cp() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    if command -v gcloud >/dev/null 2>&1; then
        gcloud storage cp "$src" "$dst"; return $?
    fi
    if command -v gsutil >/dev/null 2>&1; then
        gsutil -m cp "$src" "$dst"; return $?
    fi
    python - "$src" "$dst" <<'PYEOF'
import sys
from urllib.parse import urlparse
from google.cloud import storage

src, dst = sys.argv[1], sys.argv[2]
parsed = urlparse(src)
client = storage.Client()
bucket = client.bucket(parsed.netloc)
blob = bucket.blob(parsed.path.lstrip("/"))
blob.download_to_filename(dst)
print(f"Downloaded {src} -> {dst}")
PYEOF
}

# --- Step 1: Conda Environment Setup ---
echo "=== Step 1: Setting up Conda environment: '$ENV_NAME' ==="
CONDA_CMD="conda"

if "$CONDA_CMD" env list | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda environment from 'workflow/envs/protenix_env.yaml'..."
    "$CONDA_CMD" env create --quiet -f ./workflow/envs/opendde_env.yaml
    echo "Environment created."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# From conda set the env variable
conda env config vars set OPENDDE_ROOT_DIR="${REPO_DIR}"

echo "Python version: $(python --version)"
echo "Conda prefix: $CONDA_PREFIX"

# --- Step 2: Install OpenDDE Package ---
# Necessary due to addition of IPSAE in inference code
echo "=== Step 2: Installing OpenDDE (non-editable) ==="
uv pip install --torch-backend cu126 "opendde[gpu]"

# --- Step 3: HMMER ---
echo "=== Step 3: Verifying HMMER installation ==="
if command -v hmmsearch &>/dev/null; then
    echo "HMMER found at: $(command -v hmmsearch)"
else
    echo "WARNING: HMMER not found. It should have been installed by conda."
    echo "Manual install: conda install -c bioconda hmmer"
fi

# --- Step 4: MMseqs2 (from source, only if not already available) ---
echo "=== Step 4: Ensuring MMseqs2 is available ==="

if command -v mmseqs &>/dev/null; then
    echo "MMseqs2 already available at: $(command -v mmseqs)"
    mmseqs version
else
    echo "MMseqs2 not found. Building from source v${MMSEQS_VERSION}..."
    if [[ -f "$MMSEQS_BIN_PATH" ]]; then
        echo "Pre-built binary found at $MMSEQS_BIN_PATH; skipping build."
    else
        mkdir -p "$INSTALL_PREFIX"
        cd "$INSTALL_PREFIX"
        wget -q "https://github.com/soedinglab/MMseqs2/archive/refs/tags/${MMSEQS_VERSION}.tar.gz"
        tar xzf "${MMSEQS_VERSION}.tar.gz"
        cd "MMseqs2-${MMSEQS_VERSION}"
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}/mmseqs2" ..
        make -j"$(nproc)"
        make install
        cd "$REPO_DIR"
    fi

    # Symlink into conda env's bin
    if [[ -e "${CONDA_PREFIX}/bin/mmseqs" || -L "${CONDA_PREFIX}/bin/mmseqs" ]]; then
        rm -f "${CONDA_PREFIX}/bin/mmseqs"
    fi
    ln -sf "$MMSEQS_BIN_PATH" "${CONDA_PREFIX}/bin/mmseqs"
    echo "MMseqs2 symlinked at: ${CONDA_PREFIX}/bin/mmseqs"
    mmseqs version
fi

# --- Step 5: Download Weights, Common, Databases ---
echo "=== Step 5: Downloading model weights, common data, and search databases ==="

echo ">>> Model checkpoints -> ${CHECKPOINT_DIR}"
gcs_sync "${GCS_BUCKET}/checkpoint/" "${CHECKPOINT_DIR}"
ls -1 "$CHECKPOINT_DIR"

echo ">>> Common data files -> ${COMMON_DIR}"
gcs_sync "${GCS_BUCKET}/common/" "${COMMON_DIR}"
ls -1 "$COMMON_DIR"

echo ">>> Search databases -> ${DB_DIR}"
gcs_sync "${GCS_BUCKET}/search_database/" "${DB_DIR}"
ls -1R "$DB_DIR"

# --- Verify critical files ---
echo ">>> Verifying critical files"
REQUIRED=(
  "$CHECKPOINT_DIR/opendde_abag.pt"
  "$COMMON_DIR/components.cif"
  "$COMMON_DIR/components.cif.rdkit_mol.pkl"
  "$DB_DIR/pdb_seqres_2022_09_28.fasta"
)
missing=0
for f in "${REQUIRED[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  OK  $f ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
        missing=$((missing + 1))
    fi
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing critical file(s) missing. Aborting."
    exit 1
fi

# --- Colabfold DB (optional) ---
COLABFOLD_TAR_REMOTE="${GCS_BUCKET}/colabfold_db.tar"
COLABFOLD_TAR_LOCAL="${DB_DIR}/colabfold_db.tar"

if [[ $SKIP_DB -eq 1 ]]; then
    echo "--no-db flag set: skipping ColabFold database download."
else
    if [[ -f "$COLABFOLD_TAR_LOCAL" ]]; then
        echo "ColabFold database already exists at ${COLABFOLD_TAR_LOCAL}."
    else
        echo "Downloading ColabFold database (this may take a while)..."
        gcs_cp "$COLABFOLD_TAR_REMOTE" "$COLABFOLD_TAR_LOCAL"
        echo "Extracting ColabFold database..."
        tar -xvf "$COLABFOLD_TAR_LOCAL" -C "$DB_DIR"
    fi
fi

# --- Finalization ---
echo ""
echo "=== Running smoke test: opendde pred --help ==="
opendde pred --help | head

echo ""
echo "=========================================="
echo "OpenDDE setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Paths:"
echo "  - Checkpoint:      $CHECKPOINT_DIR"     # FIXED: was $WEIGHTS_DIR
echo "  - Common data:     $COMMON_DIR"
echo "  - Search DB:       $DB_DIR"
echo "  - MMseqs2 Binary:  $(command -v mmseqs)"
echo ""
echo "IMPORTANT: The runtime data (checkpoint/common/search_database) lives"
echo "in the repo directory ($REPO_DIR). Always run 'python scripts/run.py ...'"
echo "from the repo root so Protenix finds these paths."
echo ""
echo "Verify with a sample run:"
echo "  mkdir -p ./data/"
echo "  gcloud storage cp -r ${GCS_BUCKET}/00_sample ./data/"
echo "  python scripts/run.py \\"
echo "    -c ./data/00_sample/example_run_config.yaml \\"
echo "    -i ./data/00_sample/sample_sequences.csv \\"
echo "    -o ./data/03_predictions/ -r 0 2"
echo ""