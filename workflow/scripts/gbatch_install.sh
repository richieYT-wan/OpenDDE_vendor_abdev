#!/usr/bin/bash
#
# Unified installation script for the forked OpenDDE repository.
# This script sets up the conda environment, installs the package,
# downloads required binaries and data, and sets up the necessary tools.
# Flags have been removed for GBatch and "skip-db" has now been hardcoded for now as colabfold support is not handleda for custom target MSA
#

set -eo pipefail

# --- Argument handling ---
PROJECT_DIR="${1:-$PWD}"
cd "$PROJECT_DIR"

# --- Configuration ---
# ALL paths are absolute, anchored to PROJECT_DIR, matching OpenDDE's runtime lookups.
ENV_NAME="opendde"
INSTALL_PREFIX="${PROJECT_DIR}/opendde_bin"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoint"
COMMON_DIR="${PROJECT_DIR}/common"
DB_DIR="${PROJECT_DIR}/search_database"
GCS_BUCKET="gs://em52-ab-develop-analytics-prod-f684/data/denovo-design/snakemake/data/01_raw/opendde-inputs"
SKIP_DB=1


# --- Helpers: portable GCS sync/copy ---
# Tries in order: gcloud storage → gsutil → python (google.cloud.storage)

gcs_sync() {
    # gcs_sync <gcs://src/> <local_dir>
    local src="$1"
    local dst="$2"
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
import os
import sys
from urllib.parse import urlparse

gcs_uri = sys.argv[1]
local_dir = sys.argv[2]

if not gcs_uri.endswith("/"):
    gcs_uri += "/"

parsed = urlparse(gcs_uri)
bucket_name = parsed.netloc
prefix = parsed.path.lstrip("/")

print(f"Syncing gs://{bucket_name}/{prefix} -> {local_dir}")

from google.cloud import storage
client = storage.Client()
bucket = client.bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)

count = 0
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    rel_path = os.path.relpath(blob.name, prefix)
    dest_path = os.path.join(local_dir, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Skip if same size (rough rsync-like check)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == blob.size:
        continue

    blob.download_to_filename(dest_path)
    count += 1
    if count % 25 == 0:
        print(f"  ... {count} files downloaded")

print(f"Done. {count} file(s) downloaded to {local_dir}.")
PYEOF
}

gcs_cp() {
    # gcs_cp <gcs://src/file> <local_file>
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"

    if command -v gcloud >/dev/null 2>&1; then
        echo ">>> Using 'gcloud storage cp' for $src -> $dst"
        gcloud storage cp "$src" "$dst"
        return $?
    fi

    if command -v gsutil >/dev/null 2>&1; then
        echo ">>> Using 'gsutil -m cp' for $src -> $dst"
        gsutil -m cp "$src" "$dst"
        return $?
    fi

    echo ">>> gcloud/gsutil not found, falling back to Python (google.cloud.storage)"
    python - "$src" "$dst" <<'PYEOF'
import sys
from urllib.parse import urlparse

gcs_uri = sys.argv[1]
local_path = sys.argv[2]

parsed = urlparse(gcs_uri)
bucket_name = parsed.netloc
blob_name = parsed.path.lstrip("/")

print(f"Downloading gs://{bucket_name}/{blob_name} -> {local_path}")

from google.cloud import storage
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename(local_path)
print("Done.")
PYEOF
}

echo "=== OpenDDE Unified Installer ==="
echo "  Repo Dir:       $PROJECT_DIR"
echo "  Conda Env:      $ENV_NAME"
echo "  Install Prefix: $INSTALL_PREFIX"
echo "  Weights Dir:    $CHECKPOINT_DIR"
echo "  Database Dir:   $DB_DIR"
echo "  GCS Bucket:     $GCS_BUCKET"
echo "  Skip DB:        $([[ $SKIP_DB -eq 1 ]] && echo yes || echo no)"
echo "=================================="

# --- Step 1: Conda Environment Setup ---
echo "=== Step 1: Setting up Conda environment: '$ENV_NAME' ==="

# The protenix conda environment should be activated by default through snakemake ; TBC
CONDA_CMD="conda"

echo "Conda environment activated."
echo "Python version: $(python --version)"
echo "Conda prefix: $CONDA_PREFIX"

# --- Step 2: Install OpenDDE Package ---
echo "=== Step 2: Installing OpenDDE (non-editable) ==="
pip install $PROJECT_DIR
echo "OpenDDE package installed."

# --- Step 3: Install HMMER ---
echo "=== Step 3: Verifying HMMER installation ==="
if command -v hmmsearch &> /dev/null; then
    echo "HMMER (hmmsearch) found at: $(command -v hmmsearch)"
else
    echo "WARNING: HMMER not found. It should have been installed by conda."
    echo "You may need to install it manually: conda install -c bioconda hmmer"
fi

# --- Step 4: Install MMseqs2 from Source ---
echo "=== Step 4: Checking MMseqs2 v${MMSEQS_VERSION} from conda==="
echo "MMseqs2 is available at: $(command -v mmseqs)"
"$(command -v mmseqs)" version

# --- Step 5: Download Model Weights and Databases ---
# this step needs fallback with gcloud storage and then python directly because gsutil is not necessarily available
echo "=== Step 5: Downloading model weights, common data, and search databases ==="

echo ">>> Downloading model checkpoints to ${CHECKPOINT_DIR}..."
gcs_sync "${GCS_BUCKET}/checkpoint/" "${CHECKPOINT_DIR}"
echo "Checkpoint contents:"
ls -1 "$CHECKPOINT_DIR"

echo ">>> Downloading common data files to ${COMMON_DIR}..."
gcs_sync "${GCS_BUCKET}/common/" "${COMMON_DIR}"
echo "Common contents:"
ls -1 "$COMMON_DIR"

echo ">>> Downloading VHH + template search databases to ${DB_DIR}..."
gcs_sync "${GCS_BUCKET}/search_database/" "${DB_DIR}"
echo "Search DB contents:"
ls -1R "$DB_DIR"

# --- Verification ---
echo ">>> Verifying critical files ==="
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

# --- Colabfold DB (existing block, kept as-is) ---
COLABFOLD_TAR_REMOTE="${GCS_BUCKET}/colabfold_db.tar"
COLABFOLD_TAR_LOCAL="${DB_DIR}/colabfold_db.tar"

if [[ $SKIP_DB -eq 1 ]]; then
    echo "--no-db flag set: skipping ColabFold database download."
    echo "  (Would have downloaded: ${COLABFOLD_TAR_REMOTE})"
else
    if [[ -f "$COLABFOLD_TAR_LOCAL" ]]; then
        echo "ColabFold database already exists at ${COLABFOLD_TAR_LOCAL}. Skipping download."
    else
        echo "Downloading ColabFold database (this may take a while)..."
        gcs_cp "$COLABFOLD_TAR_REMOTE" "$COLABFOLD_TAR_LOCAL"
        echo "Extracting ColabFold database..."
        tar -xvf "$COLABFOLD_TAR_LOCAL" -C "$DB_DIR"
        echo "ColabFold database download and extraction complete."
    fi
fi

# --- Finalization ---
# Need to run opendde <command> a first time to build the fast_layer_norm_cuda_v2 module
# If this works, then everything passes the test
opendde pred --help | head
echo ""
echo "OpenDDE setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Paths:"
echo "  - Checkpoint:      $CHECKPOINT_DIR"
echo "  - Common data:     $COMMON_DIR"
if [[ $SKIP_DB -eq 1 ]]; then
    echo "  - Search DB:       $DB_DIR (colabfold skipped)"
else
    echo "  - Search DB:       $DB_DIR"
fi
