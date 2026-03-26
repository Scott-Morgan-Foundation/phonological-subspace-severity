#!/bin/bash
# Track 4: Batch MFA alignment via Docker on DGX
# Runs one Docker container per dataset (much faster than per-speaker)
#
# Usage: bash scripts/track4_mfa_docker_batch.sh [dataset_name]
#        bash scripts/track4_mfa_docker_batch.sh --all

set -euo pipefail

DYSARTHRIA_DIR="$HOME/dysarthria"
CORPUS_DIR="$DYSARTHRIA_DIR/results/track4/corpus"
ALIGNED_DIR="$DYSARTHRIA_DIR/results/track4/aligned"
MFA_ROOT="$HOME/mfa_docker_root"
MFA_IMAGE="mmcauliffe/montreal-forced-aligner:latest"

# Dataset -> acoustic_model, dictionary
declare -A ACOUSTIC=(
    [SAP]=english_mfa
    [COPAS]=dutch_cv
    [TORGO]=english_mfa
    [Neurovoz]=spanish_mfa
    [YouTube_French]=french_mfa
    [MDSC]=mandarin_mfa
    [PC-GITA]=spanish_mfa
    [IPVS]=italian_cv
    [UASPEECH]=english_mfa
    [UASPEECH_control]=english_mfa
    [LibriSpeech_English]=english_mfa
)
declare -A DICTIONARY=(
    [SAP]=english_mfa
    [COPAS]=dutch_cv
    [TORGO]=english_mfa
    [Neurovoz]=spanish_mfa
    [YouTube_French]=french_mfa
    [MDSC]=mandarin_mfa
    [PC-GITA]=spanish_mfa
    [IPVS]=italian_cv
    [UASPEECH]=english_mfa
    [UASPEECH_control]=english_mfa
    [LibriSpeech_English]=english_mfa
)

align_dataset() {
    local ds="$1"
    local acou="${ACOUSTIC[$ds]}"
    local dict="${DICTIONARY[$ds]}"
    local corpus="$CORPUS_DIR/$ds"
    local output="/tmp/mfa_out_${ds}"

    if [ ! -d "$corpus" ]; then
        echo "  SKIP: corpus dir not found: $corpus"
        return 1
    fi

    local n_wav=$(find "$corpus" -name "*.wav" | wc -l)
    local n_lab=$(find "$corpus" -name "*.lab" | wc -l)
    if [ "$n_wav" -eq 0 ] || [ "$n_lab" -eq 0 ]; then
        echo "  SKIP: no wav/lab files in $corpus (wav=$n_wav, lab=$n_lab)"
        return 1
    fi

    # Create output dir as current user (uid 1000 = mfauser in container)
    mkdir -p "$output"

    echo "  Corpus: $n_wav wav, $n_lab lab files"
    echo "  Acoustic: $acou, Dictionary: $dict"
    echo "  Output: $output"

    local start_time=$(date +%s)

    docker run --platform linux/amd64 --rm \
        -e OMP_NUM_THREADS=1 -e KMP_AFFINITY=disabled \
        -e MFA_ROOT_DIR=/tmp/mfa_root \
        -v "$MFA_ROOT":/tmp/mfa_root \
        -v "$DYSARTHRIA_DIR":$DYSARTHRIA_DIR \
        -v "$output":/tmp/output \
        "$MFA_IMAGE" \
        mfa align "$DYSARTHRIA_DIR/results/track4/corpus/$ds" \
            "$acou" "$dict" /tmp/output \
            --clean --num_jobs 1 2>&1

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    local n_tg=$(find "$output" -name "*.TextGrid" | wc -l)
    echo "  Result: $n_tg TextGrids in ${duration}s"

    # Copy to final location
    mkdir -p "$ALIGNED_DIR/$ds"
    cp -r "$output"/* "$ALIGNED_DIR/$ds/" 2>/dev/null || true
    echo "  Copied to $ALIGNED_DIR/$ds"

    return 0
}

# Parse arguments
if [ "${1:-}" = "--all" ]; then
    DATASETS=(SAP COPAS Neurovoz YouTube_French MDSC PC-GITA UASPEECH UASPEECH_control LibriSpeech_English)
    # TORGO already done, IPVS needs pseudo-labelling first, VOC-ALS has no text
elif [ -n "${1:-}" ]; then
    DATASETS=("$1")
else
    echo "Usage: $0 --all | dataset_name"
    echo "Available: SAP COPAS TORGO Neurovoz YouTube_French MDSC PC-GITA IPVS UASPEECH UASPEECH_control LibriSpeech_English"
    exit 1
fi

echo "========================================"
echo "Track 4: Batch MFA Alignment (Docker)"
echo "========================================"
echo "Datasets: ${DATASETS[*]}"
echo ""

total_success=0
total_failed=0

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== $ds ==="
    if align_dataset "$ds"; then
        total_success=$((total_success + 1))
    else
        total_failed=$((total_failed + 1))
    fi
done

echo ""
echo "========================================"
echo "SUMMARY: $total_success success, $total_failed failed"
echo "========================================"

# Report TextGrid counts
echo ""
echo "TextGrid counts per dataset:"
for ds in "${DATASETS[@]}"; do
    n=$(find "$ALIGNED_DIR/$ds" -name "*.TextGrid" 2>/dev/null | wc -l)
    echo "  $ds: $n"
done
