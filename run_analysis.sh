#!/bin/bash
set -e

# Defaults
N_ROTATIONS=50
NWALKERS=50
PRODUCTION_STEPS=1000
SNAPSHOT_PATH=""
HALO_NAME=""
OUTPUT_DIR=""

show_help() {
    cat << 'EOF'
Usage: ./run_analysis.sh -s SNAPSHOT -h HALO [OPTIONS]

Required:
  -s, --snapshot PATH     Snapshot file path
  -h, --halo NAME         Halo name

Optional:
  -n, --rotations NUM     Number of rotations (default: 50)
  -w, --walkers NUM       MCMC walkers (default: 50)
  -p, --production NUM    Production steps (default: 1000)
  -o, --output DIR        Output directory
  --help                 Show this help

Examples:
  ./run_analysis.sh -s /data/snap.hdf5 -h halo_001
  ./run_analysis.sh -s /data/snap.hdf5 -h halo_001 -n 100
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--snapshot) SNAPSHOT_PATH="$2"; shift 2 ;;
        -h|--halo) HALO_NAME="$2"; shift 2 ;;
        -n|--rotations) N_ROTATIONS="$2"; shift 2 ;;
        -w|--walkers) NWALKERS="$2"; shift 2 ;;
        -p|--production) PRODUCTION_STEPS="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Check required parameters
if [[ -z "$SNAPSHOT_PATH" ]]; then
    echo "Error: Snapshot path required"
    show_help
    exit 1
fi

if [[ -z "$HALO_NAME" ]]; then
    echo "Error: Halo name required"
    show_help
    exit 1
fi

# Validate inputs
if [[ ! -f "$SNAPSHOT_PATH" ]]; then
    echo "Error: Snapshot file not found: $SNAPSHOT_PATH"
    exit 1
fi

# Check that numbers are actually numbers
if ! [[ "$N_ROTATIONS" =~ ^[0-9]+$ ]] || [[ $N_ROTATIONS -lt 1 ]]; then
    echo "Error: Rotations must be a positive number, got: $N_ROTATIONS"
    exit 1
fi

if ! [[ "$NWALKERS" =~ ^[0-9]+$ ]] || [[ $NWALKERS -lt 10 ]]; then
    echo "Error: Walkers must be a number >= 10, got: $NWALKERS"
    exit 1
fi

if ! [[ "$PRODUCTION_STEPS" =~ ^[0-9]+$ ]] || [[ $PRODUCTION_STEPS -lt 100 ]]; then
    echo "Error: Production steps must be a number >= 100, got: $PRODUCTION_STEPS"
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="analysis_$(date +%Y%m%d_%H%M%S)_${HALO_NAME}"
fi

# Check environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -d "galaxy_morphology" ]]; then
    echo "Activating virtual environment..."
    source galaxy_morphology/bin/activate
fi

if ! python3 -c "import main" 2>/dev/null; then
    echo "Error: Cannot import main.py"
    echo "Make sure you're in the right directory and environment is set up"
    exit 1
fi

# Show what we're about to do
echo "Starting Galaxy Morphology Analysis"
echo "==================================="
echo "Snapshot: $SNAPSHOT_PATH"
echo "Halo: $HALO_NAME"
echo "Rotations: $N_ROTATIONS"
echo "MCMC Walkers: $NWALKERS"
echo "Production Steps: $PRODUCTION_STEPS"
echo "Output: $OUTPUT_DIR"
echo ""

# Estimate runtime (very rough)
total_samples=$((N_ROTATIONS * PRODUCTION_STEPS * NWALKERS))
estimated_minutes=$((total_samples / 60000))
if [[ $estimated_minutes -gt 0 ]]; then
    echo "Estimated runtime: ~${estimated_minutes} minutes"
    echo ""
fi

# Check if output directory exists
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Warning: Output directory already exists: $OUTPUT_DIR"
    echo "Results may be overwritten. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

# Create output directory and run analysis
echo "Creating output directory..."
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Starting analysis..."
log_file="analysis_$(date +%Y%m%d_%H%M%S).log"

# Run the Python analysis
python3 -c "
from main import main
import sys

try:
    main(
        snapshot_path='$SNAPSHOT_PATH',
        halo_name='$HALO_NAME',
        N=$N_ROTATIONS,
        nwalkers=$NWALKERS,
        production_steps=$PRODUCTION_STEPS
    )
    print('Analysis completed successfully!')
except Exception as e:
    print(f'Analysis failed: {e}')
    sys.exit(1)
" 2>&1 | tee "$log_file"

echo ""
echo "Analysis complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $log_file"