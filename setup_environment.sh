#!/bin/bash
set -e

VENV_NAME="galaxy_morphology"

show_help() {
    cat << 'EOF'
Usage: ./setup_environment.sh [COMMAND]

Commands:
  install    Set up virtual environment (default)
  validate   Check if environment works
  clean      Remove environment
  help       Show this help
EOF
}

install() {
    echo "Setting up virtual environment..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 not found. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version is reasonable
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Found Python $python_version"
    
    # Remove old environment if it exists
    if [[ -d "$VENV_NAME" ]]; then
        echo "Removing existing environment..."
        rm -rf "$VENV_NAME"
    fi
    
    # Create requirements.txt
    echo "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core scientific computing
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0

# MCMC and visualization
emcee>=3.1.0
corner>=2.2.0

# Simulation data handling (required for your analysis)
pynbody>=1.0.0
tangos>=1.0.0

# Development and testing
pytest>=6.0.0
EOF
    
    # Create and setup virtual environment
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
    
    echo "Installing packages..."
    source "${VENV_NAME}/bin/activate"
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    
    echo ""
    echo "Setup complete!"
    echo "To activate: source ${VENV_NAME}/bin/activate"
    echo "To test: ./setup_environment.sh validate"
}

validate() {
    echo "Validating installation..."
    
    if [[ ! -d "$VENV_NAME" ]]; then
        echo "Error: Virtual environment not found. Run './setup_environment.sh install' first."
        exit 1
    fi
    
    source "${VENV_NAME}/bin/activate"
    
    # Test each module
    modules=("numpy" "matplotlib" "scipy" "emcee" "corner" "pynbody" "tangos")
    failed=0
    
    for module in "${modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            echo "✓ $module"
        else
            echo "✗ $module - FAILED"
            failed=1
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        echo ""
        echo "All modules working correctly!"
        echo "Environment is ready to use."
    else
        echo ""
        echo "Some modules failed. Try running 'install' again."
        exit 1
    fi
}

clean() {
    echo "Cleaning up environment..."
    
    if [[ -d "$VENV_NAME" ]]; then
        rm -rf "$VENV_NAME"
        echo "Removed virtual environment"
    fi
    
    if [[ -f "requirements.txt" ]]; then
        rm -f "requirements.txt"
        echo "Removed requirements.txt"
    fi
    
    echo "Cleanup complete"
}

case "${1:-install}" in
    "install") install ;;
    "validate") validate ;;
    "clean") clean ;;
    "help"|"--help") show_help ;;
    *) echo "Unknown command: $1"; show_help; exit 1 ;;
esac