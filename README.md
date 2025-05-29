# Galaxy Morphology Analysis with MCMC

A comprehensive Python toolkit for analyzing galaxy morphology using Bayesian MCMC fitting of elliptical exponential surface brightness profiles. This package processes cosmological simulation data to extract robust morphological parameters through multiple random orientation sampling.

## Features

- **Bayesian Parameter Estimation**: Uses `emcee` for robust MCMC fitting of galaxy surface brightness profiles
- **Multiple Orientation Sampling**: Analyzes galaxies from random viewing angles to assess morphological robustness
- **Parallel Processing**: Efficient multiprocessing implementation for handling large datasets
- **Publication-Quality Visualizations**: Generates corner plots, contour maps, and statistical summaries
- **Comprehensive Output**: CSV exports, statistical analysis, and human-readable reports
- **Modular Design**: Clean, well-documented codebase following best practices
- **Automated Setup**: Bash scripts for easy environment management and analysis execution

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/agoater/galaxy-morphology-mcmc.git
cd galaxy-morphology-mcmc

# Make scripts executable
chmod +x *.sh

# Set up environment automatically
./setup_environment.sh
```

This will create a virtual environment, install all dependencies, and validate the installation.

### Manual Installation

If you prefer manual setup:

```bash
# Core scientific computing
pip install numpy matplotlib scipy

# MCMC and visualization
pip install emcee corner

# Simulation data handling
pip install pynbody tangos

# Development tools
pip install pytest
```

## Quick Start

### Using Bash Scripts (Recommended)

```bash
# Set up environment (first time only)
./setup_environment.sh

# Run analysis
./run_analysis.sh -s /path/to/simulation/snapshot -h your_halo_identifier

# Get help on available options
./run_analysis.sh --help
```

### Direct Python Usage

```python
from main import main

# Run analysis with specific data files
main(
    snapshot_path='/path/to/simulation/snapshot',
    halo_name='your_halo_identifier',
    N=100,  # Number of random rotations
    nwalkers=50,  # MCMC walkers
    production_steps=1000  # MCMC production steps
)
```

## Command Line Interface

### Environment Setup Script

```bash
./setup_environment.sh [COMMAND]

Commands:
  install    Set up virtual environment (default)
  validate   Check if environment works
  clean      Remove environment
  help       Show help
```

### Analysis Runner Script

```bash
./run_analysis.sh -s SNAPSHOT -h HALO [OPTIONS]

Required:
  -s, --snapshot PATH     Snapshot file path
  -h, --halo NAME         Halo name

Optional:
  -n, --rotations NUM     Number of rotations (default: 50)
  -w, --walkers NUM       MCMC walkers (default: 50)
  -p, --production NUM    Production steps (default: 1000)
  -o, --output DIR        Output directory
  --help                 Show help

Examples:
  ./run_analysis.sh -s /data/snap.hdf5 -h halo_001
  ./run_analysis.sh -s /data/snap.hdf5 -h halo_001 -n 100 -p 2000
```

## Methodology

### Surface Brightness Model

The package fits an elliptical exponential surface brightness profile:

```
I(r) = I₀ × exp(-r/r_scale)
```

Where:
- `r` is the elliptical radius accounting for galaxy orientation and shape
- `I₀` is the central surface brightness normalization
- `r_scale` is derived from the half-light radius

### MCMC Parameter Estimation

Three primary parameters are fitted:
1. **Ellipticity (e)**: Shape parameter [0, 1), where 0=circular, approaching 1=linear
2. **Position Angle (θ)**: Orientation in degrees [-90°, 90°)  
3. **Half-light Radius (r_h)**: Scale length in kpc

### Multi-Orientation Analysis

- Generates N random viewing orientations using uniform sphere sampling
- Fits morphological parameters for each orientation
- Provides statistical assessment of parameter robustness
- Identifies projection effects and intrinsic morphology

## Output Files

### Results Directory
- `parameters_[halo].csv`: Best-fit parameters for each rotation
- `parameter_statistics_[halo].csv`: Statistical summary across rotations
- `analysis_summary_[halo].txt`: Human-readable analysis report
- `mcmc_chain_rotation_XXX_[halo].csv`: Full MCMC chains (optional)

### Plots Directory
- `corner_rotXXX_[halo].png`: Parameter correlation plots for each rotation
- `contour_rotXXX_[halo].png`: Surface brightness contours with stellar metallicity
- `parameter_distributions_[halo].png`: Parameter distribution summary

## Module Overview

### Core Modules

- **`config.py`**: Configuration management with validation
- **`data_loader.py`**: Simulation data loading and preprocessing  
- **`rotation_handler.py`**: 3D rotation matrix generation and application
- **`model.py`**: MCMC surface brightness model implementation
- **`utils.py`**: Surface brightness calculations and coordinate transformations
- **`results_handler.py`**: Output management and statistical analysis
- **`visualisation.py`**: Publication-quality plot generation
- **`main.py`**: Analysis pipeline orchestration

### Automation Scripts

- **`setup_environment.sh`**: Automated environment setup and dependency installation
- **`run_analysis.sh`**: Command-line interface for running analyses with validation

### Key Classes

```python
# Configuration with validation
config = Config(snapshot_path='...', halo_name='...', N=50)

# Data loading and extraction  
loader = DataLoader(config)
simulation, halo = loader.load_simulation_data()
stellar_data = loader.extract_stellar_data(simulation, halo)

# MCMC fitting
model = MCMCModel(config)
sampler = model.run_mcmc(initial_params, stellar_data)
best_params = model.get_best_parameters(sampler)

# Visualization
visualizer = Visualizer(config)
visualizer.plot_corner(samples, best_params, rotation_idx)
```

## Configuration Parameters

### Physical Parameters
- `e_initial`: Initial ellipticity guess (default: 0.5)
- `theta_initial`: Initial position angle in degrees (default: 40)
- `rh_initial`: Initial half-light radius in kpc (default: 0.2)

### MCMC Parameters  
- `nwalkers`: Number of MCMC ensemble walkers (default: 50)
- `burn_in_steps`: Burn-in phase length (default: 100)
- `production_steps`: Production phase length (default: 500)

### Analysis Settings
- `SB_lim`: Surface brightness cut limit in mag/arcsec² (default: 30)
- `N`: Number of random rotations (default: 50)
- `metallicity_threshold`: Minimum stellar metallicity (default: 1e-10)

## Examples

### Command Line Examples

```bash
# Basic analysis
./run_analysis.sh -s /path/to/simulation/snapshot -h your_halo_identifier

# High-precision analysis
./run_analysis.sh -s /path/to/simulation/snapshot -h your_halo_identifier -n 100 -p 2000

# Custom output directory
./run_analysis.sh -s /path/to/simulation/snapshot -h your_halo_identidier -o my_results
```

### Python Examples

```python
# Custom surface brightness analysis
from utils import calculate_surface_brightness, create_meshgrid

# Create surface brightness map
xx, yy = create_meshgrid(x_coords, y_coords, n_points=1000)
sb_map = calculate_surface_brightness(xx, yy, e=0.3, theta=45, rh=2.5, vlum=1e10)
```

```python
# Statistical analysis
from results_handler import ResultsHandler

handler = ResultsHandler(config)
handler.save_parameter_statistics(all_results)
handler.save_mcmc_chains(all_results, thin_factor=10)
```

## Performance Notes

- **Parallel Processing**: Automatically uses available CPU cores minus one
- **Memory Efficiency**: Processes rotations independently to minimize memory usage
- **Scalability**: Handles datasets with 10⁵-10⁶ stellar particles efficiently
- **Progress Tracking**: Provides real-time feedback during long analyses
- **Environment Isolation**: Virtual environment prevents dependency conflicts

## Troubleshooting

### Environment Issues

**Installation Problems**:
```bash
# Clean and reinstall environment
./setup_environment.sh clean
./setup_environment.sh install

# Validate installation
./setup_environment.sh validate
```

**Python Import Errors**:
- Ensure virtual environment is activated: `source galaxy_morphology/bin/activate`
- Check if all modules installed: `./setup_environment.sh validate`

### Analysis Issues

**MCMC Convergence Problems**:
- Increase `burn_in_steps` and `production_steps`
- Adjust initial parameter guesses
- Check acceptance fraction (should be 0.2-0.8)

**Memory Issues**:
- Reduce number of parallel processes
- Use chain thinning for large MCMC outputs
- Process halos individually rather than in batches

**Surface Brightness Cut Removes All Stars**:
- Increase `SB_lim` value (higher = dimmer limit)
- Check stellar particle selection criteria
- Verify V-band magnitude calculations

**File Not Found Errors**:
- Check snapshot file path is correct
- Ensure halo name exists in tangos database
- Verify file permissions and accessibility

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Test bash scripts work correctly
6. Submit a pull request

## Acknowledgments

- Built with [emcee](https://emcee.readthedocs.io/) for MCMC sampling
- Uses [corner](https://corner.readthedocs.io/) for parameter visualization
- Simulation handling via [pynbody](https://pynbody.github.io/) and [tangos](https://tangos.readthedocs.io/)
- Inspired by observational galaxy morphology studies

---