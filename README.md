# Galaxy Morphology Analysis with MCMC

A comprehensive Python toolkit for analyzing galaxy morphology using Bayesian MCMC fitting of elliptical exponential surface brightness profiles. This package processes cosmological simulation data to extract robust morphological parameters through multiple random orientation sampling.

## Features

- **Bayesian Parameter Estimation**: Uses `emcee` for robust MCMC fitting of galaxy surface brightness profiles
- **Multiple Orientation Sampling**: Analyzes galaxies from random viewing angles to assess morphological robustness
- **Parallel Processing**: Efficient multiprocessing implementation for handling large datasets
- **Publication-Quality Visualizations**: Generates corner plots, contour maps, and statistical summaries
- **Comprehensive Output**: CSV exports, statistical analysis, and human-readable reports
- **Modular Design**: Clean, well-documented codebase following best practices

## Installation

### Prerequisites

```bash
# Core scientific computing
pip install numpy matplotlib scipy

# MCMC and visualization
pip install emcee corner

# Simulation data handling (install as needed for your data format)
pip install pynbody  # For simulation snapshots
pip install tangos   # For halo database queries
```

### Clone Repository

```bash
git clone https://github.com/agoater/phd/galaxy-morphology-mcmc.git
cd galaxy-morphology-mcmc
```

## Quick Start

### Basic Usage

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

### Configuration-Based Usage

```python
from config import create_config
from main import main

# Create custom configuration
config = create_config(
    snapshot_path='/path/to/snapshot',
    halo_name='halo_12345',
    e_initial=0.3,          # Initial ellipticity guess
    theta_initial=45,       # Initial position angle (degrees)
    rh_initial=2.0,         # Initial half-light radius (kpc)
    SB_lim=28.0,           # Surface brightness cut (mag/arcsec²)
    N=50,                  # Number of rotations
    random_seed=42         # For reproducibility
)

# Run analysis
main()  # Uses config defaults
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

### Analyzing Multiple Halos

```python
halo_list = ['halo_001', 'halo_002', 'halo_003']
snapshot_path = '/data/simulation_snapshot.hdf5'

for halo_name in halo_list:
    print(f"Analyzing {halo_name}...")
    main(snapshot_path, halo_name, N=100, production_steps=2000)
```

### Custom Surface Brightness Analysis

```python
from utils import calculate_surface_brightness, create_meshgrid

# Create surface brightness map
xx, yy = create_meshgrid(x_coords, y_coords, n_points=1000)
sb_map = calculate_surface_brightness(xx, yy, e=0.3, theta=45, rh=2.5, vlum=1e10)
```

### Statistical Analysis

```python
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

## Scientific Applications

This toolkit is designed for:
- **Galaxy Evolution Studies**: Tracking morphological changes across cosmic time
- **Simulation Validation**: Comparing simulated galaxy shapes with observations
- **Environmental Effects**: Analyzing morphology dependence on galaxy environment
- **Statistical Samples**: Processing large galaxy catalogs for population studies

## Troubleshooting

### Common Issues

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

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Built with [emcee](https://emcee.readthedocs.io/) for MCMC sampling
- Uses [corner](https://corner.readthedocs.io/) for parameter visualization
- Simulation handling via [pynbody](https://pynbody.github.io/) and [tangos](https://tangos.readthedocs.io/)
- Inspired by observational galaxy morphology studies