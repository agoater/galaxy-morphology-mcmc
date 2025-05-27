"""
Main script for galaxy morphology analysis using MCMC.
Processes multiple random rotations in parallel and generates visualizations.
"""

import multiprocessing
from functools import partial
from typing import Dict, Any, Optional
import numpy as np

from config import Config, create_config
from data_loader import DataLoader
from rotation_handler import RotationHandler
from model import MCMCModel
from visualisation import Visualizer
from results_handler import ResultsHandler
from utils import (
    apply_surface_brightness_cut,
    create_meshgrid,
    calculate_surface_brightness
)

def process_rotation(Rmatrix: np.ndarray, stellar_data: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """Process a single rotation and return all relevant data products.
    This function is designed to be parallelized.
    Note: All heavy computations are contained here for clean parallelization.
    
    Args:
        Rmatrix: 3x3 rotation matrix
        stellar_data: Dictionary containing stellar particle data
        config: Configuration object
        
    Returns:
        Dictionary containing MCMC samples and processed data products
        
    Raises:
        RuntimeError: If MCMC fitting fails
        ValueError: If surface brightness cut removes all stars
    """
    try:
        # Initialize modules fresh for each process (important for parallelization)
        rot_handler = RotationHandler(config)
        model = MCMCModel(config)

        # Rotate star positions using matrix multiplication
        rotated = rot_handler.apply_rotation(stellar_data['starpos'], Rmatrix)
        x, y = rotated[:, 0], rotated[:, 1]  # Extract x,y coordinates
        ox, oy = x.copy(), y.copy()  # Keep original for comparison

        # Prepare basic data
        Nstar = len(x)  # Number of stars
        stellar_mass = np.sum(stellar_data['mstar'])  # Total stellar mass
        vlum = 10 ** ((stellar_data['vmag'] - 4.84) / -2.5)  # Convert mag to luminosity

        # First MCMC run to get initial parameters
        sampler = model.run_mcmc(
            [config.e_initial, config.theta_initial, config.rh_initial],
            (x, y, stellar_data['mstar'], Nstar, stellar_mass)
        )
        
        if sampler is None:
            raise RuntimeError("MCMC fitting failed to converge")
            
        e, theta, rh = model.get_best_parameters(sampler)

        # Apply surface brightness cut using the initial fit
        SB_func = calculate_surface_brightness(x, y, e, theta, rh, vlum)
        # Apply surface brightness cut and handle return values correctly
        cut_results = apply_surface_brightness_cut(
            SB_func, config.SB_lim, x, y, stellar_data['feh']
        )
        x_cut, y_cut, feh_cut = cut_results
        
        if len(x_cut) == 0:
            raise ValueError("Surface brightness cut removed all stellar particles")

        # Create meshgrid for contour plotting
        xx, yy = create_meshgrid(x_cut, y_cut)
        zz = calculate_surface_brightness(xx, yy, e, theta, rh, vlum)

        # Return comprehensive results as dictionary
        return {
            'samples': sampler.flatchain,
            'best_params': (e, theta, rh),
            'x_cut': x_cut,
            'y_cut': y_cut,
            'x_orig': ox,
            'y_orig': oy,
            'xx': xx,
            'yy': yy,
            'zz': zz,
            'cutfeh': feh_cut,
            'SB_func': SB_func,
            'stellar_mass': stellar_mass,
            'vlum': vlum,
            'Nstar_original': Nstar,
            'Nstar_cut': len(x_cut)
        }
        
    except Exception as ex:
        print(f"Error processing rotation: {ex}")
        raise

def main(snapshot_path: Optional[str] = None, halo_name: Optional[str] = None, **config_kwargs) -> None:
    """Main analysis pipeline for galaxy morphology fitting.
    
    Args:
        snapshot_path: Path to simulation snapshot (if None, uses config default)
        halo_name: Name of halo in database (if None, uses config default)
        **config_kwargs: Additional configuration parameters to override
        
    Raises:
        ValueError: If required paths are not provided
        RuntimeError: If parallel processing fails
    """
    try:
        # Initialize configuration with validation
        if snapshot_path and halo_name:
            config = create_config(snapshot_path, halo_name, **config_kwargs)
        else:
            config = Config(**config_kwargs)
            config.validate()
            config.initialize()

        print(f"Starting analysis for halo: {config.halo_name}")
        print(f"Using {config.N} random rotations with {multiprocessing.cpu_count()} CPU cores")

        # Load data - this is done once to avoid repeated I/O
        loader = DataLoader(config)
        s, halo = loader.load_simulation_data()
        stellar_data = loader.extract_stellar_data(s, halo)
        
        print(f"Loaded {len(stellar_data['starpos'])} stellar particles within r200c")

        # Generate rotation matrices upfront
        rot_handler = RotationHandler(config)
        Rmatrices = rot_handler.generate_random_rotations()

        # Parallel processing using multiprocessing
        # Using reasonable number of processes (leave some cores free)
        n_processes = min(multiprocessing.cpu_count() - 1, config.N)
        print(f"Processing rotations using {n_processes} parallel processes...")
        
        # partial() creates a function with fixed arguments for mapping
        with multiprocessing.Pool(n_processes) as pool:
            func = partial(process_rotation, stellar_data=stellar_data, config=config)
            all_results = pool.map(func, Rmatrices)  # Distribute work across cores

        # Filter out any failed results (if error handling allows None returns)
        successful_results = [r for r in all_results if r is not None]
        print(f"Successfully processed {len(successful_results)}/{len(Rmatrices)} rotations")

        if not successful_results:
            raise RuntimeError("No rotations processed successfully")

        # Save and visualize results
        print("Saving results and generating visualizations...")
        results_handler = ResultsHandler(config)
        results_handler.save_parameters(successful_results)
        results_handler.save_parameter_statistics(successful_results)
        results_handler.save_summary_report(successful_results)

        visualizer = Visualizer(config)
        
        # Generate individual rotation plots
        for i, result in enumerate(successful_results):
            visualizer.plot_corner(result['samples'], result['best_params'], i)
            visualizer.plot_contour(
                result['xx'], result['yy'], result['zz'],
                result['x_cut'], result['y_cut'],
                result['x_orig'], result['y_orig'],
                result['cutfeh'], stellar_data['feh'], i
            )
        
        # Generate summary plot showing parameter distributions
        visualizer.plot_parameter_distributions(successful_results)

        print("Analysis completed successfully!")
        print(f"Results saved for {len(successful_results)} rotations")
        print(f"Output files saved to:")
        print(f"  - Results: {results_handler.get_output_directory()}")
        print(f"  - Plots: {visualizer.get_output_directory()}")

    except Exception as ex:
        print(f"Analysis failed: {ex}")
        raise

if __name__ == "__main__":
    # Modify these paths for your data
    # main('/path/to/snapshot', 'halo_name')
    
    # Or use config defaults (make sure to set them first)
    main()  # Only execute if run as script, not when imported