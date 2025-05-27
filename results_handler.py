"""
Handler for saving and managing MCMC analysis results.
Provides CSV export functionality and statistical analysis of fitted parameters.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from config import Config

class ResultsHandler:
    """Handles saving and processing of MCMC fitting results.
    
    Manages export of parameter fits, statistical summaries, and analysis
    results to various file formats for further analysis and visualization.
    """
    
    def __init__(self, config: Config):
        """Initialize results handler with configuration.
        
        Args:
            config: Configuration object containing halo name and file paths
        """
        self.config = config
        self._output_dir = Path("results")  # Default output directory
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist.
        
        Creates the results directory structure for organized output storage.
        """
        self._output_dir.mkdir(exist_ok=True)
    
    def save_to_csv(self, file_name: str, header: List[str], rows: List[List[Union[str, float]]], 
                    delimiter: str = '\t') -> None:
        """Generalized function to save tabular data to CSV file.
        
        Args:
            file_name: Name of output CSV file
            header: Column headers for the CSV
            rows: Data rows as list of lists
            delimiter: Field separator (default: tab-separated)
            
        Raises:
            IOError: If file cannot be written
            ValueError: If headers and data dimensions don't match
        """
        if not header:
            raise ValueError("Header cannot be empty")
        
        if rows and len(rows[0]) != len(header):
            raise ValueError(f"Row length ({len(rows[0])}) doesn't match header length ({len(header)})")
        
        output_path = self._output_dir / file_name
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=delimiter)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Successfully saved {len(rows)} rows to {output_path}")
        except IOError as e:
            raise IOError(f"Failed to write CSV file {output_path}: {e}")

    def save_parameters(self, all_results: List[Dict[str, Any]]) -> None:
        """Save MCMC best-fit parameters to CSV file.
        
        Exports the maximum a posteriori (MAP) parameter estimates
        for each rotation to enable statistical analysis of morphology.
        
        Args:
            all_results: List of result dictionaries from MCMC fitting
                        Each dict should contain 'best_params' key
                        
        Raises:
            ValueError: If results list is empty or malformed
        """
        if not all_results:
            raise ValueError("No results provided to save")
        
        # Validate result format
        for i, result in enumerate(all_results):
            if 'best_params' not in result:
                raise ValueError(f"Result {i} missing 'best_params' key")
            if len(result['best_params']) != 3:
                raise ValueError(f"Result {i} 'best_params' should have 3 elements, got {len(result['best_params'])}")
        
        # Prepare CSV data
        header = ["Rotation_ID", "Halo_Name", "Ellipticity", "Position_Angle_deg", "Half_Light_Radius_kpc"]
        rows = []
        
        for i, result in enumerate(all_results):
            e, theta, rh = result['best_params']
            row = [
                f"rotation_{i:03d}",  # Zero-padded rotation ID
                self.config.halo_name,  # Include halo name for identification
                f"{e:.6f}",  # High precision for ellipticity
                f"{theta:.3f}",  # Degrees with 3 decimal places
                f"{rh:.6f}"  # High precision for radius in kpc
            ]
            rows.append(row)
        
        # Generate descriptive filename
        filename = f"parameters_{self.config.halo_name}.csv"
        self.save_to_csv(filename, header, rows)

    def save_parameter_statistics(self, all_results: List[Dict[str, Any]]) -> None:
        """Save statistical summary of parameters across all rotations.
        
        Computes and saves mean, median, standard deviation, and percentiles
        for each fitted parameter to characterize galaxy morphology robustness.
        
        Args:
            all_results: List of result dictionaries from MCMC fitting
        """
        if not all_results:
            raise ValueError("No results provided for statistics")
        
        # Extract parameters into arrays
        ellipticities = np.array([result['best_params'][0] for result in all_results])
        position_angles = np.array([result['best_params'][1] for result in all_results])
        half_light_radii = np.array([result['best_params'][2] for result in all_results])
        
        # Handle angle wraparound for position angle statistics
        # Convert to complex numbers for circular statistics
        angles_complex = np.exp(2j * np.pi * position_angles / 180.0)
        mean_angle_complex = np.mean(angles_complex)
        mean_position_angle = np.angle(mean_angle_complex) * 180.0 / (2 * np.pi)
        # Ensure angle is in [0, 180) range
        if mean_position_angle < 0:
            mean_position_angle += 180
        
        # Calculate statistics for each parameter
        stats_data = []
        
        # Ellipticity statistics
        stats_data.append([
            "Ellipticity", len(ellipticities),
            f"{np.mean(ellipticities):.6f}", f"{np.median(ellipticities):.6f}",
            f"{np.std(ellipticities):.6f}", f"{np.min(ellipticities):.6f}", f"{np.max(ellipticities):.6f}",
            f"{np.percentile(ellipticities, 16):.6f}", f"{np.percentile(ellipticities, 84):.6f}"
        ])
        
        # Position angle statistics (using circular mean)
        stats_data.append([
            "Position_Angle_deg", len(position_angles),
            f"{mean_position_angle:.3f}", f"{np.median(position_angles):.3f}",
            f"{np.std(position_angles):.3f}", f"{np.min(position_angles):.3f}", f"{np.max(position_angles):.3f}",
            f"{np.percentile(position_angles, 16):.3f}", f"{np.percentile(position_angles, 84):.3f}"
        ])
        
        # Half-light radius statistics
        stats_data.append([
            "Half_Light_Radius_kpc", len(half_light_radii),
            f"{np.mean(half_light_radii):.6f}", f"{np.median(half_light_radii):.6f}",
            f"{np.std(half_light_radii):.6f}", f"{np.min(half_light_radii):.6f}", f"{np.max(half_light_radii):.6f}",
            f"{np.percentile(half_light_radii, 16):.6f}", f"{np.percentile(half_light_radii, 84):.6f}"
        ])
        
        header = ["Parameter", "N_Rotations", "Mean", "Median", "Std_Dev", "Min", "Max", "P16", "P84"]
        filename = f"parameter_statistics_{self.config.halo_name}.csv"
        self.save_to_csv(filename, header, stats_data)

    def save_mcmc_chains(self, all_results: List[Dict[str, Any]], thin_factor: int = 10) -> None:
        """Save MCMC chain samples for detailed posterior analysis.
        
        Exports the full posterior samples (optionally thinned) for each rotation
        to enable advanced statistical analysis and convergence diagnostics.
        
        Args:
            all_results: List of result dictionaries containing 'samples'
            thin_factor: Factor by which to thin chains (every Nth sample)
                        
        Note:
            This can generate large files. Use thin_factor to reduce file size
            while maintaining representative posterior sampling.
        """
        if not all_results:
            raise ValueError("No results provided for chain export")
        
        # Process each rotation's MCMC samples
        for rotation_idx, result in enumerate(all_results):
            if 'samples' not in result:
                print(f"Warning: No samples found for rotation {rotation_idx}")
                continue
                
            samples = result['samples']
            
            # Thin the samples if requested
            if thin_factor > 1:
                samples = samples[::thin_factor]
            
            # Prepare data for CSV
            header = ["Sample_ID", "Ellipticity", "Position_Angle_deg", "Half_Light_Radius_kpc"]
            rows = []
            
            for sample_idx, sample in enumerate(samples):
                row = [
                    sample_idx * thin_factor,  # Account for thinning in sample ID
                    f"{sample[0]:.6f}",  # Ellipticity
                    f"{sample[1]:.3f}",  # Position angle
                    f"{sample[2]:.6f}"   # Half-light radius
                ]
                rows.append(row)
            
            # Save individual rotation chain
            filename = f"mcmc_chain_rotation_{rotation_idx:03d}_{self.config.halo_name}.csv"
            self.save_to_csv(filename, header, rows)

    def save_summary_report(self, all_results: List[Dict[str, Any]]) -> None:
        """Generate and save a comprehensive analysis summary.
        
        Creates a human-readable report summarizing the morphological
        analysis results and key findings.
        
        Args:
            all_results: List of result dictionaries from MCMC fitting
        """
        if not all_results:
            print("No results available for summary report")
            return
        
        # Extract parameter arrays
        ellipticities = np.array([result['best_params'][0] for result in all_results])
        position_angles = np.array([result['best_params'][1] for result in all_results])
        half_light_radii = np.array([result['best_params'][2] for result in all_results])
        
        # Generate summary statistics
        summary_lines = [
            f"Galaxy Morphology Analysis Summary",
            f"{'='*50}",
            f"",
            f"Halo: {self.config.halo_name}",
            f"Number of rotations analyzed: {len(all_results)}",
            f"",
            f"Parameter Summary:",
            f"-----------------",
            f"Ellipticity:",
            f"  Mean ± Std: {np.mean(ellipticities):.4f} ± {np.std(ellipticities):.4f}",
            f"  Median: {np.median(ellipticities):.4f}",
            f"  Range: [{np.min(ellipticities):.4f}, {np.max(ellipticities):.4f}]",
            f"",
            f"Position Angle (degrees):",
            f"  Mean ± Std: {np.mean(position_angles):.2f} ± {np.std(position_angles):.2f}",
            f"  Median: {np.median(position_angles):.2f}",
            f"  Range: [{np.min(position_angles):.2f}, {np.max(position_angles):.2f}]",
            f"",
            f"Half-light Radius (kpc):",
            f"  Mean ± Std: {np.mean(half_light_radii):.4f} ± {np.std(half_light_radii):.4f}",
            f"  Median: {np.median(half_light_radii):.4f}",
            f"  Range: [{np.min(half_light_radii):.4f}, {np.max(half_light_radii):.4f}]",
            f"",
            f"Analysis completed successfully.",
            f"Results saved to: {self._output_dir.absolute()}"
        ]
        
        # Save summary to text file
        summary_path = self._output_dir / f"analysis_summary_{self.config.halo_name}.txt"
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            print(f"Summary report saved to {summary_path}")
        except IOError as e:
            print(f"Failed to save summary report: {e}")

    def get_output_directory(self) -> Path:
        """Get the current output directory path.
        
        Returns:
            Path object for the results output directory
        """
        return self._output_dir
    
    def set_output_directory(self, directory: Union[str, Path]) -> None:
        """Set a custom output directory for results.
        
        Args:
            directory: Path to desired output directory
            
        Raises:
            OSError: If directory cannot be created
        """
        self._output_dir = Path(directory)
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory set to: {self._output_dir.absolute()}")
        except OSError as e:
            raise OSError(f"Cannot create output directory {self._output_dir}: {e}")