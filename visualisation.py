"""
Visualization module for galaxy morphology analysis results.
Generates corner plots, contour maps, and statistical summaries of MCMC fitting results.
"""

from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import corner
from config import Config

class Visualizer:
    """Handles visualization of MCMC fitting results and galaxy morphology data.
    
    Creates publication-quality plots including:
    - Corner plots showing parameter correlations and uncertainties
    - Surface brightness contour maps with stellar metallicity overlays
    - Parameter distribution summaries across multiple rotations
    """
    
    def __init__(self, config: Config):
        """Initialize visualizer with configuration settings.
        
        Args:
            config: Configuration object containing analysis parameters
        """
        self.config = config
        self._output_dir = Path("plots")  # Default plot output directory
        self._setup_matplotlib()
        self._ensure_output_directory()
    
    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for publication-quality plots."""
        plt.style.use('default')  # Use clean default style
        
        # Set publication-quality parameters
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def _ensure_output_directory(self) -> None:
        """Create output directory for plots if it doesn't exist."""
        self._output_dir.mkdir(exist_ok=True)

    def plot_corner(self, samples: np.ndarray, best_params: np.ndarray, 
                   rotation_idx: int, save_plot: bool = True) -> Optional[matplotlib.figure.Figure]:
        """Generate corner plot showing parameter correlations and uncertainties.
        
        Creates a triangular matrix plot displaying:
        - 1D histograms of parameter distributions (diagonal)
        - 2D contour plots showing parameter correlations (off-diagonal)
        - Best-fit parameters highlighted with red lines and markers
        
        Args:
            samples: MCMC samples array, shape (n_samples, n_parameters)
            best_params: Best-fit parameter values [ellipticity, theta, rh]
            rotation_idx: Index of current rotation for filename
            save_plot: Whether to save plot to file (default: True)
            
        Returns:
            Figure object if save_plot=False, None otherwise
            
        Raises:
            ValueError: If samples array has incorrect shape
            
        Note:
            Shows 16th, 50th, 84th percentiles (≈ ±1σ for Gaussian distributions)
            Red lines and squares highlight maximum a posteriori (MAP) estimates
        """
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(f"Samples must have shape (n_samples, 3), got {samples.shape}")
        
        if len(best_params) != 3:
            raise ValueError(f"Best params must have length 3, got {len(best_params)}")
        
        # Parameter labels with proper mathematical notation
        labels = [
            'Ellipticity, $e$',
            'Position Angle, $\\theta$ [°]', 
            'Half-light Radius, $r_h$ [kpc]'
        ]
        
        # Create corner plot with statistical annotations
        fig = corner.corner(
            samples,
            labels=labels,
            plot_datapoints=True,  # Show individual samples
            quantiles=[0.16, 0.5, 0.84],  # 1σ confidence intervals
            show_titles=True,  # Display parameter statistics
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            hist_kwargs={"bins": 50, "density": True, "alpha": 0.7},
            contour_kwargs={"colors": ["#1f77b4"], "linewidths": 1.5},
            scatter_kwargs={"alpha": 0.3, "s": 2}
        )

        # Extract axes for highlighting best-fit parameters
        axes = np.array(fig.axes).reshape((3, 3))

        # Highlight best-fit parameters with red lines and markers
        for i in range(3):
            # Vertical line on diagonal histograms
            axes[i, i].axvline(best_params[i], color="red", linewidth=2, 
                             label="MAP estimate", alpha=0.8)
            
            # Lines and markers on 2D correlation plots
            for j in range(i):
                ax = axes[i, j]
                ax.axvline(best_params[j], color="red", linewidth=1.5, alpha=0.7)
                ax.axhline(best_params[i], color="red", linewidth=1.5, alpha=0.7)
                ax.plot(best_params[j], best_params[i], "sr", markersize=8, 
                       markeredgecolor="darkred", markeredgewidth=1)

        # Add overall title with halo information
        fig.suptitle(f'Parameter Correlations - {self.config.halo_name} - Rotation {rotation_idx}', 
                    fontsize=16, y=0.98)

        if save_plot:
            filename = f'corner_rot{rotation_idx:03d}_{self.config.halo_name}.png'
            filepath = self._output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Corner plot saved: {filepath}")
            return None
        else:
            return fig

    def plot_contour(self, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray,
                    x_cut: np.ndarray, y_cut: np.ndarray, 
                    x_orig: np.ndarray, y_orig: np.ndarray,
                    feh_cut: np.ndarray, feh_orig: np.ndarray, 
                    rotation_idx: int, save_plot: bool = True) -> Optional[matplotlib.figure.Figure]:
        """Plot surface brightness contours with stellar metallicity scatter overlay.
        
        Creates side-by-side comparison showing:
        - Left panel: Stars after surface brightness cut
        - Right panel: All original stars
        Both panels show surface brightness contours with metallicity color-coding
        
        Args:
            xx, yy: Coordinate meshgrids for contour plotting (kpc)
            zz: Surface brightness values on meshgrid (mag/arcsec²)
            x_cut, y_cut: Stellar positions after brightness cut (kpc)
            x_orig, y_orig: Original stellar positions (kpc)
            feh_cut: Metallicity [Fe/H] for cut stars
            feh_orig: Metallicity [Fe/H] for original stars
            rotation_idx: Index of current rotation for filename
            save_plot: Whether to save plot to file (default: True)
            
        Returns:
            Figure object if save_plot=False, None otherwise
            
        Note:
            Surface brightness contours use magnitude scale (higher = dimmer)
            Metallicity color-coding helps identify chemically distinct populations
        """
        # Validate input arrays
        if xx.shape != yy.shape or xx.shape != zz.shape:
            raise ValueError("Meshgrid arrays (xx, yy, zz) must have matching shapes")
        
        if len(x_cut) != len(y_cut) or len(x_cut) != len(feh_cut):
            raise ValueError("Cut stellar data arrays must have matching lengths")
        
        if len(x_orig) != len(y_orig) or len(x_orig) != len(feh_orig):
            raise ValueError("Original stellar data arrays must have matching lengths")

        # Create figure with side-by-side subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        fig.suptitle(f'Surface Brightness & Stellar Metallicity - {self.config.halo_name} - Rotation {rotation_idx}', 
                    fontsize=16)

        # Define plot configurations
        plot_configs = [
            {
                'ax': axes[0], 
                'x': x_cut, 
                'y': y_cut, 
                'feh': feh_cut,
                'title': f'After SB Cut (≤{self.config.SB_lim} mag/arcsec²)',
                'n_stars': len(x_cut)
            },
            {
                'ax': axes[1], 
                'x': x_orig, 
                'y': y_orig, 
                'feh': feh_orig,
                'title': 'All Original Stars',
                'n_stars': len(x_orig)
            }
        ]

        # Create plots for each configuration
        for config in plot_configs:
            ax = config['ax']
            
            # Plot surface brightness contours
            # Use levels that make sense for astronomical surface brightness
            contour_levels = np.arange(20, 35, 1)  # mag/arcsec² levels
            contour = ax.contour(xx, yy, zz, levels=contour_levels, 
                               colors='black', linewidths=0.8, alpha=0.6)
            
            # Add contour labels
            ax.clabel(contour, inline=True, fontsize=9, fmt='%.0f')
            
            # Scatter plot of stellar positions colored by metallicity
            if len(config['x']) > 0:  # Check for empty arrays
                scatter = ax.scatter(config['x'], config['y'], 
                                   s=8, marker='.', c=config['feh'], 
                                   cmap='plasma', alpha=0.7, edgecolors='none')
                
                # Add colorbar for metallicity
                cbar = fig.colorbar(scatter, ax=ax, label='[Fe/H]', shrink=0.8)
                cbar.set_label('[Fe/H]', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No stars in this selection', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='red')
            
            # Customize axes
            ax.set_title(f"{config['title']}\n({config['n_stars']} stars)", fontsize=14)
            ax.set_xlabel('X [kpc]', fontsize=12)
            ax.set_ylabel('Y [kpc]', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        # Add text box with analysis parameters
        info_text = (f"Parameters: e={config.get('best_params', [0,0,0])[0]:.3f}, "
                    f"θ={config.get('best_params', [0,0,0])[1]:.1f}°, "
                    f"rₕ={config.get('best_params', [0,0,0])[2]:.3f} kpc")
        
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
                style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        if save_plot:
            filename = f'contour_rot{rotation_idx:03d}_{self.config.halo_name}.png'
            filepath = self._output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Contour plot saved: {filepath}")
            return None
        else:
            return fig

    def plot_parameter_distributions(self, all_results: List[dict], 
                                   save_plot: bool = True) -> Optional[matplotlib.figure.Figure]:
        """Plot distributions of fitted parameters across all rotations.
        
        Creates histograms showing the distribution of best-fit parameters
        from multiple random rotations, helping assess morphological robustness.
        
        Args:
            all_results: List of result dictionaries containing 'best_params'
            save_plot: Whether to save plot to file (default: True)
            
        Returns:
            Figure object if save_plot=False, None otherwise
        """
        if not all_results:
            raise ValueError("No results provided for parameter distribution plot")
        
        # Extract parameters
        ellipticities = [result['best_params'][0] for result in all_results]
        position_angles = [result['best_params'][1] for result in all_results]
        half_light_radii = [result['best_params'][2] for result in all_results]
        
        # Create subplot grid
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Parameter Distributions Across Rotations - {self.config.halo_name}', 
                    fontsize=16)
        
        # Plot configurations
        params = [
            {'data': ellipticities, 'label': 'Ellipticity, $e$', 'ax': axes[0]},
            {'data': position_angles, 'label': 'Position Angle, $\\theta$ [°]', 'ax': axes[1]},
            {'data': half_light_radii, 'label': 'Half-light Radius, $r_h$ [kpc]', 'ax': axes[2]}
        ]
        
        for param in params:
            data = np.array(param['data'])
            ax = param['ax']
            
            # Create histogram
            ax.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', 
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.3f}')
            
            # Customize axes
            ax.set_xlabel(param['label'], fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'σ = {std_val:.3f}\nN = {len(data)}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        
        if save_plot:
            filename = f'parameter_distributions_{self.config.halo_name}.png'
            filepath = self._output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Parameter distribution plot saved: {filepath}")
            return None
        else:
            return fig

    def set_output_directory(self, directory: Union[str, Path]) -> None:
        """Set custom output directory for plots.
        
        Args:
            directory: Path to desired plot output directory
        """
        self._output_dir = Path(directory)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plot output directory set to: {self._output_dir.absolute()}")

    def get_output_directory(self) -> Path:
        """Get current plot output directory.
        
        Returns:
            Path object for plot output directory
        """
        return self._output_dir

    def close_all_figures(self) -> None:
        """Close all open matplotlib figures to free memory."""
        plt.close('all')