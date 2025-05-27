"""
MCMC model for fitting galaxy surface brightness profiles.
Uses emcee for Bayesian parameter estimation of elliptical exponential profiles.
"""

from typing import Tuple, List, Optional
import numpy as np
import emcee
from config import Config

class MCMCModel:
    """MCMC fitting class for galaxy surface brightness models.
    
    Implements an elliptical exponential surface brightness profile with
    Bayesian parameter estimation using the emcee ensemble sampler.
    """
    
    def __init__(self, config: Config):
        """Initialize MCMC model with configuration.
        
        Args:
            config: Configuration object containing MCMC parameters
        """
        self.config = config

    def model(self, params: List[float], x: np.ndarray, y: np.ndarray, 
              mstar: np.ndarray, Nstar: int, stellar_mass: float) -> np.ndarray:
        """Calculate surface brightness model values (log-likelihood terms).
        
        Implements an elliptical exponential surface brightness profile:
        I(r) = I0 * exp(-r/rh) where r is the elliptical radius.
        
        Args:
            params: Model parameters [ellipticity, position_angle_deg, half_light_radius]
            x: X coordinates of stellar particles (kpc)
            y: Y coordinates of stellar particles (kpc)
            mstar: Stellar masses (Msun)
            Nstar: Total number of stellar particles
            stellar_mass: Total stellar mass (Msun)
            
        Returns:
            Array of log surface brightness values for each particle
            
        Note:
            - Ellipticity e ranges from 0 (circular) to 1 (linear)
            - Position angle theta is measured counter-clockwise from x-axis
            - Exponential scale length is derived from half-light radius
        """
        e, theta_deg, rh = params
        theta = np.radians(theta_deg)  # Convert to radians for trigonometry

        # Apply coordinate transformation for elliptical geometry
        # Rotation and ellipticity correction in one step
        x_rot = (x * np.cos(theta) - y * np.sin(theta)) / (1 - e)  # Stretch along major axis
        y_rot = x * np.sin(theta) + y * np.cos(theta)  # Standard rotation

        # Calculate elliptical radius
        r = np.sqrt(x_rot**2 + y_rot**2)

        # Exponential surface brightness profile
        # Scale factor: converts half-light radius to exponential scale length
        scale_factor = -42 / (25 * rh)  # Derived from Sersic n=1 profile
        exponent = scale_factor * r

        # Normalization constant ensuring proper mass weighting
        # Factor accounts for: individual particle masses, geometry, and total mass
        # Note: mstar is per-particle, so each particle gets its own normalization
        norm_factor = (882 * mstar * Nstar) / (625 * np.pi * rh**2 * stellar_mass * (1 - e))
        
        # Return log surface brightness (log-likelihood contribution)
        return np.log(norm_factor) + exponent

    def lnprior(self, params: List[float]) -> float:
        """Calculate log-prior probability for model parameters.
        
        Implements uniform priors with physical constraints:
        - Ellipticity: 0 ≤ e < 1 (0=circular, 1=linear)
        - Position angle: -90° ≤ θ < 90° (avoid degeneracy)
        - Half-light radius: rh > 0 (must be positive)
        
        Args:
            params: Model parameters [ellipticity, position_angle_deg, half_light_radius]
            
        Returns:
            Log-prior probability (0.0 if valid, -inf if invalid)
        """
        e, theta, rh = params
        
        # Check physical parameter bounds
        if 0 <= e < 1 and -90 <= theta < 90 and rh > 0:
            return 0.0  # Uniform prior within bounds
        return -np.inf  # Reject parameters outside bounds

    def lnlike(self, params: List[float], x: np.ndarray, y: np.ndarray,
               mstar: np.ndarray, Nstar: int, stellar_mass: float) -> float:
        """Calculate log-likelihood function.
        
        Sums log surface brightness values across all stellar particles.
        Assumes Poisson statistics for stellar particle counts.
        
        Args:
            params: Model parameters [ellipticity, position_angle_deg, half_light_radius]
            x: X coordinates of stellar particles (kpc)
            y: Y coordinates of stellar particles (kpc)
            mstar: Stellar masses (Msun)
            Nstar: Total number of stellar particles
            stellar_mass: Total stellar mass (Msun)
            
        Returns:
            Total log-likelihood value
        """
        model_vals = self.model(params, x, y, mstar, Nstar, stellar_mass)
        return np.sum(model_vals)

    def lnprob(self, params: List[float], x: np.ndarray, y: np.ndarray,
               mstar: np.ndarray, Nstar: int, stellar_mass: float) -> float:
        """Calculate log-posterior probability (prior × likelihood).
        
        Combines prior and likelihood using Bayes' theorem:
        P(params|data) ∝ P(data|params) × P(params)
        
        Args:
            params: Model parameters [ellipticity, position_angle_deg, half_light_radius]
            x: X coordinates of stellar particles (kpc)
            y: Y coordinates of stellar particles (kpc)
            mstar: Stellar masses (Msun)
            Nstar: Total number of stellar particles
            stellar_mass: Total stellar mass (Msun)
            
        Returns:
            Log-posterior probability
        """
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf  # Reject if prior is invalid
        
        # Add likelihood to prior (log space = multiplication in linear space)
        return lp + self.lnlike(params, x, y, mstar, Nstar, stellar_mass)

    def run_mcmc(self, initial_params: List[float], 
                 data: Tuple[np.ndarray, np.ndarray, np.ndarray, int, float],
                 verbose: bool = True) -> Optional[emcee.EnsembleSampler]:
        """Run MCMC sampling using emcee ensemble sampler.
        
        Performs two-phase sampling:
        1. Burn-in phase: allows walkers to reach equilibrium distribution
        2. Production phase: samples from equilibrium for parameter estimation
        
        Args:
            initial_params: Starting parameter values [e, theta, rh]
            data: Tuple of (x, y, mstar, Nstar, stellar_mass)
            verbose: Whether to print progress information
            
        Returns:
            Configured emcee sampler with completed chains
            None if sampling fails
            
        Raises:
            RuntimeError: If MCMC sampling encounters numerical issues
        """
        try:
            x, y, mstar, Nstar, stellar_mass = data
            ndim = len(initial_params)

            # Initialize walker positions near initial guess
            # Small perturbations prevent walkers from starting identically
            perturbation = 1e-3 * np.random.randn(self.config.nwalkers, ndim)
            p0 = np.array(initial_params) + perturbation
            
            # Ensure all initial positions satisfy priors
            for i in range(self.config.nwalkers):
                while self.lnprior(p0[i]) == -np.inf:
                    p0[i] = np.array(initial_params) + 1e-3 * np.random.randn(ndim)

            # Create ensemble sampler
            sampler = emcee.EnsembleSampler(
                self.config.nwalkers,
                ndim,
                self.lnprob,
                args=(x, y, mstar, Nstar, stellar_mass)
            )

            # Burn-in phase: equilibration
            if verbose:
                print(f"Running burn-in for {self.config.burn_in_steps} steps...")
            sampler.run_mcmc(p0, self.config.burn_in_steps, progress=verbose)
            
            # Reset sampler to discard burn-in samples
            sampler.reset()

            # Production phase: parameter estimation
            if verbose:
                print(f"Running production for {self.config.production_steps} steps...")
            sampler.run_mcmc(p0, self.config.production_steps, progress=verbose)

            # Check convergence (basic diagnostic)
            if verbose:
                acceptance_fraction = np.mean(sampler.acceptance_fraction)
                print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
                if acceptance_fraction < 0.2 or acceptance_fraction > 0.8:
                    print("Warning: Acceptance fraction outside optimal range (0.2-0.8)")

            return sampler
            
        except Exception as ex:
            print(f"MCMC sampling failed: {ex}")
            return None

    def get_best_parameters(self, sampler: emcee.EnsembleSampler) -> np.ndarray:
        """Extract parameters with highest posterior probability.
        
        Finds the sample with maximum log-posterior from the production phase.
        This gives the maximum a posteriori (MAP) estimate.
        
        Args:
            sampler: Completed emcee sampler object
            
        Returns:
            Array of best-fit parameters [ellipticity, position_angle, half_light_radius]
            
        Raises:
            ValueError: If sampler contains no valid samples
        """
        if sampler.flatchain.shape[0] == 0:
            raise ValueError("Sampler contains no samples")
            
        # Find sample with highest log-posterior probability
        flat_samples = sampler.flatchain
        flat_lnprob = sampler.flatlnprobability
        
        if len(flat_lnprob) == 0 or not np.any(np.isfinite(flat_lnprob)):
            raise ValueError("No finite log-probability values found")
            
        best_idx = np.argmax(flat_lnprob)
        best_params = flat_samples[best_idx]
        
        return best_params

    def get_parameter_statistics(self, sampler: emcee.EnsembleSampler) -> dict:
        """Calculate parameter statistics from MCMC chains.
        
        Computes median values and confidence intervals for each parameter.
        
        Args:
            sampler: Completed emcee sampler object
            
        Returns:
            Dictionary containing parameter statistics:
            - 'medians': Median values for each parameter
            - 'uncertainties': 16th-84th percentile uncertainties
            - 'percentiles': Full percentile arrays (16th, 50th, 84th)
        """
        flat_samples = sampler.flatchain
        
        # Calculate percentiles for each parameter (16%, 50%, 84% ≈ ±1σ for Gaussian)
        percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        medians = percentiles[1]  # 50th percentile
        
        # Uncertainty as average of upper and lower error bars
        uncertainties = np.mean([percentiles[2] - medians, medians - percentiles[0]], axis=0)
        
        return {
            'medians': medians,
            'uncertainties': uncertainties,
            'percentiles': percentiles
        }