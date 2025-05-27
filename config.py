"""
Configuration settings for galaxy morphology analysis using MCMC.
Contains parameters for initial guesses, MCMC sampling, and file paths.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    # Physical and model parameters
    e_initial: float = 0.5       # Initial guess for ellipticity
    theta_initial: float = 40    # Initial guess for position angle (degrees)
    rh_initial: float = 0.2      # Initial guess for half-light radius (kpc)

    # MCMC parameters
    nwalkers: int = 50           # Number of MCMC walkers (parallel chains)
    burn_in_steps: int = 100     # Steps for MCMC burn-in phase
    production_steps: int = 500  # Steps for MCMC production phase

    # Additional simulation settings
    SB_lim: float = 30           # Surface brightness cut limit (mag/arcsec²)
    N: int = 50                  # Number of random rotations to perform
    random_seed: int = 42        # Seed for reproducibility
    metallicity_threshold: float = 1e-10  # Minimum metallicity for stellar particle selection

    # File paths - to be set by user
    snapshot_path: str = ''      # Path to simulation snapshot (must be set by user)
    halo_name: str = ''          # Name of halo in Tangos database (must be set by user)

    def initialize(self) -> None:
        """Initialize the configuration with the random seed.
        Note: This ensures reproducibility of random number generation.
        """
        np.random.seed(self.random_seed)

    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameters are invalid
        """
        if not self.snapshot_path:
            raise ValueError("snapshot_path must be set")
        
        if not self.halo_name:
            raise ValueError("halo_name must be set")
        
        if self.e_initial < 0 or self.e_initial >= 1:
            raise ValueError("e_initial must be between 0 and 1 (exclusive)")
        
        if self.theta_initial < 0 or self.theta_initial >= 180:
            raise ValueError("theta_initial must be between 0 and 180 degrees")
        
        if self.rh_initial <= 0:
            raise ValueError("rh_initial must be positive")
        
        if self.nwalkers <= 0:
            raise ValueError("nwalkers must be positive")
        
        if self.burn_in_steps < 0:
            raise ValueError("burn_in_steps must be non-negative")
        
        if self.production_steps <= 0:
            raise ValueError("production_steps must be positive")
        
        if self.N <= 0:
            raise ValueError("N (number of rotations) must be positive")
        
        if self.metallicity_threshold < 0:
            raise ValueError("metallicity_threshold must be non-negative")

# Helper function to initialize configuration
def initialize_config(config: Config) -> None:
    """Initialize configurations like random seed.
    This wrapper function provides a clean interface for initialization.
    
    Args:
        config: Configuration object to initialize
    """
    config.initialize()

def create_config(snapshot_path: str, halo_name: str, **kwargs) -> Config:
    """Create and validate a configuration object.
    
    Args:
        snapshot_path: Path to simulation snapshot file
        halo_name: Name of halo in Tangos database
        **kwargs: Additional configuration parameters to override defaults
        
    Returns:
        Validated Config object
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    config = Config(snapshot_path=snapshot_path, halo_name=halo_name, **kwargs)
    config.validate()
    config.initialize()
    return config