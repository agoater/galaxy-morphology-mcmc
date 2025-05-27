import pynbody
import tangos
from typing import Tuple, Dict, Any
from config import Config

class DataLoader:
    def __init__(self, config: Config):
        self.config = config  # Store configuration for later use
    
    def load_simulation_data(self) -> Tuple[Any, Any]:
        """Load simulation data and center it on the halo.
        Uses pynbody for snapshot loading and tangos for halo properties.
        Note: The centering operation is done in-place for memory efficiency.
        
        Returns:
            Tuple of (simulation, halo) objects
            
        Raises:
            FileNotFoundError: If snapshot file doesn't exist
            ValueError: If halo name isn't found in tangos database
        """
        try:
            # Load snapshot and convert to physical units
            # maxlevel=0 prevents loading full hierarchy for memory efficiency
            s = pynbody.load(self.config.snapshot_path, maxlevel=0)
            s.physical_units()  # Convert to physical units (kpc, Msun, etc.)
        except (FileNotFoundError, IOError) as ex:
            raise FileNotFoundError(f"Could not load snapshot from {self.config.snapshot_path}: {ex}")
        
        try:
            # Get the halo data and center the simulation
            # Using tangos to query halo properties from database
            halo = tangos.get_halo(self.config.halo_name)
            if halo is None:
                raise ValueError(f"Halo '{self.config.halo_name}' not found in tangos database")
            
            # Centering operation (vectorized for performance)
            s['pos'] -= halo['shrink_center']
        except Exception as ex:
            raise ValueError(f"Error accessing halo data: {ex}")
        
        return s, halo
    
    def extract_stellar_data(self, s, halo) -> Dict[str, Any]:
        """Extract stellar data within r200c with nonzero metallicity.
        Uses boolean masking for efficient data selection.
        Returns a dictionary for easy data access.
        
        Args:
            s: Simulation object from pynbody
            halo: Halo object from tangos
            
        Returns:
            Dictionary containing stellar data arrays and halo properties
            
        Raises:
            ValueError: If no stellar particles meet the selection criteria
        """
        stars = s.s  # Access stellar particles
        r = stars['r']  # Radial distances
        metal = stars['metal']  # Metallicity values
        
        # Create combined mask using vectorized operations
        # Note: Using & for element-wise AND is more efficient than nested ifs
        mask = (r <= halo['r200c']) & (metal > self.config.metallicity_threshold)
        
        # Check if any particles meet criteria
        if not mask.any():
            raise ValueError("No stellar particles found meeting selection criteria (within r200c and above metallicity threshold)")
        
        # Boolean indexing for fast data extraction
        # This avoids creating intermediate arrays
        starpos = stars['pos'][mask]
        mstar = stars['mass'][mask]
        met = stars['metal'][mask]
        
        # Extract other halo properties
        # These are scalar values from the halo database
        feh = halo['iron_hydrogen_ratios']
        oxh = halo['oxygen_iron_ratios']
        vmag = halo['stellar_V_mag']
        
        # Return as dictionary for clear data organization
        return {
            'starpos': starpos,  # 3D positions of stars
            'mstar': mstar,      # Stellar masses
            'met': met,          # Metallicity values
            'feh': feh,          # Iron/hydrogen ratio
            'oxh': oxh,          # Oxygen/iron ratio
            'vmag': vmag         # V-band magnitude
        }