"""
Utility functions for galaxy morphology analysis.
Provides surface brightness calculations, data filtering, and coordinate transformations.
"""

from typing import Tuple, Union
import numpy as np

def apply_surface_brightness_cut(SB_func: np.ndarray, SB_limit: float, 
                                *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Apply surface brightness cut to filter stellar particles and associated data.
    
    Removes particles with surface brightness above the specified limit,
    effectively filtering out low surface brightness regions that may
    introduce noise in morphological fitting.
    
    Args:
        SB_func: Array of surface brightness values (mag/arcsec²)
        SB_limit: Maximum allowed surface brightness (mag/arcsec²)
                 Note: Higher values = dimmer (magnitude scale)
        *arrays: Variable number of arrays to filter (positions, masses, etc.)
                All arrays must have same length as SB_func
    
    Returns:
        Tuple of filtered arrays in the same order as input
        Arrays where SB_func <= SB_limit (brighter than limit)
        
    Raises:
        ValueError: If arrays have mismatched lengths
        
    Example:
        x_cut, y_cut, mass_cut = apply_surface_brightness_cut(
            SB_values, 28.0, x_coords, y_coords, stellar_masses
        )
        
    Note:
        Surface brightness uses magnitude scale where:
        - Lower values = brighter
        - Higher values = dimmer
        - Cut keeps particles with SB <= limit (brighter regions)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided to filter")
    
    # Validate array lengths
    reference_length = len(SB_func)
    for i, arr in enumerate(arrays):
        if arr is not None and len(arr) != reference_length:
            raise ValueError(f"Array {i} length ({len(arr)}) doesn't match SB_func length ({reference_length})")
    
    # Create brightness mask (keep particles brighter than limit)
    mask = SB_func <= SB_limit
    
    if not np.any(mask):
        raise ValueError(f"Surface brightness cut (limit={SB_limit}) removed all particles")
    
    # Apply mask to all input arrays
    filtered_arrays = []
    for arr in arrays:
        if arr is not None:
            filtered_arrays.append(arr[mask])
        else:
            filtered_arrays.append(None)
    
    n_original = len(SB_func)
    n_remaining = np.sum(mask)
    print(f"Surface brightness cut: kept {n_remaining}/{n_original} particles ({100*n_remaining/n_original:.1f}%)")
    
    return tuple(filtered_arrays)

def create_meshgrid(x: np.ndarray, y: np.ndarray, n_points: int = 2000, 
                   padding_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 2D coordinate meshgrid for surface brightness contour plotting.
    
    Generates a regular grid covering the stellar particle distribution
    with optional padding to ensure smooth contours at the edges.
    
    Args:
        x: Array of x-coordinates (kpc)
        y: Array of y-coordinates (kpc)
        n_points: Number of grid points per dimension (default: 2000)
        padding_fraction: Fraction of data range to add as padding (default: 0.1)
    
    Returns:
        Tuple of (xx, yy) meshgrid arrays for contour plotting
        
    Raises:
        ValueError: If input arrays are empty or have invalid parameters
        
    Note:
        - Uses 'ij' indexing convention: xx[i,j] gives x-coordinate at grid point (i,j)
        - Padding prevents edge effects in surface brightness interpolation
        - Higher n_points gives smoother contours but increases computation time
        
    Example:
        xx, yy = create_meshgrid(star_x, star_y, n_points=1000, padding_fraction=0.15)
        SB_grid = calculate_surface_brightness(xx, yy, e, theta, rh, vlum)
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input coordinate arrays cannot be empty")
    
    if n_points <= 0:
        raise ValueError("Number of grid points must be positive")
    
    if padding_fraction < 0:
        raise ValueError("Padding fraction cannot be negative")
    
    # Calculate data bounds
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Add padding to prevent edge effects
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Handle degenerate cases (all points at same coordinate)
    if x_range == 0:
        x_range = 0.1  # Minimum range in kpc
    if y_range == 0:
        y_range = 0.1
    
    dx = x_range * padding_fraction
    dy = y_range * padding_fraction

    # Create coordinate arrays
    grid_x = np.linspace(x_min - dx, x_max + dx, n_points)
    grid_y = np.linspace(y_min - dy, y_max + dy, n_points)

    # Generate meshgrid with matrix indexing (ij)
    # This ensures xx[i,j] corresponds to x-coordinate at row i, column j
    xx, yy = np.meshgrid(grid_x, grid_y, indexing='ij')
    
    return xx, yy

def calculate_surface_brightness(xx: np.ndarray, yy: np.ndarray, e: float, 
                               theta: float, rh: float, vlum: float) -> np.ndarray:
    """
    Calculate surface brightness on coordinate meshgrid for elliptical exponential profile.
    
    Implements the elliptical exponential surface brightness model:
    SB(r) = SB_0 - 2.5 * log10(I_0 * exp(-r/r_scale))
    
    where r is the elliptical radius accounting for galaxy orientation and shape.
    
    Args:
        xx: X-coordinate meshgrid (kpc)
        yy: Y-coordinate meshgrid (kpc)  
        e: Ellipticity [0, 1) where 0=circular, approaching 1=linear
        theta: Position angle (degrees, measured counter-clockwise from x-axis)
        rh: Half-light radius (kpc) - radius containing 50% of total light
        vlum: V-band luminosity (solar units)
    
    Returns:
        Surface brightness array (mag/arcsec²) on the input meshgrid
        
    Raises:
        ValueError: If parameters are outside valid ranges
        
    Notes:
        - Surface brightness in astronomical magnitudes (higher = dimmer)
        - Uses Sérsic n=1 (exponential) profile scaling
        - Zero-point of 26.39 mag/arcsec² for V-band solar surface brightness
        - Coordinate transformation handles galaxy rotation and ellipticity
        
    Mathematical Details:
        1. Rotate coordinates by position angle θ
        2. Apply ellipticity correction: x' = x_rot / (1-e)
        3. Calculate elliptical radius: r = sqrt(x'² + y_rot²)
        4. Apply exponential profile with proper normalization
        
    Example:
        SB = calculate_surface_brightness(xx, yy, e=0.3, theta=45, rh=2.5, vlum=1e10)
    """
    # Validate input parameters
    if not (0 <= e < 1):
        raise ValueError(f"Ellipticity must be in range [0, 1), got {e}")
    
    if rh <= 0:
        raise ValueError(f"Half-light radius must be positive, got {rh}")
    
    if vlum <= 0:
        raise ValueError(f"Luminosity must be positive, got {vlum}")
    
    # Convert angles and units
    theta_rad = np.radians(theta)  # Convert position angle to radians
    rh_pc = 1000 * rh  # Convert half-light radius from kpc to pc for normalization
    
    # Apply coordinate transformation for elliptical geometry
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
    
    # Rotation matrix application with ellipticity correction
    # x-component gets ellipticity correction: divide by (1-e) to stretch major axis
    x_rot = (xx * cos_theta - yy * sin_theta) / (1 - e)
    y_rot = xx * sin_theta + yy * cos_theta  # Standard rotation for y-component
    
    # Calculate elliptical radius
    r = np.sqrt(x_rot**2 + y_rot**2)
    
    # Exponential surface brightness profile
    # Scale factor converts half-light radius to exponential scale length
    # Factor 42/25 ≈ 1.68 comes from Sérsic n=1 profile: r_e = 1.678 * r_scale
    scale_factor = -42 / (25 * rh)  # Negative for exponential decay
    exponent = scale_factor * r
    
    # Surface brightness normalization
    # Accounts for total luminosity, profile geometry, and coordinate units
    # Factor 882/625 comes from integrating the 2D exponential profile
    normalization = (882 * vlum) / (625 * np.pi * rh_pc**2)
    
    # Convert to surface brightness in magnitudes
    # 26.39 is the V-band solar surface brightness zero-point (mag/arcsec²)
    # Formula: SB = zero_point - 2.5 * log10(intensity)
    surface_brightness = 26.39 - 2.5 * np.log10(normalization * np.exp(exponent))
    
    return surface_brightness

def elliptical_radius(x: np.ndarray, y: np.ndarray, e: float, theta: float) -> np.ndarray:
    """
    Calculate elliptical radius for given coordinates and shape parameters.
    
    Transforms Cartesian coordinates to elliptical radius accounting for
    galaxy orientation (position angle) and flattening (ellipticity).
    
    Args:
        x: X-coordinates (any units)
        y: Y-coordinates (same units as x)
        e: Ellipticity [0, 1) where 0=circular, approaching 1=linear  
        theta: Position angle (degrees, counter-clockwise from x-axis)
    
    Returns:
        Array of elliptical radii in same units as input coordinates
        
    Note:
        This is the same coordinate transformation used in surface brightness
        calculations, extracted as a utility function for reuse.
    """
    if not (0 <= e < 1):
        raise ValueError(f"Ellipticity must be in range [0, 1), got {e}")
    
    theta_rad = np.radians(theta)
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
    
    # Apply rotation and ellipticity transformation
    x_rot = (x * cos_theta - y * sin_theta) / (1 - e)
    y_rot = x * sin_theta + y * cos_theta
    
    return np.sqrt(x_rot**2 + y_rot**2)

def magnitude_to_flux(magnitude: Union[float, np.ndarray], 
                     zero_point: float = 0.0) -> Union[float, np.ndarray]:
    """
    Convert astronomical magnitudes to linear flux units.
    
    Uses the standard astronomical magnitude-flux relation:
    flux = 10^(-(magnitude - zero_point) / 2.5)
    
    Args:
        magnitude: Magnitude value(s) to convert
        zero_point: Magnitude zero-point (default: 0.0)
        
    Returns:
        Linear flux in units corresponding to the zero-point system
        
    Example:
        flux = magnitude_to_flux(magnitude=20.0, zero_point=26.39)  # V-band
    """
    return 10 ** (-(magnitude - zero_point) / 2.5)

def flux_to_magnitude(flux: Union[float, np.ndarray], 
                     zero_point: float = 0.0) -> Union[float, np.ndarray]:
    """
    Convert linear flux to astronomical magnitudes.
    
    Uses the standard astronomical flux-magnitude relation:
    magnitude = zero_point - 2.5 * log10(flux)
    
    Args:
        flux: Linear flux value(s) to convert (must be positive)
        zero_point: Magnitude zero-point (default: 0.0)
        
    Returns:
        Magnitude values in the specified system
        
    Raises:
        ValueError: If flux contains non-positive values
    """
    flux = np.asarray(flux)
    if np.any(flux <= 0):
        raise ValueError("Flux values must be positive for magnitude conversion")
    
    return zero_point - 2.5 * np.log10(flux)