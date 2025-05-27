"""
Handler for generating and applying 3D rotation matrices.
Uses Rodrigues' rotation formula to create random rotations for galaxy orientation analysis.
"""

import numpy as np
from numpy.linalg import norm
from config import Config

class RotationHandler:
    """Handles generation and application of 3D rotation matrices.
    
    Generates uniformly distributed random rotations on the unit sphere
    and applies them to stellar particle positions for morphology analysis.
    """
    
    def __init__(self, config: Config):
        """Initialize rotation handler with configuration.
        
        Args:
            config: Configuration object containing number of rotations (N)
        """
        self.config = config
    
    def generate_random_rotations(self) -> np.ndarray:
        """Generate N random rotation matrices using uniform sphere sampling.
        
        Creates rotation matrices that rotate from the initial direction [1,0,0]
        to uniformly distributed points on the unit sphere. This ensures
        unbiased sampling of galaxy orientations.
        
        Algorithm:
        1. Generate uniform random points on sphere using spherical coordinates
        2. For each target direction, compute rotation matrix using Rodrigues' formula
        3. Handle special cases (parallel/antiparallel vectors)
        
        Returns:
            Array of shape (N, 3, 3) containing rotation matrices
            
        Note:
            Uses Marsaglia's method for uniform sphere sampling:
            - θ uniform in [0, 2π] for azimuthal angle
            - φ = arccos(1 - 2u) where u ~ U(0,1) for polar angle
            This avoids clustering at poles that naive sampling would produce.
        """
        N = self.config.N
        
        # Generate uniform random points on unit sphere
        # Azimuthal angle: uniform distribution
        theta = 2 * np.pi * np.random.rand(N)
        # Polar angle: corrected for uniform surface distribution
        phi = np.arccos(1 - 2 * np.random.rand(N))

        # Convert spherical to Cartesian coordinates
        x = np.cos(theta) * np.sin(phi)  # x-component
        y = np.sin(theta) * np.sin(phi)  # y-component  
        z = np.cos(phi)                  # z-component
        
        # Stack target vectors as N × 3 array
        targets = np.vstack((x, y, z)).T  

        # Reference direction: unit vector along x-axis
        # All rotations will rotate from this direction to target directions
        pos0 = np.array([1.0, 0.0, 0.0])
        
        Rmatrices = []
        
        for target in targets:
            R = self._compute_rotation_matrix(pos0, target)
            Rmatrices.append(R)

        return np.array(Rmatrices)
    
    def _compute_rotation_matrix(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from source to target vector using Rodrigues' formula.
        
        Args:
            source: Source unit vector (3D)
            target: Target unit vector (3D)
            
        Returns:
            3×3 rotation matrix that rotates source to target
            
        Note:
            Rodrigues' rotation formula: R = I + [v]× + [v]×² * (1-cos(θ))/sin²(θ)
            where [v]× is the skew-symmetric matrix of the rotation axis v = source × target
        """
        # Ensure target is normalized
        target = target / norm(target)
        
        # Compute rotation axis (cross product) and angle components
        v = np.cross(source, target)  # Rotation axis
        s = norm(v)                   # sin(angle) 
        c = np.dot(source, target)    # cos(angle)
        
        # Handle special cases
        if s < 1e-10:  # Vectors are parallel or antiparallel
            if c > 0:
                # Same direction: identity matrix
                return np.eye(3)
            else:
                # Opposite direction: 180° rotation
                # Find perpendicular vector for rotation axis
                if abs(source[0]) < 0.9:
                    perp = np.array([1.0, 0.0, 0.0])
                else:
                    perp = np.array([0.0, 1.0, 0.0])
                
                # Create perpendicular vector
                v = np.cross(source, perp)
                v = v / norm(v)
                
                # 180° rotation matrix: R = 2vv^T - I
                return 2 * np.outer(v, v) - np.eye(3)
        else:
            # General case: use Rodrigues' formula
            # Create skew-symmetric matrix [v]×
            vx = np.array([
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0]
            ])
            
            # Rodrigues' rotation formula: R = I + [v]× + [v]×² * (1-cos(θ))/sin²(θ)
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
            
            return R
    
    @staticmethod
    def apply_rotation(starpos: np.ndarray, Rmatrix: np.ndarray) -> np.ndarray:
        """Apply 3×3 rotation matrix to stellar particle positions.
        
        Transforms stellar coordinates using matrix multiplication.
        Uses transpose for proper right-multiplication: pos_new = pos_old @ R^T
        
        Args:
            starpos: Array of stellar positions, shape (N_stars, 3)
            Rmatrix: 3×3 rotation matrix
            
        Returns:
            Rotated stellar positions, same shape as input
            
        Raises:
            ValueError: If input arrays have incompatible shapes
            
        Note:
            For rotation matrix R and position vector p:
            - Mathematical rotation: p' = R @ p (column vector)
            - NumPy implementation: p' = p @ R^T (row vectors)
            The transpose accounts for NumPy's row-vector convention.
        """
        starpos = np.asarray(starpos)
        
        # Validate input shapes
        if starpos.ndim != 2 or starpos.shape[1] != 3:
            raise ValueError(f"starpos must be shape (N, 3), got {starpos.shape}")
        
        if Rmatrix.shape != (3, 3):
            raise ValueError(f"Rmatrix must be shape (3, 3), got {Rmatrix.shape}")
        
        # Apply rotation: each row is transformed by R^T
        # This is equivalent to: result[i] = R @ starpos[i] for column vectors
        return starpos @ Rmatrix.T
    
    def validate_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Validate that a matrix is a proper rotation matrix.
        
        Checks that:
        1. R is orthogonal: R @ R^T = I
        2. R has determinant +1 (proper rotation, not reflection)
        
        Args:
            R: 3×3 matrix to validate
            tolerance: Numerical tolerance for checks
            
        Returns:
            True if R is a valid rotation matrix
        """
        if R.shape != (3, 3):
            return False
            
        # Check orthogonality: R @ R^T should equal identity
        should_be_identity = R @ R.T
        if not np.allclose(should_be_identity, np.eye(3), atol=tolerance):
            return False
            
        # Check determinant equals +1 (proper rotation, not reflection)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False
            
        return True
    
    def rotation_angle(self, R: np.ndarray) -> float:
        """Calculate the rotation angle from a rotation matrix.
        
        Args:
            R: 3×3 rotation matrix
            
        Returns:
            Rotation angle in radians [0, π]
            
        Note:
            Uses the trace formula: cos(θ) = (trace(R) - 1) / 2
        """
        trace = np.trace(R)
        cos_angle = (trace - 1) / 2
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)