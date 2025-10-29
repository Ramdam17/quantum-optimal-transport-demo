"""
Hyperscanning data simulator for brain activity in dyadic interactions.

This module simulates correlated brain activity patterns between two subjects
during social interactions, as measured in hyperscanning experiments.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.data.base_simulator import BaseSimulator
from src.utils.helpers import validate_array


class HyperscanningSimulator(BaseSimulator):
    """
    Simulate brain activity data for hyperscanning experiments.
    
    This simulator generates time series of brain activity for two subjects
    with controllable inter-subject synchrony patterns.
    
    Parameters
    ----------
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    **kwargs
        Additional simulator parameters
        
    Examples
    --------
    >>> simulator = HyperscanningSimulator(seed=42)
    >>> data = simulator.generate(n_subjects=2, n_regions=20, n_timepoints=500)
    >>> print(data['subject1'].shape)
    (500, 20)
    """
    
    def generate(
        self,
        n_subjects: int = 2,
        n_regions: int = 20,
        n_timepoints: int = 500,
        baseline_activity: float = 0.5,
        noise_level: float = 0.1,
        synchrony_level: float = 0.6,
        synchrony_regions: Optional[List[int]] = None,
        autocorrelation: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate simulated hyperscanning data.
        
        Parameters
        ----------
        n_subjects : int, optional
            Number of subjects (typically 2 for dyads), by default 2
        n_regions : int, optional
            Number of brain regions (ROIs), by default 20
        n_timepoints : int, optional
            Length of time series, by default 500
        baseline_activity : float, optional
            Mean baseline activity level, by default 0.5
        noise_level : float, optional
            Gaussian noise standard deviation, by default 0.1
        synchrony_level : float, optional
            Inter-subject correlation (0-1), by default 0.6
        synchrony_regions : Optional[List[int]], optional
            Indices of regions with high synchrony, by default None (first 5)
        autocorrelation : float, optional
            Temporal autocorrelation (0-1), by default 0.8
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'subject1': Brain activity array (n_timepoints, n_regions)
            - 'subject2': Brain activity array (n_timepoints, n_regions)
            - 'synchrony_matrix': Region-wise synchrony (n_regions, n_regions)
            - Parameters used for generation
        """
        self.logger.info(
            f"Generating hyperscanning data: {n_subjects} subjects, "
            f"{n_regions} regions, {n_timepoints} timepoints"
        )
        
        # Default synchrony regions (first 5)
        if synchrony_regions is None:
            synchrony_regions = list(range(min(5, n_regions)))
        
        # Generate base activity with temporal autocorrelation
        subject_data = []
        
        for subject_idx in range(n_subjects):
            # Initialize with random activity
            activity = np.zeros((n_timepoints, n_regions))
            activity[0, :] = np.random.randn(n_regions) * noise_level + baseline_activity
            
            # Generate temporally correlated activity
            for t in range(1, n_timepoints):
                # AR(1) process for temporal autocorrelation
                activity[t, :] = (
                    autocorrelation * activity[t-1, :] +
                    np.sqrt(1 - autocorrelation**2) * np.random.randn(n_regions) * noise_level +
                    baseline_activity * (1 - autocorrelation)
                )
            
            subject_data.append(activity)
        
        # Add inter-subject synchrony for specific regions
        if n_subjects == 2 and synchrony_level > 0:
            shared_component = np.zeros((n_timepoints, n_regions))
            
            # Generate shared signal with autocorrelation
            shared_component[0, :] = np.random.randn(n_regions) * noise_level
            for t in range(1, n_timepoints):
                shared_component[t, :] = (
                    autocorrelation * shared_component[t-1, :] +
                    np.sqrt(1 - autocorrelation**2) * np.random.randn(n_regions) * noise_level
                )
            
            # Add shared component to synchrony regions
            for region_idx in synchrony_regions:
                for subject_idx in range(n_subjects):
                    subject_data[subject_idx][:, region_idx] = (
                        (1 - synchrony_level) * subject_data[subject_idx][:, region_idx] +
                        synchrony_level * shared_component[:, region_idx]
                    )
        
        # Compute synchrony matrix (correlation between regions across subjects)
        if n_subjects == 2:
            synchrony_matrix = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                for j in range(n_regions):
                    synchrony_matrix[i, j] = np.corrcoef(
                        subject_data[0][:, i],
                        subject_data[1][:, j]
                    )[0, 1]
        else:
            synchrony_matrix = None
        
        # Store data
        self.data = {
            'subject1': subject_data[0],
            'subject2': subject_data[1] if n_subjects > 1 else None,
            'synchrony_matrix': synchrony_matrix,
            'n_subjects': n_subjects,
            'n_regions': n_regions,
            'n_timepoints': n_timepoints,
            'baseline_activity': baseline_activity,
            'noise_level': noise_level,
            'synchrony_level': synchrony_level,
            'synchrony_regions': synchrony_regions,
            'autocorrelation': autocorrelation
        }
        
        # Validate
        self.validate(self.data)
        
        self.logger.info("Hyperscanning data generated successfully")
        return self.data
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated hyperscanning data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate
            
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        # Check subject 1 data
        validate_array(
            data['subject1'],
            expected_ndim=2,
            name="subject1"
        )
        
        # Check subject 2 data if present
        if data.get('subject2') is not None:
            validate_array(
                data['subject2'],
                expected_ndim=2,
                name="subject2"
            )
            
            # Check shapes match
            if data['subject1'].shape != data['subject2'].shape:
                raise ValueError(
                    f"Subject data shapes don't match: "
                    f"{data['subject1'].shape} vs {data['subject2'].shape}"
                )
        
        # Check synchrony matrix if present
        if data.get('synchrony_matrix') is not None:
            validate_array(
                data['synchrony_matrix'],
                expected_ndim=2,
                name="synchrony_matrix"
            )
            
            # Check synchrony matrix is square
            n_regions = data['synchrony_matrix'].shape[0]
            if data['synchrony_matrix'].shape != (n_regions, n_regions):
                raise ValueError(
                    f"Synchrony matrix must be square, got {data['synchrony_matrix'].shape}"
                )
        
        self.logger.debug("Data validation passed")
        return True
