"""
Base simulator class for data generation.

This module provides an abstract base class for all scenario-specific simulators,
ensuring a consistent interface and common functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import json

from src.utils.logger import setup_logger
from src.utils.helpers import ensure_path, set_random_seed


class BaseSimulator(ABC):
    """
    Abstract base class for data simulators.
    
    All scenario-specific simulators should inherit from this class and
    implement the abstract methods.
    
    Parameters
    ----------
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    **kwargs
        Additional parameters specific to the simulator
        
    Attributes
    ----------
    logger : logging.Logger
        Logger instance for this simulator
    seed : Optional[int]
        Random seed used for generation
    data : Optional[Dict[str, Any]]
        Generated data (None until generate() is called)
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize base simulator."""
        self.logger = setup_logger(self.__class__.__name__)
        self.seed = seed
        self.data: Optional[Dict[str, Any]] = None
        self.params = kwargs
        
        if seed is not None:
            set_random_seed(seed)
            self.logger.info(f"Random seed set to {seed}")
    
    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate simulated data.
        
        This method must be implemented by subclasses.
        
        Parameters
        ----------
        **kwargs
            Parameters specific to the data generation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing generated data and metadata
        """
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated data.
        
        This method must be implemented by subclasses to verify
        that generated data meets expected criteria.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate
            
        Returns
        -------
        bool
            True if data is valid
            
        Raises
        ------
        ValueError
            If data is invalid
        """
        pass
    
    def save(
        self,
        output_dir: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> Path:
        """
        Save generated data to disk.
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directory to save data
        filename_prefix : Optional[str], optional
            Prefix for output filenames, by default None
            
        Returns
        -------
        Path
            Path to saved data directory
            
        Raises
        ------
        RuntimeError
            If no data has been generated yet
        """
        if self.data is None:
            raise RuntimeError("No data to save. Call generate() first.")
        
        output_dir = ensure_path(output_dir, create=True)
        
        if filename_prefix is None:
            filename_prefix = self.__class__.__name__.lower()
        
        # Save numpy arrays
        arrays_to_save = {}
        metadata = {}
        
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                arrays_to_save[key] = value
            else:
                metadata[key] = value
        
        # Save arrays as .npz
        if arrays_to_save:
            arrays_path = output_dir / f"{filename_prefix}_data.npz"
            np.savez_compressed(arrays_path, **arrays_to_save)
            self.logger.info(f"Saved arrays to {arrays_path}")
        
        # Save metadata as JSON
        if metadata:
            # Add generation info
            metadata['seed'] = self.seed
            metadata['simulator'] = self.__class__.__name__
            
            metadata_path = output_dir / f"{filename_prefix}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"Saved metadata to {metadata_path}")
        
        return output_dir
    
    @classmethod
    def load(
        cls,
        data_dir: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load previously saved data.
        
        Parameters
        ----------
        data_dir : Union[str, Path]
            Directory containing saved data
        filename_prefix : Optional[str], optional
            Prefix of saved files, by default None
            
        Returns
        -------
        Dict[str, Any]
            Loaded data dictionary
            
        Raises
        ------
        FileNotFoundError
            If data files are not found
        """
        data_dir = Path(data_dir)
        
        if filename_prefix is None:
            filename_prefix = cls.__name__.lower()
        
        data = {}
        
        # Load arrays
        arrays_path = data_dir / f"{filename_prefix}_data.npz"
        if arrays_path.exists():
            loaded = np.load(arrays_path)
            data.update({key: loaded[key] for key in loaded.files})
        
        # Load metadata
        metadata_path = data_dir / f"{filename_prefix}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                data.update(metadata)
        
        if not data:
            raise FileNotFoundError(
                f"No data files found in {data_dir} with prefix {filename_prefix}"
            )
        
        return data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of generated data.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics
            
        Raises
        ------
        RuntimeError
            If no data has been generated yet
        """
        if self.data is None:
            raise RuntimeError("No data to summarize. Call generate() first.")
        
        summary = {
            'simulator': self.__class__.__name__,
            'seed': self.seed,
            'arrays': {}
        }
        
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                summary['arrays'][key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'min': float(np.min(value)),
                    'max': float(np.max(value)),
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value))
                }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation."""
        status = "with data" if self.data is not None else "no data"
        return f"{self.__class__.__name__}(seed={self.seed}, {status})"
