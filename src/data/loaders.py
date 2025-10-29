"""
Data loading utilities.

This module provides functions to load and manage simulated data
for different scenarios.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import importlib

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def get_simulator(scenario_name: str):
    """
    Get simulator class for a given scenario.
    
    Parameters
    ----------
    scenario_name : str
        Scenario name ('hyperscanning', 'llm_alignment', or 'genetics')
        
    Returns
    -------
    class
        Simulator class
        
    Raises
    ------
    ValueError
        If scenario name is invalid
        
    Examples
    --------
    >>> SimulatorClass = get_simulator('hyperscanning')
    >>> simulator = SimulatorClass(seed=42)
    """
    scenario_map = {
        'hyperscanning': 'src.data.hyperscanning_simulator.HyperscanningSimulator',
        'llm_alignment': 'src.data.llm_simulator.LLMAlignmentSimulator',
        'genetics': 'src.data.genetics_simulator.GeneticsSimulator'
    }
    
    if scenario_name not in scenario_map:
        raise ValueError(
            f"Unknown scenario: {scenario_name}. "
            f"Valid options: {list(scenario_map.keys())}"
        )
    
    module_path, class_name = scenario_map[scenario_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    simulator_class = getattr(module, class_name)
    
    logger.info(f"Loaded simulator for scenario: {scenario_name}")
    return simulator_class


def load_data_from_config(
    config_path: Union[str, Path],
    force_regenerate: bool = False,
    save_data: bool = True,
    data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load or generate data based on configuration file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to scenario configuration file
    force_regenerate : bool, optional
        Force regeneration even if data exists, by default False
    save_data : bool, optional
        Save generated data to disk, by default True
    data_dir : Optional[Union[str, Path]], optional
        Directory to save/load data, by default None (uses data/)
        
    Returns
    -------
    Dict[str, Any]
        Generated or loaded data
        
    Examples
    --------
    >>> data = load_data_from_config('config/scenario_hyperscanning.yaml')
    >>> print(data['subject1'].shape)
    (500, 20)
    """
    # Load configuration
    config = ConfigLoader(config_path)
    scenario_name = config.get('scenario.name')
    
    logger.info(f"Loading data for scenario: {scenario_name}")
    
    # Get simulator class
    SimulatorClass = get_simulator(scenario_name)
    
    # Determine data directory
    if data_dir is None:
        data_dir = Path("data") / scenario_name
    else:
        data_dir = Path(data_dir)
    
    # Try to load existing data if not forcing regeneration
    if not force_regenerate and data_dir.exists():
        try:
            data = SimulatorClass.load(data_dir)
            logger.info(f"Loaded existing data from {data_dir}")
            return data
        except FileNotFoundError:
            logger.info("No existing data found, generating new data")
    
    # Generate new data
    data_params = config.get_section('data')
    seed = data_params.get('seed', config.get('random_seed', 42))
    
    simulator = SimulatorClass(seed=seed)
    data = simulator.generate(**data_params)
    
    # Save data if requested
    if save_data:
        simulator.save(data_dir)
        logger.info(f"Saved data to {data_dir}")
    
    return data


class DataLoader:
    """
    Data loader class for managing scenario data.
    
    This class provides a convenient interface for loading data
    from different scenarios with caching support.
    
    Parameters
    ----------
    config_dir : Union[str, Path], optional
        Directory containing configuration files, by default "config"
    data_dir : Union[str, Path], optional
        Directory for data storage, by default "data"
    cache_data : bool, optional
        Cache loaded data in memory, by default True
        
    Examples
    --------
    >>> loader = DataLoader()
    >>> data = loader.load('hyperscanning')
    >>> print(data['subject1'].shape)
    (500, 20)
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "config",
        data_dir: Union[str, Path] = "data",
        cache_data: bool = True
    ):
        """Initialize data loader."""
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.cache_data = cache_data
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"DataLoader initialized: config_dir={self.config_dir}, "
                   f"data_dir={self.data_dir}")
    
    def load(
        self,
        scenario_name: str,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Load data for a scenario.
        
        Parameters
        ----------
        scenario_name : str
            Scenario name
        force_regenerate : bool, optional
            Force regeneration, by default False
            
        Returns
        -------
        Dict[str, Any]
            Loaded data
        """
        # Check cache
        if self.cache_data and scenario_name in self._cache and not force_regenerate:
            logger.info(f"Loading {scenario_name} from cache")
            return self._cache[scenario_name]
        
        # Load from config
        config_path = self.config_dir / f"scenario_{scenario_name}.yaml"
        data = load_data_from_config(
            config_path,
            force_regenerate=force_regenerate,
            data_dir=self.data_dir / scenario_name
        )
        
        # Cache if enabled
        if self.cache_data:
            self._cache[scenario_name] = data
        
        return data
    
    def clear_cache(self, scenario_name: Optional[str] = None):
        """
        Clear data cache.
        
        Parameters
        ----------
        scenario_name : Optional[str], optional
            Specific scenario to clear, by default None (clears all)
        """
        if scenario_name:
            if scenario_name in self._cache:
                del self._cache[scenario_name]
                logger.info(f"Cleared cache for {scenario_name}")
        else:
            self._cache.clear()
            logger.info("Cleared all cache")
