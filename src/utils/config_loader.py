"""
Configuration loader for YAML files.

This module provides utilities for loading and managing YAML configuration files
with support for merging default and scenario-specific configurations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigLoader:
    """
    Load and manage YAML configuration files.
    
    This class handles loading YAML configs with support for:
    - Default configuration merging
    - Nested key access with dot notation
    - Configuration validation
    - Multiple config file merging
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the main configuration file
    load_defaults : bool, optional
        Whether to load and merge default.yaml, by default True
    
    Examples
    --------
    >>> config = ConfigLoader("config/scenario_hyperscanning.yaml")
    >>> n_regions = config.get("data.n_regions")
    >>> ot_params = config.get_section("classical_ot")
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        load_defaults: bool = True
    ):
        """Initialize configuration loader."""
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the main configuration
        self.config = self._load_yaml(self.config_path)
        
        # Merge with defaults if requested
        if load_defaults:
            default_path = self.config_path.parent / "default.yaml"
            if default_path.exists():
                default_config = self._load_yaml(default_path)
                self.config = self._merge_configs(default_config, self.config)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Parameters
        ----------
        path : Path
            Path to YAML file
            
        Returns
        -------
        Dict[str, Any]
            Parsed configuration dictionary
        """
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_configs(
        self,
        default: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        The override config takes precedence over default config.
        
        Parameters
        ----------
        default : Dict[str, Any]
            Default configuration
        override : Dict[str, Any]
            Override configuration
            
        Returns
        -------
        Dict[str, Any]
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key with dot notation (e.g., "data.n_regions")
        default : Any, optional
            Default value if key not found, by default None
            
        Returns
        -------
        Any
            Configuration value
            
        Examples
        --------
        >>> config = ConfigLoader("config/scenario_hyperscanning.yaml")
        >>> n_regions = config.get("data.n_regions", 10)
        >>> ot_method = config.get("classical_ot.method", "sinkhorn")
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Parameters
        ----------
        section : str
            Section name (top-level key)
            
        Returns
        -------
        Dict[str, Any]
            Configuration section as dictionary
            
        Raises
        ------
        KeyError
            If section does not exist
            
        Examples
        --------
        >>> config = ConfigLoader("config/scenario_hyperscanning.yaml")
        >>> data_config = config.get_section("data")
        >>> print(data_config["n_regions"])
        """
        if section not in self.config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        return self.config[section]
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration.
        
        Returns
        -------
        Dict[str, Any]
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def validate_required(self, required_keys: list) -> None:
        """
        Validate that required keys exist in configuration.
        
        Parameters
        ----------
        required_keys : list
            List of required keys (dot notation supported)
            
        Raises
        ------
        ValueError
            If any required key is missing
            
        Examples
        --------
        >>> config = ConfigLoader("config/scenario_hyperscanning.yaml")
        >>> config.validate_required(["data.n_regions", "classical_ot.method"])
        """
        missing = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing.append(key)
        
        if missing:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing)}"
            )
    
    def __repr__(self) -> str:
        """String representation."""
        scenario_name = self.get("scenario.name", "unknown")
        return f"ConfigLoader(scenario='{scenario_name}')"
