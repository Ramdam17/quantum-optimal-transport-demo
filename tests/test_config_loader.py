"""
Unit tests for configuration loader.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from src.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with test config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Create default config
            default_config = {
                'project': {'name': 'test_project', 'version': '1.0'},
                'classical_ot': {'method': 'sinkhorn', 'reg': 0.01},
                'logging': {'level': 'INFO'}
            }
            with open(config_dir / 'default.yaml', 'w') as f:
                yaml.dump(default_config, f)
            
            # Create scenario config
            scenario_config = {
                'scenario': {'name': 'test_scenario'},
                'data': {'n_samples': 100, 'n_features': 10},
                'classical_ot': {'reg': 0.05}  # Override default
            }
            with open(config_dir / 'scenario_test.yaml', 'w') as f:
                yaml.dump(scenario_config, f)
            
            yield config_dir
    
    def test_load_config_file(self, temp_config_dir):
        """Test loading a configuration file."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        assert config.get('scenario.name') == 'test_scenario'
        assert config.get('data.n_samples') == 100
    
    def test_merge_with_defaults(self, temp_config_dir):
        """Test merging scenario config with defaults."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=True
        )
        
        # Check scenario-specific values
        assert config.get('scenario.name') == 'test_scenario'
        assert config.get('data.n_samples') == 100
        
        # Check default values
        assert config.get('project.name') == 'test_project'
        assert config.get('classical_ot.method') == 'sinkhorn'
        assert config.get('logging.level') == 'INFO'
        
        # Check overridden value (scenario overrides default)
        assert config.get('classical_ot.reg') == 0.05
    
    def test_get_with_default(self, temp_config_dir):
        """Test getting value with default fallback."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        # Existing key
        assert config.get('data.n_samples', 999) == 100
        
        # Non-existing key with default
        assert config.get('nonexistent.key', 42) == 42
        
        # Non-existing key without default
        assert config.get('nonexistent.key') is None
    
    def test_get_section(self, temp_config_dir):
        """Test getting entire configuration section."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        data_section = config.get_section('data')
        assert isinstance(data_section, dict)
        assert data_section['n_samples'] == 100
        assert data_section['n_features'] == 10
    
    def test_get_section_missing(self, temp_config_dir):
        """Test error when getting non-existent section."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        with pytest.raises(KeyError, match="section_missing"):
            config.get_section('section_missing')
    
    def test_get_all(self, temp_config_dir):
        """Test getting entire configuration."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        all_config = config.get_all()
        assert isinstance(all_config, dict)
        assert 'scenario' in all_config
        assert 'data' in all_config
    
    def test_validate_required_success(self, temp_config_dir):
        """Test validation with all required keys present."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        # Should not raise
        config.validate_required(['scenario.name', 'data.n_samples'])
    
    def test_validate_required_missing(self, temp_config_dir):
        """Test validation with missing required keys."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        with pytest.raises(ValueError, match="Missing required"):
            config.validate_required(['scenario.name', 'missing.key'])
    
    def test_file_not_found(self, temp_config_dir):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader(temp_config_dir / 'nonexistent.yaml')
    
    def test_nested_dict_merge(self, temp_config_dir):
        """Test deep merging of nested dictionaries."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=True
        )
        
        # Default has project.name and project.version
        # Scenario doesn't override project, so both should be present
        assert config.get('project.name') == 'test_project'
        assert config.get('project.version') == '1.0'
        
        # classical_ot has method from default and reg from scenario
        assert config.get('classical_ot.method') == 'sinkhorn'
        assert config.get('classical_ot.reg') == 0.05
    
    def test_repr(self, temp_config_dir):
        """Test string representation."""
        config = ConfigLoader(
            temp_config_dir / 'scenario_test.yaml',
            load_defaults=False
        )
        
        repr_str = repr(config)
        assert 'test_scenario' in repr_str
        assert 'ConfigLoader' in repr_str
