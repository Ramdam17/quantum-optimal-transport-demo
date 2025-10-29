# Copilot Instructions for Quantum Optimal Transport Project

**Project Version:** v0.1.0 (Initial Development)  
**Last Updated:** October 28, 2025  
**Status:** Development Phase

---

## Project Overview

### Mission Statement
This is an **educational repository** demonstrating Quantum Optimal Transport (QOT) concepts with practical applications in social neuroscience, AI/LLM alignment, and genetics. The primary goal is pedagogical clarity and professional code quality suitable for public GitHub distribution.

### Target Audience
- Researchers in social neuroscience (hyperscanning, dyadic interaction)
- AI/ML researchers working with LLMs and embedding spaces
- Computational biologists studying gene expression
- Students learning optimal transport or quantum computing
- Workshop participants with basic Python knowledge

### Key Constraints
⚠️ **CRITICAL**: This is an educational project. The quantum components are **simulated** on classical hardware. We must be transparent about limitations and avoid "quantum hype" while maintaining scientific rigor.

---

## General Guidelines

### Language Requirements
🌍 **ALL project artifacts MUST be in ENGLISH**:
- ✅ Code (variable names, function names, class names)
- ✅ Documentation (docstrings, comments, markdown files)
- ✅ Commit messages
- ✅ Issue/PR descriptions
- ✅ Log messages
- ✅ Error messages
- ✅ Configuration files

**Rationale**: English ensures maximum accessibility for the scientific community and facilitates collaboration.

### File Operations - CRITICAL RULES
⚠️ **NEVER create, modify, or delete ANY file without explicit user approval**

**Mandatory workflow**:
1. **Propose** the change with full context
2. **Show** what will be created/modified (code preview)
3. **Wait** for explicit approval
4. **Execute** only after confirmation
5. **Report** what was done

**Example dialogue**:
```
Assistant: "I propose creating src/data/hyperscanning_simulator.py with the following structure:
[shows code preview]
This will implement the brain activity simulation for the hyperscanning scenario.
Shall I create this file?"

User: "Yes, go ahead" or "Wait, let me review first"
```

### Code Quality Standards

#### Maximum File Length
- **Hard limit: 300 lines** per file (excluding docstrings and blank lines)
- **Target: 200 lines** for most modules
- If approaching 300 lines, refactor into multiple modules
- Exception: Notebooks may exceed this limit for pedagogical flow

#### Complexity Limits
- **Cyclomatic complexity**: Max 10 per function
- **Function length**: Max 50 lines (excluding docstring)
- **Class length**: Max 250 lines (excluding docstrings)
- If exceeded, split into smaller components

#### Code Style
- **PEP 8 compliance**: Mandatory
- **Formatter**: Black (line length: 88)
- **Import sorter**: isort
- **Linter**: pylint + flake8
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all public functions/classes (NumPy style)

---

## Project Architecture

### Directory Structure

```
quantum-optimal-transport/
│
├── config/                             # YAML configuration files
│   ├── default.yaml                    # Default parameters
│   ├── scenario_hyperscanning.yaml     # Hyperscanning config
│   ├── scenario_llm_alignment.yaml     # LLM alignment config
│   └── scenario_genetics.yaml          # Genetics config
│
├── src/                                # Source code (all modules)
│   ├── __init__.py
│   ├── data/                           # Data generation and loading
│   │   ├── __init__.py
│   │   ├── base_simulator.py          # Abstract base for simulators
│   │   ├── hyperscanning_simulator.py # Brain activity simulation
│   │   ├── llm_simulator.py           # LLM embedding simulation
│   │   ├── genetics_simulator.py      # Gene expression simulation
│   │   └── loaders.py                 # Data loading utilities
│   │
│   ├── optimal_transport/              # Classical OT implementation
│   │   ├── __init__.py
│   │   ├── classical.py               # Sinkhorn, exact OT
│   │   └── metrics.py                 # Wasserstein distances
│   │
│   ├── quantum/                        # Quantum computing modules
│   │   ├── __init__.py
│   │   ├── circuits.py                # Quantum circuit construction
│   │   ├── state_preparation.py       # Encode distributions
│   │   ├── simulators.py              # Qiskit/PennyLane wrappers
│   │   ├── qot_algorithms.py          # QAOA/VQE for OT
│   │   └── qot_metrics.py             # Quantum distance metrics
│   │
│   ├── visualization/                  # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── ot_plots.py                # OT visualizations
│   │   ├── quantum_plots.py           # Quantum circuit viz
│   │   └── comparisons.py             # OT vs QOT comparisons
│   │
│   └── utils/                          # Utilities and helpers
│       ├── __init__.py
│       ├── config_loader.py           # YAML config parsing
│       ├── logger.py                  # Logging setup
│       └── helpers.py                 # Common utilities
│
├── notebooks/                          # Jupyter notebooks
│   ├── main_quantum_ot.ipynb          # Main config-driven notebook
│   └── examples/                       # Pre-configured demos
│       ├── 01_hyperscanning_demo.ipynb
│       ├── 02_llm_alignment_demo.ipynb
│       └── 03_genetics_demo.ipynb
│
├── tests/                              # Unit and integration tests
│   ├── __init__.py
│   ├── test_config_loader.py
│   ├── test_simulators.py
│   ├── test_classical_ot.py
│   ├── test_quantum.py
│   └── test_integration.py
│
├── data/                               # Simulated data (gitignored)
│   ├── .gitkeep
│   ├── README.md                      # Data generation instructions
│   └── generate_data.py               # Data generation script
│
├── outputs/                            # Generated outputs (gitignored)
│   ├── figures/                        # Plots and visualizations
│   └── results/                        # Numerical results (JSON/CSV)
│
├── logs/                               # Log files (gitignored)
│   └── .gitkeep
│
├── docs/                               # Documentation
│   ├── README.md                      # Documentation index
│   ├── THEORY.md                      # Theoretical foundations
│   ├── RESOURCES.md                   # Bibliography & links
│   ├── SCENARIOS.md                   # Scenario descriptions
│   ├── SETUP.md                       # Installation guide
│   └── API.md                         # API reference
│
├── .github/                            # GitHub configuration
│   └── workflows/
│       └── tests.yml                  # CI/CD pipeline
│
├── pyproject.toml                      # Poetry dependencies
├── requirements.txt                    # Pip fallback
├── README.md                           # Main documentation
├── LICENSE                             # License file
├── .gitignore                          # Git ignore patterns
├── CHANGELOG.md                        # Version history
└── TODO.md                             # Project TODO list
```

### Module Organization Principles

#### 1. Single Responsibility
Each module should have **one clear purpose**:
- ✅ `hyperscanning_simulator.py`: Only simulate brain activity
- ❌ `hyperscanning_simulator.py`: Simulate + visualize + analyze

#### 2. Dependency Hierarchy
```
utils (lowest level, no internal dependencies)
  ↓
data (depends on utils)
  ↓
optimal_transport, quantum (depend on data, utils)
  ↓
visualization (depends on all above)
```

**Rules**:
- Lower-level modules **never** import from higher-level modules
- Avoid circular dependencies at all costs
- Use dependency injection for flexibility

#### 3. Configuration-Driven Design
All modules must accept configuration via:
1. **YAML files** (primary method)
2. **Direct parameters** (for programmatic use)
3. **Default values** (sensible fallbacks)

**Pattern**:
```python
class MyModule:
    def __init__(self, config_path: Optional[Union[str, Path]] = None, **kwargs):
        """Initialize with optional config path or direct parameters."""
        if config_path:
            self.config = ConfigLoader(config_path).get_section('my_section')
        else:
            self.config = kwargs or {}
```

---

## Python Style Guide

### Naming Conventions

```python
# Modules and packages: lowercase with underscores
src/data/hyperscanning_simulator.py

# Classes: PascalCase
class HyperscanningSimulator:
    pass

# Functions and methods: lowercase with underscores
def generate_brain_activity():
    pass

# Constants: UPPERCASE with underscores
MAX_REGIONS = 100
DEFAULT_SAMPLING_RATE = 1.0

# Private methods/attributes: single leading underscore
def _internal_helper():
    pass

# Protected in inheritance: single underscore
class Base:
    def _protected_method(self):
        pass
```

### Type Hints

**Always use type hints** for function signatures:

```python
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import numpy as np

def process_data(
    data: np.ndarray,
    config_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process input data with specified threshold.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None
    threshold : float, optional
        Processing threshold, by default 0.5
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        Processed data and metrics dictionary
    """
    pass
```

### Docstring Format

**Use NumPy-style docstrings** for all public functions and classes:

```python
def optimal_transport(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Compute optimal transport between source and target distributions.
    
    This function implements the Sinkhorn algorithm for entropic
    regularized optimal transport.
    
    Parameters
    ----------
    source : np.ndarray
        Source distribution of shape (n_samples_source, n_features)
    target : np.ndarray
        Target distribution of shape (n_samples_target, n_features)
    reg : float, optional
        Entropic regularization parameter, by default 0.01
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'transport_plan': Transport matrix (n_source, n_target)
        - 'cost': Optimal transport cost (scalar)
        - 'iterations': Number of Sinkhorn iterations
    
    Raises
    ------
    ValueError
        If source or target have invalid shapes or negative values
    
    Examples
    --------
    >>> source = np.random.randn(100, 2)
    >>> target = np.random.randn(150, 2)
    >>> result = optimal_transport(source, target, reg=0.05)
    >>> print(result['cost'])
    0.234
    
    Notes
    -----
    The algorithm converges when the marginal constraints are satisfied
    within a tolerance of 1e-6.
    
    References
    ----------
    .. [1] Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation
       of Optimal Transport. NIPS.
    """
    pass
```

### Import Organization

**Use isort** with the following order:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import sys
from pathlib import Path
from typing import Optional, Dict

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# Local
from src.utils.config_loader import ConfigLoader
from src.data.base_simulator import BaseSimulator
```

### Error Handling

**Always use specific exception types**:

```python
# ❌ Bad
try:
    result = compute_something()
except:
    print("Error occurred")

# ✅ Good
try:
    result = compute_something()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}")
    raise ConfigurationError(f"Missing config: {e}")
```

**Custom exceptions** for domain-specific errors:

```python
# src/utils/exceptions.py
class QuantumOTError(Exception):
    """Base exception for quantum OT operations."""
    pass

class SimulationError(QuantumOTError):
    """Raised when data simulation fails."""
    pass

class QuantumCircuitError(QuantumOTError):
    """Raised when quantum circuit construction fails."""
    pass
```

---

## Configuration System

### YAML Structure

**Standard configuration template**:

```yaml
# config/scenario_name.yaml

# Scenario metadata
scenario:
  name: "scenario_name"
  description: "Brief description"
  domain: "neuroscience|ai|genetics"

# Data generation parameters
data:
  type: "data_type"
  n_samples: 1000
  n_features: 50
  seed: 42
  # ... scenario-specific params

# Classical optimal transport
classical_ot:
  method: "sinkhorn"  # or "exact"
  reg: 0.01           # entropic regularization
  num_iter: 1000
  tolerance: 1e-6

# Quantum optimal transport
quantum_ot:
  backend: "qiskit"   # or "pennylane"
  n_qubits: 8
  optimizer: "COBYLA"
  max_iter: 100
  shots: 1024
  use_simulator: true

# Visualization settings
visualization:
  save_figures: true
  output_dir: "outputs/figures"
  format: "png"
  dpi: 300
  style: "seaborn"
  show_interactive: false

# Logging and outputs
logging:
  level: "INFO"       # DEBUG, INFO, WARNING, ERROR
  file: "logs/{scenario}_{{date}}.log"
  
outputs:
  save_results: true
  results_dir: "outputs/results"
  format: "json"      # or "csv"
```

### Config Loader Pattern

**All modules must use this initialization pattern**:

```python
from pathlib import Path
from typing import Optional, Union
from src.utils.config_loader import ConfigLoader

class MyModule:
    """Module description."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize module.
        
        Parameters
        ----------
        config_path : Optional[Union[str, Path]], optional
            Path to YAML configuration file. If None, uses kwargs.
        **kwargs
            Direct parameter overrides
        """
        # Load configuration
        if config_path:
            self.config = ConfigLoader(config_path)
            module_config = self.config.get_section('my_section')
        else:
            module_config = {}
        
        # Merge with kwargs (kwargs take precedence)
        self.params = {**module_config, **kwargs}
        
        # Extract and validate parameters
        self.param1 = self.params.get('param1', default_value)
        self._validate_params()
    
    def _validate_params(self):
        """Validate configuration parameters."""
        if self.param1 < 0:
            raise ValueError("param1 must be non-negative")
```

---

## Testing Requirements

### Testing Philosophy
- **Test-Driven Development (TDD)**: Write tests before or alongside implementation
- **Coverage target**: >80% code coverage
- **Test isolation**: Each test should be independent
- **Fast execution**: Unit tests should complete in <1 second

### Test Structure

```python
# tests/test_module_name.py
import pytest
import numpy as np
from src.module import MyClass

class TestMyClass:
    """Test suite for MyClass."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for test data."""
        return np.random.randn(100, 10)
    
    @pytest.fixture
    def config_path(self, tmp_path):
        """Fixture for temporary config file."""
        config = tmp_path / "test_config.yaml"
        config.write_text("param1: 42\n")
        return config
    
    def test_initialization_with_config(self, config_path):
        """Test class initialization with config file."""
        obj = MyClass(config_path=config_path)
        assert obj.param1 == 42
    
    def test_initialization_with_kwargs(self):
        """Test class initialization with direct parameters."""
        obj = MyClass(param1=100)
        assert obj.param1 == 100
    
    def test_process_valid_data(self, sample_data):
        """Test processing with valid input."""
        obj = MyClass()
        result = obj.process(sample_data)
        assert result.shape == sample_data.shape
    
    def test_process_invalid_shape(self):
        """Test error handling for invalid input shape."""
        obj = MyClass()
        with pytest.raises(ValueError, match="Invalid shape"):
            obj.process(np.array([1, 2, 3]))
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same seed."""
        obj1 = MyClass(seed=42)
        obj2 = MyClass(seed=42)
        result1 = obj1.process(sample_data)
        result2 = obj2.process(sample_data)
        np.testing.assert_array_equal(result1, result2)
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_simulators.py

# Run with verbose output
poetry run pytest -v

# Run and stop on first failure
poetry run pytest -x
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test interactions between modules
3. **End-to-End Tests**: Test complete pipelines for each scenario

**Example integration test**:
```python
def test_full_hyperscanning_pipeline():
    """Test complete hyperscanning workflow."""
    # Generate data
    simulator = HyperscanningSimulator(seed=42)
    data = simulator.generate(n_subjects=2, n_regions=10)
    
    # Classical OT
    ot = OptimalTransport()
    ot_result = ot.compute(data['subject1'], data['subject2'])
    
    # Quantum OT
    qot = QuantumOT(n_qubits=8)
    qot_result = qot.compute(data['subject1'], data['subject2'])
    
    # Verify results
    assert 'cost' in ot_result
    assert 'cost' in qot_result
    assert ot_result['cost'] > 0
```

---

## Logging Standards

### Logger Setup

**Use Python's `logging` module** with the following configuration:

```python
# src/utils/logger.py
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: Path = Path("logs"),
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    log_dir : Path, optional
        Directory for log files, by default Path("logs")
    level : str, optional
        Logging level, by default "INFO"
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # File handler
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### Logging Levels

**Use appropriate log levels**:

```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# DEBUG: Detailed diagnostic information
logger.debug(f"Processing {n_samples} samples with {n_features} features")

# INFO: General information about execution
logger.info("Starting optimal transport computation")

# WARNING: Potential issues that don't prevent execution
logger.warning(f"High regularization parameter: {reg}. Results may be inaccurate")

# ERROR: Errors that prevent specific operations
logger.error(f"Failed to load configuration from {config_path}")

# CRITICAL: Critical failures requiring immediate attention
logger.critical("Quantum backend initialization failed. Aborting")
```

### Timing Context Manager

```python
import time
from contextlib import contextmanager

@contextmanager
def log_timing(logger, operation: str):
    """Context manager for timing operations."""
    start = time.time()
    logger.info(f"Starting {operation}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed {operation} in {elapsed:.2f}s")

# Usage
with log_timing(logger, "optimal transport computation"):
    result = compute_ot(data)
```

---

## Notebook Guidelines

### Structure

Every notebook should follow this structure:

```markdown
# Title: Clear and Descriptive

## 0. Setup and Configuration
- Import libraries
- Set random seeds
- Load configuration
- Setup logging

## 1. Introduction
- Context and motivation
- Learning objectives
- Overview of what will be covered

## 2. Theory Section
- Mathematical foundations
- Intuitive explanations
- Visual illustrations

## 3. Implementation
- Step-by-step code
- Intermediate outputs
- Sanity checks

## 4. Visualization
- Results visualization
- Interpretation
- Discussion

## 5. Conclusion
- Summary of findings
- Limitations
- Next steps
```

### Cell Organization

```python
# Import cell (always first)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path.cwd().parent))

from src.utils.config_loader import ConfigLoader
from src.data.hyperscanning_simulator import HyperscanningSimulator

# Configuration cell
SCENARIO = "hyperscanning"  # Change for different scenarios
config = ConfigLoader(f"../config/scenario_{SCENARIO}.yaml")
print(f"Loaded configuration for: {config.get('scenario.name')}")

# Execution cells with clear outputs
simulator = HyperscanningSimulator(seed=42)
data = simulator.generate(**config.get('data'))
print(f"Generated data: {data['subject1'].shape}")

# Visualization cells with titles
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(data['subject1'].T)
axes[0].set_title("Subject 1 Activity")
axes[1].plot(data['subject2'].T)
axes[1].set_title("Subject 2 Activity")
plt.tight_layout()
plt.show()
```

### Best Practices

1. **Clear markdown explanations** before each code cell
2. **Print intermediate results** for debugging
3. **Use descriptive variable names**
4. **Add comments for complex code**
5. **Visualize results** whenever possible
6. **Include error handling** in long computations
7. **Save important outputs** to disk
8. **Clear outputs before committing** (use `jupyter nbconvert --clear-output`)

---

## Git Workflow

### Branching Strategy

```
main (protected)
  ├── develop (primary development)
  │   ├── feature/data-simulators
  │   ├── feature/classical-ot
  │   ├── feature/quantum-circuits
  │   └── feature/notebooks
  ├── fix/bug-description
  └── docs/documentation-update
```

### Commit Message Format

**Use conventional commits**:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples**:
```bash
feat(data): add hyperscanning simulator with correlation control

fix(quantum): correct qubit indexing in state preparation

docs(theory): add mathematical derivation of Wasserstein distance

test(ot): add unit tests for Sinkhorn algorithm convergence

refactor(utils): extract config validation to separate module
```

### Before Committing

```bash
# Format code
poetry run black src tests
poetry run isort src tests

# Run linters
poetry run pylint src
poetry run flake8 src

# Run tests
poetry run pytest

# Check coverage
poetry run pytest --cov=src --cov-report=term
```

---

## Documentation Standards

### README.md Structure

```markdown
# Project Title

[![Tests](badge)](link) [![Coverage](badge)](link) [![License](badge)](link)

Brief one-paragraph description.

## 🚀 Quick Start

Three commands to get running.

## 📖 Documentation

Links to detailed docs.

## 🎯 Scenarios

Brief description of three scenarios.

## 🛠️ Installation

Detailed setup instructions.

## 📊 Usage

Examples and tutorials.

## 🧪 Testing

How to run tests.

## 📚 References

Key papers and resources.

## 📝 License

License information.

## 🙏 Acknowledgments

Credits and thanks.
```

### API Documentation

**Use docstrings that can be auto-generated** with Sphinx or mkdocs:

```python
"""
Module description.

This module provides functionality for...

Examples
--------
>>> from src.module import MyClass
>>> obj = MyClass(param=42)
>>> result = obj.process(data)
"""
```

---

## Performance Guidelines

### Memory Efficiency

```python
# ❌ Bad: Load all data into memory
data = [load_sample(i) for i in range(1000000)]
process(data)

# ✅ Good: Use generators
def data_generator():
    for i in range(1000000):
        yield load_sample(i)

for sample in data_generator():
    process(sample)
```

### Parallelization

```python
from concurrent.futures import ProcessPoolExecutor

def process_scenario(scenario_name):
    """Process a single scenario."""
    # ... processing logic
    return results

# Parallel processing
scenarios = ["hyperscanning", "llm_alignment", "genetics"]
with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_scenario, scenarios))
```

---

## Security & Best Practices

### Never Commit Sensitive Data

**Add to `.gitignore`**:
```
# Data files
data/*.npz
data/*.csv
data/*.json

# Outputs
outputs/
logs/

# Credentials
.env
*.key
credentials.json

# System files
.DS_Store
Thumbs.db
```

### Input Validation

```python
def validate_input(data: np.ndarray, expected_shape: Optional[Tuple] = None):
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(data)}")
    
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    
    if expected_shape and data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
```

---

## Quantum Computing Specifics

### Backend Selection

```python
from qiskit import Aer

# Available backends
BACKENDS = {
    'statevector': Aer.get_backend('statevector_simulator'),
    'qasm': Aer.get_backend('qasm_simulator'),
    'unitary': Aer.get_backend('unitary_simulator')
}

# Use appropriate backend for task
def run_circuit(circuit, backend='statevector', shots=1024):
    """Execute quantum circuit."""
    backend_instance = BACKENDS.get(backend)
    if backend_instance is None:
        raise ValueError(f"Unknown backend: {backend}")
    
    job = backend_instance.run(circuit, shots=shots)
    return job.result()
```

### Circuit Optimization

```python
from qiskit import transpile

# Optimize circuit before execution
def optimize_circuit(circuit, optimization_level=2):
    """Transpile and optimize quantum circuit."""
    optimized = transpile(
        circuit,
        optimization_level=optimization_level,
        basis_gates=['u1', 'u2', 'u3', 'cx']
    )
    return optimized
```

### Limitations to Document

⚠️ **Always be transparent about**:
- Simulations run on classical hardware (no quantum advantage)
- Number of qubits limited by simulator memory
- Shot noise in measurements
- Approximations in algorithms
- Current lack of error correction

---

## Educational Principles

### Pedagogical Clarity

1. **Explain before implementing**: Theory comes first
2. **Simple examples first**: Start with 1D, then generalize
3. **Visualize everything**: Plots aid understanding
4. **Compare and contrast**: Show classical vs quantum side-by-side
5. **Acknowledge limitations**: Be honest about what works and what doesn't

### Notebook Writing Style

```markdown
## 3.2 Optimal Transport: Intuition

Imagine you have a pile of sand (source distribution) and you want to
move it to fill a hole (target distribution). Optimal transport finds
the most efficient way to move the sand, minimizing the total "work"
(mass × distance).

### Mathematical Formulation

The optimal transport problem seeks a transport plan $T: X \to Y$ that
minimizes:

$$W(P, Q) = \inf_{\pi \in \Pi(P,Q)} \int_{X \times Y} c(x, y) d\pi(x, y)$$

Let's see this in action with a simple example:
```

---

## Troubleshooting Common Issues

### Import Errors

```python
# If you get "ModuleNotFoundError: No module named 'src'"
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
```

### Qiskit Installation Issues

```bash
# If qiskit-aer fails to install
poetry add qiskit --python "^3.9,<3.13"
poetry add qiskit-aer --python "^3.9,<3.13"
```

### Jupyter Kernel Issues

```bash
# Register kernel with Poetry environment
poetry run python -m ipykernel install --user --name=quantum-ot
```

---

## Release Checklist

Before releasing v1.0.0:

- [ ] All tests passing (>80% coverage)
- [ ] All notebooks execute without errors
- [ ] Documentation complete and proofread
- [ ] Code formatted (black, isort)
- [ ] No pylint/flake8 warnings
- [ ] README.md up to date
- [ ] CHANGELOG.md populated
- [ ] LICENSE file present
- [ ] .gitignore comprehensive
- [ ] No sensitive data in repository
- [ ] Version tagged in git
- [ ] GitHub release created

---

## Success Metrics

### Code Quality
- ✅ >80% test coverage
- ✅ Pylint score >8.0/10
- ✅ All type hints present
- ✅ All docstrings complete

### Functionality
- ✅ All three scenarios work end-to-end
- ✅ Notebooks execute in <10 minutes
- ✅ Reproducible results (seeding)
- ✅ YAML configs control all behavior

### Documentation
- ✅ README clear and comprehensive
- ✅ Theory docs accurate
- ✅ API reference auto-generated
- ✅ Examples and tutorials complete

### Professional Appearance
- ✅ Clean repository structure
- ✅ Badges on README
- ✅ No dead links
- ✅ Consistent formatting

---

## Key Reminders 🎯

1. **ASK before changing files** - Always get approval
2. **English only** - All documentation and code
3. **Test everything** - >80% coverage target
4. **Document as you go** - Don't leave it for the end
5. **Be honest about limitations** - This is educational
6. **Keep it simple** - Pedagogical clarity over complexity
7. **Think modular** - Small, focused modules
8. **Use type hints** - Help users and future you
9. **Log appropriately** - DEBUG for details, INFO for progress
10. **Have fun** - This is a cool project! 🚀

---

**Version History:**
- v0.1.0 (October 28, 2025): Initial instructions created