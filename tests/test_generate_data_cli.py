"""
Tests for the data generation CLI script.

This module tests the command-line interface for generating scenario data.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def cli_script(project_root):
    """Get the path to the CLI script."""
    return project_root / "scripts" / "generate_data.py"


def test_cli_help(cli_script):
    """Test that the CLI script shows help message."""
    result = subprocess.run(
        [sys.executable, str(cli_script), "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Generate simulated data" in result.stdout
    assert "--scenario" in result.stdout
    assert "--all" in result.stdout
    assert "--seed" in result.stdout


def test_cli_requires_scenario_or_all(cli_script):
    """Test that CLI requires either --scenario or --all."""
    result = subprocess.run(
        [sys.executable, str(cli_script)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode != 0
    assert "Must specify either --scenario or --all" in result.stderr


def test_cli_single_scenario(cli_script, tmp_path):
    """Test generating a single scenario."""
    output_dir = tmp_path / "test_output"
    
    result = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--scenario", "hyperscanning",
            "--seed", "42",
            "--output", str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # Check successful execution
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Check output messages
    assert "GENERATION COMPLETE" in result.stdout
    assert "✓ Successful: 1" in result.stdout
    
    # Check output file was created
    assert (output_dir / "hyperscanning_data.npz").exists()


def test_cli_all_scenarios(cli_script, tmp_path):
    """Test generating all scenarios."""
    output_dir = tmp_path / "test_output_all"
    
    result = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--all",
            "--seed", "123",
            "--output", str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    # Check successful execution
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Check output messages
    assert "GENERATION COMPLETE" in result.stdout
    assert "✓ Successful: 3" in result.stdout
    
    # Check all output files were created
    assert (output_dir / "hyperscanning_data.npz").exists()
    assert (output_dir / "llm_alignment_data.npz").exists()
    assert (output_dir / "genetics_data.npz").exists()


def test_cli_invalid_scenario(cli_script, tmp_path):
    """Test that invalid scenario name produces error."""
    result = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--scenario", "invalid_scenario",
            "--output", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    
    # Should fail with argument error
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()


def test_cli_verbose_mode(cli_script, tmp_path):
    """Test verbose mode provides additional output."""
    output_dir = tmp_path / "test_verbose"
    
    result = subprocess.run(
        [
            sys.executable,
            str(cli_script),
            "--scenario", "hyperscanning",
            "--seed", "42",
            "--output", str(output_dir),
            "--verbose",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    assert result.returncode == 0
    # Verbose mode should include INFO log messages
    assert "INFO" in result.stdout or "INFO" in result.stderr
