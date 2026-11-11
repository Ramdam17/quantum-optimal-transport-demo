"""Run the free-second-harmonic bench from the configuration and write its results.

The bench couples two noisy phase oscillators through a first and a second
harmonic of their phase difference (:mod:`qot_course.quantum_ot.biharmonic`) and
compares coupling settings with the protocol of
:mod:`qot_course.quantum_ot.bench_protocol`. This script reads every tunable
value from the section ``second_harmonic_bench`` of the configuration, runs the
whole protocol and writes eight files to the output directory, in this order:

``map.csv``
    The exact long-run moments on the grid of couplings ``(k1, k2)``.
``freedom.csv``
    The range of the exact ``abs_m2`` reachable in each bin of ``abs_m1``, next
    to the single value reached with ``k2 = 0``.
``continuity.json``
    The Spearman rank correlations, along the ``k1`` grid at ``k2 = 0``, of the
    exact ``abs_m1`` with the exact ``abs_m2`` and with the exact ``qmi_rich``.
``timestep_check.csv``
    Simulated against exact moduli at the configured time steps, the exact
    values being those of the continuous-time model (``abs_m*_exact``) and those
    of the Euler--Maruyama scheme at the row's step (``abs_m*_exact_discrete``),
    each with the gap of the simulated modulus in standard errors.
``comparisons.csv``
    One row per comparison, record duration, master seed and statistic, with
    both exact references: ``exact_a`` and ``exact_b`` for the continuous-time
    model, then ``exact_discrete_a`` and ``exact_discrete_b`` for the scheme
    simulated at the model's ``dt``, which is what the simulated means converge
    to.
``summary.csv``
    The comparisons summarised over master seeds, with the same two references,
    including the mean numbers of valid records and of null differences and the
    mean null standard deviations.
``reliability.csv``
    The shortest duration whose mean detection rate reaches the threshold, with
    the mean numbers of valid records and of null differences at that duration.
    The experiment ``trap`` has no row.
``run_metadata.json``
    The resolved configuration and the path of its file relative to the
    repository, the software versions, the solved settings and
    the comparisons that were not made, with the reason for each. Each solved
    setting that comes from a solver records the law it was matched on
    (``matched_on``, ``"discretised scheme"``) and the step (``matched_dt``).

The first three files describe the continuous-time model. The matched pairs of
the experiments ``equal_plv`` and ``equal_moduli`` are solved on the stationary
law of the scheme simulated at the model's ``dt``
(:func:`qot_course.quantum_ot.biharmonic.discretised_stationary_moments`),
because the step shifts the moments by a different amount in the two settings
of a pair: a pair matched in continuous time would, as simulated, no longer be
matched.

The sign convention is that of the library: :math:`\\Delta = \\theta_1 - \\theta_2`
and :math:`m_k = E[e^{ik\\Delta}]`. The script is pure NumPy and SciPy: it never
touches quantum hardware or the network.

Undefined numbers. A number that is undefined is written as an empty cell in
every CSV file and as ``null`` in every JSON file, never as ``nan``. This covers
the exact relative phase at ``k1 = 0``, where the exact first moment is zero,
the relative-phase statistics on records whose first moment is below the
threshold, a bin centre that no ``k1`` reaches, and a duration that never
reaches the reliability threshold.

Seeds. Every comparison draws its records from a fixed tree of
:class:`numpy.random.SeedSequence` children, three levels deep. For each master
seed ``s`` of ``protocol.master_seeds``, child ``e`` of ``SeedSequence(s)`` goes
to experiment ``e`` in the fixed order detection (0), equal phase-locking value
(1), equal moduli (2) and trap (3); child ``i`` of that child goes to entry
``i`` of the experiment (entry ``i`` of ``detection.k2_values`` or of
``equal_plv``, entry 0 for the other two); and child ``d`` of that one goes to
duration ``d`` of ``protocol.durations_s``. The script builds that child
directly as ``SeedSequence(s, spawn_key=(e, i, d))``, which has the same state
as the spawned one. Adding an entry at the end of any experiment's list, or a
duration at the end of ``durations_s``, therefore leaves every earlier number
unchanged, and a comparison that is not made frees its slot without moving any
other. :func:`compare_conditions` then spawns its six record sets from the
child it receives. The time-step check uses its own tree, rooted at the
top-level ``seed`` of the configuration file rather than at a key of the
section (the section has none for it), and independent of the master seeds:
``SeedSequence(seed, spawn_key=(i, j, c))``, that is child ``c`` of child ``j``
of child ``i``, goes to setting ``i`` of ``timestep_check.settings``, time step
``j`` of ``timestep_check.dt_values`` and chunk ``c`` of records, the chunks
holding ``timestep_check.records_per_chunk`` records each, in record order,
with a smaller last chunk when the count does not divide evenly. The chunk
seeds therefore depend on ``records_per_chunk``: changing that memory setting
changes every number of the time-step check.

Rank identity. ``continuity.json`` writes, beside each Spearman coefficient as
:func:`scipy.stats.spearmanr` returns it, a boolean that says whether the two
rank vectors of :func:`scipy.stats.rankdata` are equal element by element. Both
are written because the coefficient is a floating-point Pearson correlation of
the ranks: on the full 26-point ``k1`` grid of the default configuration,
``spearmanr`` returned 0.9999999999999999 for rank vectors that are identical
(measured on 2026-11-11), so the coefficient alone cannot say whether the
order is exactly preserved.

Run::

    uv run python scripts/run_second_harmonic_bench.py            # full run
    uv run python scripts/run_second_harmonic_bench.py --quick    # smoke-sized run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

from qot_course.quantum_ot.bench_protocol import compare_conditions, long_run_statistics
from qot_course.quantum_ot.biharmonic import (
    discretised_stationary_moments,
    relative_phase,
    simulate_biharmonic_dyad,
    solve_equal_moduli_pair,
    solve_k1_for_plv,
    stationary_moments,
)
from qot_course.utils.config import deep_merge, load_config
from qot_course.utils.logging_config import get_logger

logger = get_logger("run_second_harmonic_bench")

# Repository root, from which the default configuration and output paths are
# resolved (structural: this file lives in <repo>/scripts).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "base.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "second_harmonic_bench"
# Name of the configuration section read by this script (structural).
CONFIG_SECTION = "second_harmonic_bench"
# Width of the bins of abs_m1 in freedom.csv. Layout constant fixed by the plan
# of the bench (Task 7 of docs/superpowers/plans/2026-11-11-free-second-harmonic-
# bench.md), not a tuning choice.
FREEDOM_BIN_WIDTH = 0.05
# Decimals kept on the bin edges written to freedom.csv, so that an edge such as
# 14 * 0.05 is written 0.7 rather than 0.7000000000000001 (layout, far finer
# than the bin width).
BIN_EDGE_DECIMALS = 12
# Highest circular moment read from stationary_moments: the bench uses m1 and m2
# (structural).
MAX_HARMONIC = 2
# Keys of the model mapping forwarded to the solvers and to stationary_moments,
# and to the simulator (structural, fixed by the library signatures).
SOLVER_KEYS = ("omega_1", "omega_2", "noise_std", "n_modes")
SIMULATOR_KEYS = ("omega_1", "omega_2", "noise_std", "dt", "burn_in")
# Value of the key matched_on in run_metadata.json for a setting solved on the
# stationary law of the simulated Euler-Maruyama scheme (structural).
MATCHED_ON_SCHEME = "discretised scheme"
# Experiments in their fixed numbering order (see "Seeds" in the module docstring).
EXPERIMENTS = ("detection", "equal_plv", "equal_moduli", "trap")
# Columns that prefix every row of compare_conditions in comparisons.csv.
COMPARISON_PREFIX = (
    "experiment",
    "label",
    "k1_a",
    "k2_a",
    "k1_b",
    "k2_b",
    "duration_s",
    "master_seed",
)
# Keys of the rows returned by compare_conditions, in their order.
COMPARISON_KEYS = (
    "statistic",
    "mean_a",
    "mean_b",
    "sd_a",
    "sd_b",
    "exact_a",
    "exact_b",
    "exact_discrete_a",
    "exact_discrete_b",
    "score",
    "auc",
    "detection_rate",
    "null_sd_a",
    "null_sd_b",
    "n_valid",
    "n_null",
)


@dataclass
class PlannedComparison:
    """One comparison of the protocol, solved before any record is simulated.

    Attributes
    ----------
    experiment : str
        One of :data:`EXPERIMENTS`.
    entry : int
        Index of the comparison within its experiment's list, the second level
        of the seed tree.
    label : str
        Short description of the comparison within its experiment.
    condition_a, condition_b : tuple of float or None
        Couplings ``(K1, K2)`` in rad/s of the reference and compared settings;
        ``condition_b`` is None when the comparison is not made.
    solved : dict
        The settings as solved, recorded in ``run_metadata.json``.
    not_made_reason : str or None
        Why the comparison is not made, or None when it is made.
    """

    experiment: str
    entry: int
    label: str
    condition_a: tuple[float, float]
    condition_b: tuple[float, float] | None
    solved: dict[str, Any] = field(default_factory=dict)
    not_made_reason: str | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_undefined(value: Any) -> bool:
    """True for None and for a float NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _cell(value: Any) -> str:
    """Text of one CSV cell: empty when undefined, full precision for floats."""
    if _is_undefined(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _json_safe(obj: Any) -> Any:
    """Copy of ``obj`` with NaN replaced by None and NumPy scalars made native."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write ``rows`` to ``path`` with a header, undefined numbers as empty cells."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([_cell(row[name]) for name in fieldnames])


def _write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as strict JSON (no NaN), undefined numbers as null."""
    path.write_text(
        json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _finite_summary(values: Sequence[float]) -> tuple[float, float, float, int]:
    """Mean, minimum, maximum and count of the finite values; NaN when there is none."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(x)), float(np.min(x)), float(np.max(x)), int(x.size)


def _select(mapping: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """Sub-mapping of ``mapping`` restricted to ``keys``."""
    return {key: mapping[key] for key in keys}


def _model_mapping(section: Mapping[str, Any]) -> dict[str, float]:
    """The six-key model mapping of compare_conditions, built once from the section."""
    model = section["model"]
    return {
        "omega_1": float(model["omega_1"]),
        "omega_2": float(model["omega_2"]),
        "noise_std": float(model["noise_std"]),
        "dt": float(model["dt"]),
        "burn_in": float(model["burn_in_s"]),
        "n_modes": int(section["solver"]["n_modes"]),
    }


def _grid(spec: Mapping[str, Any]) -> np.ndarray:
    """Evenly spaced grid ``linspace(min, max, points)`` from a map axis entry."""
    return np.linspace(float(spec["min"]), float(spec["max"]), int(spec["points"]))


class _Stage:
    """Context manager that logs the start, the duration and the output of a stage."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path

    def __enter__(self) -> _Stage:
        logger.info("Stage %s: started", self.name)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            logger.info(
                "Stage %s: done in %.2f s, wrote %s",
                self.name,
                time.perf_counter() - self.start,
                self.path,
            )


# ---------------------------------------------------------------------------
# One function per output file
# ---------------------------------------------------------------------------


def write_map(
    section: Mapping[str, Any], model: Mapping[str, float], out: Path
) -> list[dict[str, float]]:
    """Exact moments on the coupling grid, written to ``map.csv``.

    Parameters
    ----------
    section : mapping
        Resolved configuration section; reads ``map``, ``protocol.min_first_moment``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`.
    out : Path
        Output directory.

    Returns
    -------
    rows : list of dict
        One row per grid point, ``k1`` major and ``k2`` minor, with the columns
        ``k1, k2`` (rad/s), the real and imaginary parts of ``m1`` and ``m2``,
        ``abs_m1``, ``abs_m2``, ``relative_phase`` (rad, NaN at ``k1 = 0``) and
        ``qmi_rich_exact`` (nats).
    """
    path = out / "map.csv"
    with _Stage("map", path):
        solver_kwargs = _select(model, SOLVER_KEYS)
        min_first_moment = float(section["protocol"]["min_first_moment"])
        rows = []
        for k1 in _grid(section["map"]["k1"]):
            for k2 in _grid(section["map"]["k2"]):
                m = stationary_moments(
                    float(k1), float(k2), max_harmonic=MAX_HARMONIC, **solver_kwargs
                )
                stats = long_run_statistics(
                    m, weight_sets={}, min_first_moment=min_first_moment
                )
                rows.append(
                    {
                        "k1": float(k1),
                        "k2": float(k2),
                        "m1_re": float(m[1].real),
                        "m1_im": float(m[1].imag),
                        "m2_re": float(m[2].real),
                        "m2_im": float(m[2].imag),
                        "abs_m1": float(abs(m[1])),
                        "abs_m2": float(abs(m[2])),
                        "relative_phase": relative_phase(m),
                        "qmi_rich_exact": float(stats["qmi_rich"]),
                    }
                )
        _write_csv(path, list(rows[0]), rows)
        logger.info("map: %d grid points", len(rows))
    return rows


def write_freedom(
    map_rows: Sequence[Mapping[str, float]],
    section: Mapping[str, Any],
    model: Mapping[str, float],
    out: Path,
) -> list[dict[str, Any]]:
    """Range of ``abs_m2`` in each bin of ``abs_m1``, written to ``freedom.csv``.

    A bin ``[j w, (j + 1) w)`` of width ``w = FREEDOM_BIN_WIDTH`` is kept when it
    holds at least one grid point. The column ``abs_m2_at_k2_zero`` is the exact
    ``abs_m2`` of the setting ``(k1, 0)`` whose ``abs_m1`` equals the bin centre,
    with ``k1`` from :func:`solve_k1_for_plv`; it is undefined (empty) when the
    solver finds no such ``k1``. This is the only place where a ``ValueError``
    of that solver is caught.

    Parameters
    ----------
    map_rows : sequence of mapping
        Rows returned by :func:`write_map`.
    section : mapping
        Resolved configuration section; reads ``solver.k1_search_max``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`.
    out : Path
        Output directory.

    Returns
    -------
    rows : list of dict
        One row per kept bin, in increasing ``abs_m1``.
    """
    path = out / "freedom.csv"
    with _Stage("freedom", path):
        solver_kwargs = _select(model, SOLVER_KEYS)
        k1_search_max = float(section["solver"]["k1_search_max"])
        abs_m1 = np.array([r["abs_m1"] for r in map_rows])
        abs_m2 = np.array([r["abs_m2"] for r in map_rows])
        bins = np.floor(abs_m1 / FREEDOM_BIN_WIDTH).astype(int)
        rows = []
        for j in np.unique(bins):
            inside = bins == j
            low = round(float(j) * FREEDOM_BIN_WIDTH, BIN_EDGE_DECIMALS)
            high = round(float(j + 1) * FREEDOM_BIN_WIDTH, BIN_EDGE_DECIMALS)
            centre = 0.5 * (low + high)
            try:
                k1 = solve_k1_for_plv(
                    centre, 0.0, k1_search_max=k1_search_max, **solver_kwargs
                )
                at_zero = float(
                    abs(
                        stationary_moments(
                            k1, 0.0, max_harmonic=MAX_HARMONIC, **solver_kwargs
                        )[2]
                    )
                )
            except ValueError as err:
                logger.info(
                    "freedom: bin centre |m1|=%.3f not reached at k2=0 (%s)",
                    centre,
                    err,
                )
                at_zero = float("nan")
            rows.append(
                {
                    "abs_m1_low": low,
                    "abs_m1_high": high,
                    "n_points": int(np.count_nonzero(inside)),
                    "abs_m2_min": float(abs_m2[inside].min()),
                    "abs_m2_max": float(abs_m2[inside].max()),
                    "abs_m2_at_k2_zero": at_zero,
                }
            )
        _write_csv(path, list(rows[0]), rows)
        logger.info("freedom: %d bins of abs_m1", len(rows))
    return rows


def write_continuity(
    section: Mapping[str, Any], model: Mapping[str, float], out: Path
) -> dict[str, Any]:
    """Rank correlations along the ``k1`` grid at ``k2 = 0``, written to ``continuity.json``.

    Parameters
    ----------
    section : mapping
        Resolved configuration section; reads ``map.k1`` and
        ``protocol.min_first_moment``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`.
    out : Path
        Output directory.

    Returns
    -------
    result : dict
        ``k1`` (the grid), ``abs_m1``, ``abs_m2`` and ``qmi_rich`` along it, and
        ``spearman_abs_m1_abs_m2``, ``spearman_abs_m1_qmi_rich``, the Spearman
        rank correlations of :func:`scipy.stats.spearmanr`, and
        ``ranks_identical_abs_m1_abs_m2``, ``ranks_identical_abs_m1_qmi_rich``,
        true when the rank vectors of :func:`scipy.stats.rankdata` are equal.
    """
    path = out / "continuity.json"
    with _Stage("continuity", path):
        solver_kwargs = _select(model, SOLVER_KEYS)
        min_first_moment = float(section["protocol"]["min_first_moment"])
        k1_grid = _grid(section["map"]["k1"])
        abs_m1, abs_m2, qmi = [], [], []
        for k1 in k1_grid:
            m = stationary_moments(
                float(k1), 0.0, max_harmonic=MAX_HARMONIC, **solver_kwargs
            )
            stats = long_run_statistics(
                m, weight_sets={}, min_first_moment=min_first_moment
            )
            abs_m1.append(float(abs(m[1])))
            abs_m2.append(float(abs(m[2])))
            qmi.append(float(stats["qmi_rich"]))
        result = {
            "k2": 0.0,
            "k1": [float(k) for k in k1_grid],
            "abs_m1": abs_m1,
            "abs_m2": abs_m2,
            "qmi_rich": qmi,
            "spearman_abs_m1_abs_m2": float(spearmanr(abs_m1, abs_m2).statistic),
            "spearman_abs_m1_qmi_rich": float(spearmanr(abs_m1, qmi).statistic),
            # Exact statement of rank identity (see "Rank identity" in the module
            # docstring): spearmanr can round below 1 for identical ranks.
            "ranks_identical_abs_m1_abs_m2": bool(
                np.array_equal(rankdata(abs_m1), rankdata(abs_m2))
            ),
            "ranks_identical_abs_m1_qmi_rich": bool(
                np.array_equal(rankdata(abs_m1), rankdata(qmi))
            ),
        }
        _write_json(path, result)
        logger.info(
            "continuity: Spearman(|m1|, |m2|) = %r, Spearman(|m1|, qmi_rich) = %r",
            result["spearman_abs_m1_abs_m2"],
            result["spearman_abs_m1_qmi_rich"],
        )
    return result


def mean_modulus_and_se(z: np.ndarray) -> tuple[float, float]:
    """Modulus of the mean of complex values and the standard error of that modulus.

    With :math:`\\bar z` the mean of the ``n`` values and :math:`u = \\bar z/|\\bar z|`
    its direction, the standard error of :math:`|\\bar z|` is, to first order,
    that of the projection of the values on :math:`u`:
    ``se = std(Re(z * conj(u)), ddof=1) / sqrt(n)``. Scatter across the direction
    of the mean moves the modulus only to second order and does not enter.

    Parameters
    ----------
    z : ndarray of complex, shape (n,)
        One complex value per record, for example the sample circular moment
        :math:`\\langle e^{ik\\Delta}\\rangle_t` of each record. Dimensionless.

    Returns
    -------
    modulus : float
        :math:`|\\bar z|`, dimensionless.
    se : float
        Standard error of ``modulus``, dimensionless. NaN, without warning, when
        the mean is exactly zero, since it then has no direction to project on.

    Raises
    ------
    ValueError
        If ``z`` holds fewer than two values.

    Notes
    -----
    When to use: to read how far the modulus of a mean of complex moments lies
    from an exact value, in units of its own sampling scatter, as in
    ``timestep_check.csv``. The first-order formula holds when :math:`|\\bar z|`
    is well above ``se``. When it is not, as for the first moment at
    :math:`K_1 = 0` whose exact value is zero, the modulus of the mean is biased
    upward (a modulus is never negative) and a gap expressed in this standard
    error is not meaningful. This is the delta method applied to the modulus;
    no source is cited because the derivation is one line.

    Examples
    --------
    >>> import numpy as np
    >>> mean_modulus_and_se(np.array([1 + 1j, 1 - 1j]))
    (1.0, 0.0)
    """
    z = np.asarray(z, dtype=complex).ravel()
    if z.size < 2:
        raise ValueError(f"at least two values are needed, got {z.size}")
    mean = complex(np.mean(z))
    modulus = abs(mean)
    if modulus == 0.0:
        return 0.0, float("nan")
    direction = mean / modulus
    projected = (z * np.conj(direction)).real
    se = float(np.std(projected, ddof=1) / np.sqrt(z.size))
    return float(modulus), se


def _gap_in_se(simulated: float, exact: float, se: float) -> float:
    """``(simulated - exact) / se``; NaN when the standard error is zero or NaN."""
    if se == 0.0 or math.isnan(se):
        return float("nan")
    return (simulated - exact) / se


def write_timestep_check(
    section: Mapping[str, Any], model: Mapping[str, float], seed: int, out: Path
) -> list[dict[str, float]]:
    """Simulated against exact moduli at each time step, written to ``timestep_check.csv``.

    For each setting and each time step, ``timestep_check.n_records`` records of
    ``timestep_check.duration_s`` seconds are simulated in chunks of at most
    ``timestep_check.records_per_chunk`` records, so that memory holds the phases
    of one chunk only. Each record is reduced at once to its complex sample
    moments (one complex number per record and harmonic), and only those are
    kept. The simulated modulus is the modulus of their mean over all records,
    and its ``se`` is the first-order standard error of that modulus from
    :func:`mean_modulus_and_se`, ``std(Re(z * conj(u)), ddof=1) / sqrt(n)`` with
    ``u`` the direction of the mean. The columns ``abs_m1_gap_se`` and
    ``abs_m2_gap_se`` give ``(sim - exact) / se``, with ``exact`` the value of
    the continuous-time model. The columns ``abs_m1_exact_discrete`` and
    ``abs_m2_exact_discrete`` give the moduli of
    :func:`qot_course.quantum_ot.biharmonic.discretised_stationary_moments` at
    the row's ``dt``, the value the simulated modulus converges to at that step,
    and ``abs_m1_gap_discrete_se``, ``abs_m2_gap_discrete_se`` the simulated
    modulus minus that value, in the same standard errors. Where the modulus of
    the mean is not well above ``se``, as for the first moment of the setting with
    ``k1 = 0``, the modulus is biased upward and its gap in standard errors is
    not meaningful. When the mean is exactly zero, ``se`` and the gap are
    undefined (empty cells).

    Parameters
    ----------
    section : mapping
        Resolved configuration section; reads ``timestep_check``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`; its ``dt`` is replaced by each
        configured time step.
    seed : int
        Root of the seed tree of the check (see "Seeds" in the module docstring).
    out : Path
        Output directory.

    Returns
    -------
    rows : list of dict
        One row per setting and time step, setting major.
    """
    path = out / "timestep_check.csv"
    with _Stage("timestep_check", path):
        check = section["timestep_check"]
        settings = [(float(k1), float(k2)) for k1, k2 in check["settings"]]
        dt_values = [float(dt) for dt in check["dt_values"]]
        n_records = int(check["n_records"])
        records_per_chunk = int(check["records_per_chunk"])
        # A standard error needs at least two records (structural).
        if n_records < 2:
            raise ValueError(
                f"timestep_check.n_records must be at least 2, got {n_records}"
            )
        if records_per_chunk < 1:
            raise ValueError(
                f"timestep_check.records_per_chunk must be at least 1, got {records_per_chunk}"
            )
        duration = float(check["duration_s"])
        solver_kwargs = _select(model, SOLVER_KEYS)
        # Chunk sizes: full chunks, then the remainder (the last chunk may be smaller).
        chunk_sizes = [
            min(records_per_chunk, n_records - first)
            for first in range(0, n_records, records_per_chunk)
        ]
        rows = []
        for setting_index, (k1, k2) in enumerate(settings):
            exact = stationary_moments(
                k1, k2, max_harmonic=MAX_HARMONIC, **solver_kwargs
            )
            for dt_index, dt in enumerate(dt_values):
                start = time.perf_counter()
                simulator_kwargs = {**_select(model, SIMULATOR_KEYS), "dt": dt}
                # One complex sample moment per record and harmonic (rows 1 and 2).
                record_moments = np.empty((MAX_HARMONIC + 1, n_records), dtype=complex)
                first = 0
                for chunk_index, chunk_size in enumerate(chunk_sizes):
                    # Child chunk_index of child dt_index of child setting_index of
                    # SeedSequence(seed) (see "Seeds" in the module docstring).
                    chunk_seed = np.random.SeedSequence(
                        seed, spawn_key=(setting_index, dt_index, chunk_index)
                    )
                    theta_1, theta_2 = simulate_biharmonic_dyad(
                        k1,
                        k2,
                        n_records=chunk_size,
                        duration=duration,
                        seed=chunk_seed,
                        **simulator_kwargs,
                    )
                    delta = theta_1 - theta_2
                    del theta_1, theta_2
                    for harmonic in range(1, MAX_HARMONIC + 1):
                        record_moments[harmonic, first : first + chunk_size] = np.mean(
                            np.exp(1j * harmonic * delta), axis=1
                        )
                    del delta
                    first += chunk_size
                abs_m1, se1 = mean_modulus_and_se(record_moments[1])
                abs_m2, se2 = mean_modulus_and_se(record_moments[2])
                abs_m1_exact, abs_m2_exact = float(abs(exact[1])), float(abs(exact[2]))
                exact_discrete = discretised_stationary_moments(
                    k1, k2, dt=dt, max_harmonic=MAX_HARMONIC, **solver_kwargs
                )
                abs_m1_discrete = float(abs(exact_discrete[1]))
                abs_m2_discrete = float(abs(exact_discrete[2]))
                row = {
                    "k1": k1,
                    "k2": k2,
                    "dt": dt,
                    "n_records": n_records,
                    "abs_m1_exact": abs_m1_exact,
                    "abs_m1_sim": abs_m1,
                    "abs_m1_se": se1,
                    "abs_m1_gap_se": _gap_in_se(abs_m1, abs_m1_exact, se1),
                    "abs_m1_exact_discrete": abs_m1_discrete,
                    "abs_m1_gap_discrete_se": _gap_in_se(abs_m1, abs_m1_discrete, se1),
                    "abs_m2_exact": abs_m2_exact,
                    "abs_m2_sim": abs_m2,
                    "abs_m2_se": se2,
                    "abs_m2_gap_se": _gap_in_se(abs_m2, abs_m2_exact, se2),
                    "abs_m2_exact_discrete": abs_m2_discrete,
                    "abs_m2_gap_discrete_se": _gap_in_se(abs_m2, abs_m2_discrete, se2),
                }
                rows.append(row)
                logger.info(
                    "timestep_check: K1=%g K2=%g dt=%g, %d records in %d chunks: "
                    "|m1| %.4f sim vs %.4f exact (gap %.2f se) and %.4f scheme (gap "
                    "%.2f se), |m2| %.4f sim vs %.4f exact (gap %.2f se) and %.4f "
                    "scheme (gap %.2f se), %.1f s",
                    k1,
                    k2,
                    dt,
                    n_records,
                    len(chunk_sizes),
                    row["abs_m1_sim"],
                    abs_m1_exact,
                    row["abs_m1_gap_se"],
                    abs_m1_discrete,
                    row["abs_m1_gap_discrete_se"],
                    row["abs_m2_sim"],
                    abs_m2_exact,
                    row["abs_m2_gap_se"],
                    abs_m2_discrete,
                    row["abs_m2_gap_discrete_se"],
                    time.perf_counter() - start,
                )
        _write_csv(path, list(rows[0]), rows)
    return rows


def plan_comparisons(
    section: Mapping[str, Any], model: Mapping[str, float]
) -> list[PlannedComparison]:
    """Solve the settings of every comparison, in the fixed numbering order.

    The matched settings of ``equal_plv`` and ``equal_moduli`` are solved on
    the stationary law of the scheme simulated at the model's ``dt``, and each
    records ``matched_on`` and ``matched_dt``; the residual of the equal-moduli
    pair is its residual on that law. A ``ValueError`` of
    :func:`solve_k1_for_plv` in the equal-phase-locking comparisons is not
    caught: a missing matched comparison stops the run with the solver's
    message. An equal-moduli pair that is not found within the
    bound is kept in the plan with ``condition_b = None`` and a reason that
    gives the closest partner and its residual.

    Parameters
    ----------
    section : mapping
        Resolved configuration section; reads ``detection``, ``equal_plv``,
        ``equal_moduli``, ``trap`` and ``solver``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`.

    Returns
    -------
    plan : list of PlannedComparison
        Detection, equal phase-locking value, equal moduli and trap, in that order.
    """
    solver_kwargs = _select(model, SOLVER_KEYS)
    k1_search_max = float(section["solver"]["k1_search_max"])
    # The pairs are compared on simulated records, so they are matched on the
    # stationary law of the scheme at the simulation step.
    dt = float(model["dt"])
    matched = {"matched_on": MATCHED_ON_SCHEME, "matched_dt": dt}
    uncoupled = (0.0, 0.0)
    plan: list[PlannedComparison] = []

    for entry_index, k2 in enumerate(section["detection"]["k2_values"]):
        k2 = float(k2)
        plan.append(
            PlannedComparison(
                "detection",
                entry_index,
                f"k2={k2:g}",
                uncoupled,
                (0.0, k2),
                solved={"condition_a": uncoupled, "condition_b": (0.0, k2)},
            )
        )

    for entry_index, entry in enumerate(section["equal_plv"]):
        target, k2 = float(entry["target"]), float(entry["k2"])
        k1_a = solve_k1_for_plv(
            target, 0.0, k1_search_max=k1_search_max, dt=dt, **solver_kwargs
        )
        k1_b = solve_k1_for_plv(
            target, k2, k1_search_max=k1_search_max, dt=dt, **solver_kwargs
        )
        logger.info(
            "equal_plv: |m1|=%g of the scheme at dt=%g reached at (K1=%.6f, K2=0) "
            "and (K1=%.6f, K2=%g)",
            target,
            dt,
            k1_a,
            k1_b,
            k2,
        )
        plan.append(
            PlannedComparison(
                "equal_plv",
                entry_index,
                f"target={target:g},k2={k2:g}",
                (k1_a, 0.0),
                (k1_b, k2),
                solved={
                    "target": target,
                    "condition_a": (k1_a, 0.0),
                    "condition_b": (k1_b, k2),
                    **matched,
                },
            )
        )

    eq = section["equal_moduli"]
    pair = solve_equal_moduli_pair(
        float(eq["k1_ref"]),
        float(eq["k2_ref"]),
        k2_bound=float(eq["k2_bound"]),
        tolerance=float(eq["tolerance"]),
        k1_search_max=k1_search_max,
        dt=dt,
        **solver_kwargs,
    )
    label = f"k1_ref={pair.k1_ref:g},k2_ref={pair.k2_ref:g}"
    reference = (pair.k1_ref, pair.k2_ref)
    if pair.found:
        logger.info(
            "equal_moduli: partner of (K1=%g, K2=%g) on the scheme at dt=%g is "
            "(K1=%.6f, K2=%.6f), residual %.2e",
            pair.k1_ref,
            pair.k2_ref,
            dt,
            pair.k1,
            pair.k2,
            pair.residual,
        )
        plan.append(
            PlannedComparison(
                "equal_moduli",
                0,
                label,
                reference,
                (pair.k1, pair.k2),
                solved={"pair": asdict(pair), **matched},
            )
        )
    else:
        reason = (
            f"no partner of (k1={pair.k1_ref:g}, k2={pair.k2_ref:g}) with equal |m1| "
            f"and |m2| on the scheme at dt={dt:g} and |k2| <= "
            f"{float(eq['k2_bound']):g} within tolerance "
            f"{float(eq['tolerance']):g}; the closest partner found is "
            f"(k1={pair.k1:.6f}, k2={pair.k2:.6g}) with residual {pair.residual:.3e}"
        )
        plan.append(
            PlannedComparison(
                "equal_moduli",
                0,
                label,
                reference,
                None,
                solved={"pair": asdict(pair), **matched},
                not_made_reason=reason,
            )
        )

    trap_k2 = float(section["trap"]["k2"])
    plan.append(
        PlannedComparison(
            "trap",
            0,
            f"k2={trap_k2:g}",
            uncoupled,
            (0.0, trap_k2),
            solved={"condition_a": uncoupled, "condition_b": (0.0, trap_k2)},
        )
    )
    return plan


def write_comparisons(
    plan: Sequence[PlannedComparison],
    section: Mapping[str, Any],
    model: Mapping[str, float],
    out: Path,
) -> list[dict[str, Any]]:
    """Run every planned comparison at every duration and seed, written to ``comparisons.csv``.

    Parameters
    ----------
    plan : sequence of PlannedComparison
        Output of :func:`plan_comparisons`; comparisons with ``condition_b``
        None are skipped but keep their seed slot.
    section : mapping
        Resolved configuration section; reads ``protocol`` and ``weight_sets``.
    model : mapping of str to float
        Model mapping of :func:`_model_mapping`.
    out : Path
        Output directory.

    Returns
    -------
    rows : list of dict
        The rows of :func:`compare_conditions`, each prefixed with
        :data:`COMPARISON_PREFIX`.
    """
    path = out / "comparisons.csv"
    with _Stage("comparisons", path):
        protocol = section["protocol"]
        durations = [float(d) for d in protocol["durations_s"]]
        weight_sets = {
            k: [float(w) for w in v] for k, v in section["weight_sets"].items()
        }
        n_made = sum(c.condition_b is not None for c in plan)
        n_total = n_made * len(durations) * len(protocol["master_seeds"])
        done = 0
        rows: list[dict[str, Any]] = []
        for master_seed in protocol["master_seeds"]:
            master_seed = int(master_seed)
            for comparison in plan:
                if comparison.condition_b is None:
                    continue
                experiment_index = EXPERIMENTS.index(comparison.experiment)
                for duration_index, duration in enumerate(durations):
                    # Child (experiment, entry, duration) of SeedSequence(master_seed),
                    # built directly so that no spawn counter is shared between
                    # comparisons (see "Seeds" in the module docstring).
                    duration_seed = np.random.SeedSequence(
                        master_seed,
                        spawn_key=(experiment_index, comparison.entry, duration_index),
                    )
                    start = time.perf_counter()
                    result = compare_conditions(
                        comparison.condition_a,
                        comparison.condition_b,
                        duration=duration,
                        n_records=int(protocol["n_records"]),
                        seed=duration_seed,
                        model=model,
                        weight_sets=weight_sets,
                        false_alarm_rate=float(protocol["false_alarm_rate"]),
                        min_first_moment=float(protocol["min_first_moment"]),
                    )
                    prefix = {
                        "experiment": comparison.experiment,
                        "label": comparison.label,
                        "k1_a": comparison.condition_a[0],
                        "k2_a": comparison.condition_a[1],
                        "k1_b": comparison.condition_b[0],
                        "k2_b": comparison.condition_b[1],
                        "duration_s": duration,
                        "master_seed": master_seed,
                    }
                    rows.extend({**prefix, **row} for row in result)
                    done += 1
                    logger.info(
                        "comparison %d/%d done: %s %s, %g s, master seed %d, in %.1f s",
                        done,
                        n_total,
                        comparison.experiment,
                        comparison.label,
                        duration,
                        master_seed,
                        time.perf_counter() - start,
                    )
        _write_csv(path, COMPARISON_PREFIX + COMPARISON_KEYS, rows)
    return rows


def write_summary(rows: Sequence[Mapping[str, Any]], out: Path) -> list[dict[str, Any]]:
    """Comparisons summarised over master seeds, written to ``summary.csv``.

    Means, minima and maxima are taken over the seeds at which the value is
    defined; ``n_seeds`` counts the seeds, and ``n_seeds_score_defined``,
    ``n_seeds_auc_defined`` and ``n_seeds_detection_defined`` count those at
    which the score, the area under the curve and the detection rate are
    defined. ``exact_a``, ``exact_b``, ``exact_discrete_a`` and
    ``exact_discrete_b`` do not depend on the seed. ``mean_n_valid`` and
    ``mean_n_null`` are the means over the seeds of ``n_valid`` and ``n_null``;
    ``mean_null_sd_a`` and ``mean_null_sd_b`` are the means over the seeds of
    the finite values of ``null_sd_a`` and ``null_sd_b``. Undefined everywhere
    gives an empty cell.

    Parameters
    ----------
    rows : sequence of mapping
        Rows returned by :func:`write_comparisons`.
    out : Path
        Output directory.

    Returns
    -------
    summary : list of dict
        One row per ``experiment, label, duration_s, statistic``, in the order
        of first appearance.
    """
    path = out / "summary.csv"
    with _Stage("summary", path):
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in rows:
            key = (row["experiment"], row["label"], row["duration_s"], row["statistic"])
            groups.setdefault(key, []).append(row)
        summary = []
        for (experiment, label, duration, statistic), group in groups.items():
            out_row: dict[str, Any] = {
                "experiment": experiment,
                "label": label,
                "duration_s": duration,
                "statistic": statistic,
                "n_seeds": len(group),
            }
            summaries = {
                name: _finite_summary([r[name] for r in group])
                for name in ("score", "auc", "detection_rate")
            }
            out_row["n_seeds_score_defined"] = summaries["score"][3]
            out_row["n_seeds_auc_defined"] = summaries["auc"][3]
            out_row["n_seeds_detection_defined"] = summaries["detection_rate"][3]
            for name, (mean, low, high, _) in summaries.items():
                out_row[f"{name}_mean"] = mean
                out_row[f"{name}_min"] = low
                out_row[f"{name}_max"] = high
            out_row["mean_a"] = _finite_summary([r["mean_a"] for r in group])[0]
            out_row["exact_a"] = group[0]["exact_a"]
            out_row["mean_b"] = _finite_summary([r["mean_b"] for r in group])[0]
            out_row["exact_b"] = group[0]["exact_b"]
            out_row["exact_discrete_a"] = group[0]["exact_discrete_a"]
            out_row["exact_discrete_b"] = group[0]["exact_discrete_b"]
            out_row["mean_n_valid"] = float(np.mean([r["n_valid"] for r in group]))
            out_row["mean_n_null"] = float(np.mean([r["n_null"] for r in group]))
            out_row["mean_null_sd_a"] = _finite_summary(
                [r["null_sd_a"] for r in group]
            )[0]
            out_row["mean_null_sd_b"] = _finite_summary(
                [r["null_sd_b"] for r in group]
            )[0]
            summary.append(out_row)
        fieldnames = list(summary[0]) if summary else ["experiment"]
        _write_csv(path, fieldnames, summary)
    return summary


def write_reliability(
    summary: Sequence[Mapping[str, Any]], section: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """Shortest reliable duration per comparison and statistic, written to ``reliability.csv``.

    The shortest reliable duration is the shortest configured duration whose
    mean detection rate over seeds reaches ``protocol.reliability_threshold``.
    The threshold is the usual power convention, not a derived quantity; the
    column ``threshold_is_a_convention`` says so in the file itself. The columns
    ``mean_n_valid`` and ``mean_n_null`` give the values of ``summary.csv`` at
    the reported duration, because a relative-phase row can be reported as
    reliable while resting on one or two records; they are empty when no
    duration is reported.

    The experiment ``trap`` has no row. It is reported as a documented limit of
    the bench, outside any comparison, because its records do not explore the
    long-run law, so a shortest reliable duration has no meaning there.

    Parameters
    ----------
    summary : sequence of mapping
        Rows returned by :func:`write_summary`.
    section : mapping
        Resolved configuration section; reads ``protocol.reliability_threshold``.
    out : Path
        Output directory.

    Returns
    -------
    rows : list of dict
        One row per ``experiment, label, statistic``.
    """
    path = out / "reliability.csv"
    with _Stage("reliability", path):
        threshold = float(section["protocol"]["reliability_threshold"])
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in summary:
            if row["experiment"] == "trap":
                continue
            key = (row["experiment"], row["label"], row["statistic"])
            groups.setdefault(key, []).append(row)
        rows = []
        for (experiment, label, statistic), group in groups.items():
            reliable = [
                r["duration_s"]
                for r in group
                if not _is_undefined(r["detection_rate_mean"])
                and r["detection_rate_mean"] >= threshold
            ]
            shortest = min(reliable) if reliable else None
            at_shortest = next((r for r in group if r["duration_s"] == shortest), None)
            rows.append(
                {
                    "experiment": experiment,
                    "label": label,
                    "statistic": statistic,
                    "shortest_reliable_duration_s": shortest,
                    "reliability_threshold": threshold,
                    "threshold_is_a_convention": True,
                    "mean_n_valid": at_shortest["mean_n_valid"]
                    if at_shortest
                    else None,
                    "mean_n_null": at_shortest["mean_n_null"] if at_shortest else None,
                }
            )
        fieldnames = (
            "experiment",
            "label",
            "statistic",
            "shortest_reliable_duration_s",
            "reliability_threshold",
            "threshold_is_a_convention",
            "mean_n_valid",
            "mean_n_null",
        )
        _write_csv(path, fieldnames, rows)
    return rows


def portable_path(path: Path) -> str:
    """Path of a file as it is recorded in the results.

    A path inside the repository is written relative to the repository root,
    with forward slashes. The results are committed and can be published, and
    an absolute path would record the directory layout of the machine that
    made the run instead of the location of the file in the project. A path
    outside the repository has no such relative form and is written as given.

    Parameters
    ----------
    path : Path
        File whose location is recorded.

    Returns
    -------
    recorded : str
        Repository-relative POSIX path, or ``str(path)`` when the file lies
        outside the repository.

    Examples
    --------
    >>> portable_path(DEFAULT_CONFIG)
    'config/base.yaml'
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_commit() -> dict[str, Any]:
    """Commit hash of the repository and whether the working tree has changes."""
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as err:
        # A missing git binary must not lose a finished run: record it as unknown.
        logger.warning("git commit unknown: git could not be run (%s)", err)
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        logger.warning(
            "git commit unknown: git rev-parse exited with %d (%s)",
            head.returncode,
            head.stderr.strip(),
        )
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def write_metadata(
    section: Mapping[str, Any],
    *,
    quick: bool,
    config_path: Path,
    timestep_seed: int,
    plan: Sequence[PlannedComparison],
    start: float,
    out: Path,
) -> dict[str, Any]:
    """Provenance of the run, written to ``run_metadata.json``.

    Parameters
    ----------
    section : mapping
        Resolved configuration section (after the ``quick`` merge, without the
        ``quick`` sub-section itself).
    quick : bool
        Whether ``--quick`` was given.
    config_path : Path
        Configuration file read. It is recorded through :func:`portable_path`.
    timestep_seed : int
        Root seed of the time-step check.
    plan : sequence of PlannedComparison
        Output of :func:`plan_comparisons`.
    start : float
        :func:`time.perf_counter` value at the start of the run.
    out : Path
        Output directory.

    Returns
    -------
    metadata : dict
        The content written.
    """
    path = out / "run_metadata.json"
    with _Stage("run_metadata", path):
        git = _git_commit()
        metadata = {
            "config": section,
            "config_path": portable_path(config_path),
            "quick": quick,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "utc_date": datetime.now(timezone.utc).isoformat(),
            "timestep_check_seed": timestep_seed,
            "solved_settings": [
                {
                    "experiment": c.experiment,
                    "label": c.label,
                    "made": c.condition_b is not None,
                    # First two levels of the spawn key; the third is the duration.
                    "seed_spawn_key_prefix": [EXPERIMENTS.index(c.experiment), c.entry],
                    **c.solved,
                }
                for c in plan
            ],
            "comparisons_not_made": [
                {
                    "experiment": c.experiment,
                    "label": c.label,
                    "reason": c.not_made_reason,
                }
                for c in plan
                if c.condition_b is None
            ],
            "wall_time_s": time.perf_counter() - start,
        }
        _write_json(path, metadata)
    return metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the free-second-harmonic bench and write its result files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="merge the 'quick' sub-section over the section (smoke-sized run)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the whole bench from the configuration and write the eight result files.

    Parameters
    ----------
    argv : sequence of str or None, default None
        Command-line arguments without the program name; None reads
        ``sys.argv``. Options: ``--config`` (default ``config/base.yaml`` of the
        repository), ``--out`` (default ``results/second_harmonic_bench`` of the
        repository) and ``--quick`` (merges the ``quick`` sub-section of the
        configuration section over the section with
        :func:`qot_course.utils.config.deep_merge`).

    Returns
    -------
    status : int
        0 on success. A matched comparison that cannot be solved raises the
        solver's ``ValueError`` instead of returning.

    Notes
    -----
    When to use: to produce the result files of the bench; ``--quick`` runs a
    reduced grid and protocol, measured at about 7 s inside this function (about
    8 s for the whole command) on 2026-11-11, and is what the smoke test calls.
    About 6 s of it is the solving of the matched pairs on the discretised
    scheme, whose moments cost a few milliseconds each against a fraction of a
    millisecond in continuous time.
    The model is the two-harmonic phase interaction of Haken, Kelso and Bunz
    (1985), DOI: 10.1007/BF00336922, and the exact moments come from the
    stationary Fokker--Planck solution of Risken (1996),
    DOI: 10.1007/978-3-642-61544-3. Seeds and undefined numbers follow the
    rules of the module docstring.

    Examples
    --------
    >>> import tempfile
    >>> main(["--quick", "--out", tempfile.mkdtemp()])  # doctest: +SKIP
    0
    """
    start = time.perf_counter()
    args = _parse_args(argv)
    full_config = load_config(args.config)
    section = copy.deepcopy(full_config[CONFIG_SECTION])
    quick_overrides = section.pop("quick", None)
    if args.quick:
        if quick_overrides is None:
            raise ValueError(
                f"--quick given but {args.config} has no '{CONFIG_SECTION}.quick'"
            )
        section = deep_merge(section, quick_overrides)
    timestep_seed = int(full_config["seed"])
    model = _model_mapping(section)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Free-second-harmonic bench: config %s%s, output %s",
        args.config,
        " (quick)" if args.quick else "",
        out,
    )

    map_rows = write_map(section, model, out)
    write_freedom(map_rows, section, model, out)
    write_continuity(section, model, out)
    write_timestep_check(section, model, timestep_seed, out)
    logger.info("Solving the settings of every comparison")
    plan = plan_comparisons(section, model)
    rows = write_comparisons(plan, section, model, out)
    summary = write_summary(rows, out)
    write_reliability(summary, section, out)
    write_metadata(
        section,
        quick=args.quick,
        config_path=args.config,
        timestep_seed=timestep_seed,
        plan=plan,
        start=start,
        out=out,
    )
    logger.info("Bench finished in %.1f s", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
