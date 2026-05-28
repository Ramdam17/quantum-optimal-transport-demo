import numpy as np
import ot
import pytest

from qot_course.transport.discrete import (
    cityblock_cost,
    discrete_ot_plan,
    is_doubly_stochastic,
    is_transportation_polytope_member,
    squared_euclidean_cost,
    transport_cost,
)


# ----------------------------- Cost matrices ------------------------------- #
def test_squared_euclidean_cost_simple():
    cost = squared_euclidean_cost([0, 4], [1, 2, 3])
    np.testing.assert_allclose(cost, [[1, 4, 9], [9, 4, 1]])


def test_cityblock_cost_simple():
    cost = cityblock_cost([0, 4], [1, 2, 3])
    np.testing.assert_allclose(cost, [[1, 2, 3], [3, 2, 1]])


def test_costs_match_pot_for_shared_grid():
    # POT's ot.dist is the reference implementation on a shared grid.
    grid = np.arange(5, dtype=float).reshape(-1, 1)
    np.testing.assert_allclose(
        squared_euclidean_cost(grid.ravel(), grid.ravel()),
        ot.dist(grid, grid, metric="sqeuclidean"),
    )


# ----------------------------- LP solver wrapper --------------------------- #
def test_discrete_ot_plan_lies_in_transportation_polytope():
    source = np.array([0.5, 0.5])
    target = np.array([1 / 3, 1 / 3, 1 / 3])
    cost = squared_euclidean_cost([0, 4], [1, 2, 3])
    plan = discrete_ot_plan(source, target, cost)
    assert is_transportation_polytope_member(plan, source, target)


def test_one_to_many_splits_mass():
    """Monge-fails example: a single source atom must split between two targets."""
    source = np.array([1.0])
    target = np.array([0.5, 0.5])
    cost = squared_euclidean_cost([0.0], [-1.0, 1.0])
    plan = discrete_ot_plan(source, target, cost)
    np.testing.assert_allclose(plan, [[0.5, 0.5]])
    # No deterministic Monge map could yield this --- yet the LP solves it.


def test_two_to_three_mass_splitting_yields_known_plan():
    # Hand-computed optimum for the syllabus's canonical S8 example.
    source = np.array([0.5, 0.5])
    target = np.array([1 / 3, 1 / 3, 1 / 3])
    cost = squared_euclidean_cost([0.0, 4.0], [1.0, 2.0, 3.0])
    plan = discrete_ot_plan(source, target, cost)
    expected = np.array([[1 / 3, 1 / 6, 0.0], [0.0, 1 / 6, 1 / 3]])
    np.testing.assert_allclose(plan, expected, atol=1e-9)
    assert transport_cost(plan, cost) == pytest.approx(2.0, abs=1e-9)


def test_emd2_matches_transport_cost_on_the_plan():
    source = np.array([0.4, 0.3, 0.3])
    target = np.array([0.2, 0.5, 0.3])
    cost = squared_euclidean_cost([1, 3, 5], [2, 4, 6])
    plan = discrete_ot_plan(source, target, cost)
    assert transport_cost(plan, cost) == pytest.approx(ot.emd2(source, target, cost))


# ----------------------------- Birkhoff polytope --------------------------- #
def test_uniform_equal_size_yields_a_permutation():
    """For uniform equal-size source/target the LP vertex is a permutation matrix.

    Optimal: match by *rank* (1-D sort-and-match). With source [0, 1, 2] and target
    [3, 1, 5], the rank-1 target is at index 1 (value 1), so source 0 -> target 1,
    source 1 -> target 0, source 2 -> target 2.
    """
    source = np.full(3, 1 / 3)
    target = np.full(3, 1 / 3)
    cost = squared_euclidean_cost([0, 1, 2], [3, 1, 5])
    plan = discrete_ot_plan(source, target, cost)
    expected = (1 / 3) * np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float
    )
    np.testing.assert_allclose(plan, expected, atol=1e-9)
    # The (3 * plan) is a 0/1 permutation matrix --- doubly stochastic.
    assert is_doubly_stochastic(3 * plan)


def test_birkhoff_decomposition_example():
    """Birkhoff--von Neumann theorem demo: an interior doubly-stochastic matrix
    is a convex combination of permutation matrices."""
    interior = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    identity = np.eye(3)
    cycle = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    reconstructed = 0.5 * identity + 0.5 * cycle
    np.testing.assert_allclose(interior, reconstructed)
    assert is_doubly_stochastic(interior)
    assert is_doubly_stochastic(identity)
    assert is_doubly_stochastic(cycle)


# ----------------------------- Polytope membership ------------------------- #
def test_is_transportation_polytope_member_basic():
    source = np.array([0.5, 0.5])
    target = np.array([0.5, 0.5])
    good_plan = np.array([[0.3, 0.2], [0.2, 0.3]])
    bad_row_sum = np.array([[0.6, 0.2], [0.2, 0.3]])
    bad_negative = np.array([[0.6, -0.1], [-0.1, 0.6]])
    wrong_shape = np.array([[0.5, 0.5, 0.0]])
    assert is_transportation_polytope_member(good_plan, source, target)
    assert not is_transportation_polytope_member(bad_row_sum, source, target)
    assert not is_transportation_polytope_member(bad_negative, source, target)
    assert not is_transportation_polytope_member(wrong_shape, source, target)


def test_is_doubly_stochastic_identity_permutation_and_interior():
    assert is_doubly_stochastic(np.eye(3))
    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    assert is_doubly_stochastic(perm)
    interior = np.full((3, 3), 1 / 3)
    assert is_doubly_stochastic(interior)


def test_is_doubly_stochastic_rejects_non_square_and_negative():
    assert not is_doubly_stochastic(np.array([[0.5, 0.5, 0.0]]))
    assert not is_doubly_stochastic(np.array([[1.5, -0.5], [-0.5, 1.5]]))
