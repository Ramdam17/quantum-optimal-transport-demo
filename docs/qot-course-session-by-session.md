# QOT Course — Session-by-Session Companion

*A companion to the teaser. Sixteen 2-hour hands-on sessions over ~3 months, in four
movements: **M1** Quantum foundations · **M2** Information theory & geometry · **M3**
Classical optimal transport · **M4** Quantum optimal transport (S1 is the kickoff).*

---

### S1 — Teaser & Roadmap *(kickoff)*
- **Objective:** make everyone want the journey, and get every laptop running.
- **Content:** a visual transport demo (mass flowing between shapes), the classical↔quantum dictionary as a map of the trip, course mechanics, environment setup, a first one-line `ot.emd2`.
- **Concepts learned:** what optimal transport *is* intuitively; what "quantizing" it will mean; the dictionary idea.
- **Skills acquired:** set up the Python/`uv` environment; run and read a first OT computation and plot.
- **Challenges:** no heavy math yet — resist over-explaining; make heterogeneous machines all work.

## M1 — Quantum foundations

### S2 — Qubits & states
- **Objective:** build the qubit from zero and understand measurement.
- **Content:** complex amplitudes, the Bloch sphere, superposition, measurement and the Born rule; optionally measure a real qubit.
- **Concepts learned:** state vector, superposition, measurement probabilities, Bloch representation.
- **Skills acquired:** construct and visualize single-qubit states in Qiskit; predict and sample measurement outcomes.
- **Challenges:** letting go of classical intuition (a qubit isn't 0/1 until measured); amplitudes vs probabilities.

### S3 — Density matrices
- **Objective:** introduce the central object of the whole course — the density matrix ρ.
- **Content:** pure vs mixed states, positive-semidefinite + unit trace, von Neumann entropy, purity, fidelity, trace distance; tomography of a real (noisy → slightly *mixed*) qubit.
- **Concepts learned:** density matrix, mixedness, entropy, state distinguishability.
- **Skills acquired:** build ρ from data; compute purity / entropy / fidelity; reconstruct ρ from measurements.
- **Challenges:** *why* we need ρ at all — the "real states are mixed" reveal; the jump from vectors to operators.

### S4 — Composite systems & channels
- **Objective:** combine systems and model how they evolve.
- **Content:** tensor product, partial trace, entanglement, quantum channels (Kraus, CPTP), Stinespring; a Bell state, optionally on hardware.
- **Concepts learned:** tensor product, partial trace (= marginal), entanglement, quantum channel (= noisy Markov map).
- **Skills acquired:** build bipartite states; take partial traces; apply channels; create and measure entanglement.
- **Challenges:** entanglement has no classical analog; partial trace is unintuitive; the direct-sum vs tensor-product distinction (which returns in the capstone).

## M2 — Information theory & geometry *(the spine)*

### S5 — Classical information theory
- **Objective:** the information toolbox underlying everything that follows.
- **Content:** Shannon entropy, Kullback–Leibler divergence, mutual information, conditional MI, transfer entropy (Schreiber); a circumspect note on partial information decomposition.
- **Concepts learned:** entropy, relative entropy (KL), mutual information, directed information / transfer entropy.
- **Skills acquired:** estimate entropies / MI / TE from data; reason carefully about information flow.
- **Challenges:** estimation pitfalls (finite samples, bias); the genuine interpretive subtleties of TE and PID.

### S6 — Information geometry
- **Objective:** see the space of probability distributions as a curved geometry.
- **Content:** the Fisher–Rao metric, statistical manifolds, exponential families, dual connections (Amari); the two geometries of the simplex — information vs transport.
- **Concepts learned:** Fisher information metric, statistical manifold, the information-geometry / transport-geometry duality.
- **Skills acquired:** compute the Fisher metric for simple families; visualize geodesics on the probability simplex.
- **Challenges:** the differential-geometry flavor is abstract; holding "two different geometries on one space" in mind.

### S7 — Quantum information theory
- **Objective:** lift the information toolbox to density matrices.
- **Content:** von Neumann entropy, Umegaki relative entropy (the *quantum KL*), quantum mutual information, quantum conditional entropy (can be **negative**), strong subadditivity; quantum Fisher metrics (Bures/BKM); the unsettled status of quantum transfer entropy.
- **Concepts learned:** quantum relative entropy as the parent quantity; quantum MI; negative conditional entropy as a purely quantum effect.
- **Skills acquired:** compute quantum entropies, relative entropy and MI for ρ; fill the information row of the dictionary.
- **Challenges:** negative conditional entropy is counterintuitive; strong subadditivity is deep; quantum TE is genuinely open.

## M3 — Classical optimal transport

### S8 — Monge → Kantorovich
- **Objective:** state the formal OT problem and its linear-program relaxation.
- **Content:** transport maps vs couplings, marginals, the LP, the Birkhoff polytope, the assignment problem.
- **Concepts learned:** transport plan / coupling, marginal constraints, LP formulation, doubly-stochastic matrices.
- **Skills acquired:** set up and solve a discrete OT linear program (POT/SciPy); extract and read a transport plan.
- **Challenges:** why deterministic Monge maps can fail (mass splitting) and thus why Kantorovich relaxes.

### S9 — Wasserstein distances
- **Objective:** turn OT cost into a true metric and compute it.
- **Content:** the W_p distances, the exact 1D solution (quantiles / sorting), metric properties, McCann displacement interpolation.
- **Concepts learned:** Wasserstein-p distance, the 1D closed form, interpolation between distributions.
- **Skills acquired:** compute W_1 / W_2 (including exact 1D); animate displacement interpolation.
- **Challenges:** intuiting *why* Wasserstein respects geometry where KL does not; the jump from easy-1D to hard-general.

### S10 — Duality & Sinkhorn
- **Objective:** the dual viewpoint, the algorithm that made OT practical, and its link to information geometry.
- **Content:** Kantorovich duality, the 1-Lipschitz dual of W_1, entropic regularization, the Sinkhorn algorithm; **Amari's result: Sinkhorn is where Wasserstein meets KL / Fisher–Rao**.
- **Concepts learned:** LP duality, Kantorovich potentials, entropic regularization, Sinkhorn iterations.
- **Skills acquired:** implement Sinkhorn; tune the regularization ε; connect entropic OT to a KL projection.
- **Challenges:** numerical stability (log-domain); the ε trade-off (blur vs speed); *seeing* the IT↔OT bridge.

### S11 — Gaussians & dynamics
- **Objective:** the closed-form case that opens the door to the quantum world, plus the dynamic view.
- **Content:** the Bures–Wasserstein distance between Gaussians (built from covariance matrices), the Benamou–Brenier fluid formulation, Otto calculus.
- **Concepts learned:** Bures–Wasserstein distance, transport as a flow, the Riemannian view of the space of distributions.
- **Skills acquired:** compute W_2 between Gaussians from means/covariances; reason about geodesics of distributions.
- **Challenges:** covariance-matrix transport is the hinge to ρ; the abstraction of "geometry on distributions."

## M4 — Quantum optimal transport

### S12 — Why QOT
- **Objective:** motivate quantum OT and state precisely what breaks classically.
- **Content:** non-commutativity, the diagonal-collapse lesson, the plus-state vs maximally-mixed example, Trevisan's taxonomy; the dictionary completed.
- **Concepts learned:** why a *diagonal* ρ reduces QOT to classical OT; non-commutativity as the source of quantum-ness; the QOT landscape.
- **Skills acquired:** build the plus-state-vs-mixed example; articulate exactly where classical OT is blind.
- **Challenges:** resisting hype; grasping that "same statistics, different state" is the whole point.

### S13 — Coupling QOT = a semidefinite program
- **Objective:** compute a genuine quantum-Wasserstein distance.
- **Content:** quantum couplings as bipartite ρ with partial-trace marginals, cost `tr(Cρ)`, the SWAP cost → quantum W_2, solved with `cvxpy`; validation on qubits and Gaussians (Cole et al.); optionally QOT between experimentally reconstructed states.
- **Concepts learned:** quantum coupling, the SDP formulation, validation against known cases.
- **Skills acquired:** formulate and solve the QOT SDP in `cvxpy`; sanity-check against closed forms.
- **Challenges:** setting up SDP constraints as partial traces; *trusting* the result via numerical validation (the discipline the old demo lacked).

### S14 — Quantum Sinkhorn
- **Objective:** the scalable, regularized QOT and its information-geometric meaning.
- **Content:** von Neumann-entropy regularization, the dual, the matrix-exponential + partial-trace fixed point (Peyré tensor fields; Pelikh–Gerolin); the quantum Amari bridge; a diffusion-tensor / texture application.
- **Concepts learned:** quantum entropic regularization, the PSD-generalized Sinkhorn, tensor-field transport.
- **Skills acquired:** implement quantum Sinkhorn (matrix exp + partial traces); apply it to tensor-field / imaging data.
- **Challenges:** matrix exponentials and convergence; connecting it back to the classical Sinkhorn of S10.

### S15 — Capstone *(open research problem)*
- **Objective:** assemble everything into a real, unsolved question.
- **Content:** inter-system coupling measured as the quantum relative entropy to the decoupled (block-diagonal) state, complemented by the QOT coupling, on a **synthetic Kuramoto dyad** where the injected coupling is known; honest caveats.
- **Concepts learned:** the direct-sum vs tensor-product trap, the relative-entropy coupling measure, validation against synthetic ground truth.
- **Skills acquired:** build a coupled Kuramoto dyad; compute the coupling measure; verify it recovers the injected coupling.
- **Challenges:** the tensor/direct-sum subtlety; honesty (it must beat PLV / Euclidean baselines); estimation (shrinkage, leakage correction).

### S16 — Frontier & synthesis
- **Objective:** close the loop — see the limits and the open horizon.
- **Content:** the variational-quantum-algorithm limitation result (De Palma et al. 2023, closing the loop with the original broken demo); a taxonomy recap (channels, qubit W_1, Carlen–Maas dynamic QOT); open problems (quantum TE, hyperscanning QOT); optionally run the naive VQE on hardware and watch it underperform.
- **Concepts learned:** why naive variational quantum algorithms underdeliver; the breadth of QOT formulations; what remains open.
- **Skills acquired:** run and critique a VQE; situate any "quantum X" claim; identify concrete next research steps.
- **Challenges:** synthesizing fifteen sessions; intellectual honesty about limits; resisting the urge to overclaim.

---

*Companion to `docs/qot-course-teaser.md`; mirrors the curriculum in
`docs/superpowers/specs/2026-05-27-qot-course-design.md`. Compressible to ~14 sessions by
merging S5+S6 and S15+S16. Available in French on request.*
