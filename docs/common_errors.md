# Common Errors & Surprises

A field guide to the misconceptions and "wait, that's not what I expected" moments this course tends
to produce. If you hit one of these, you are in good company — each is a place where careful
intuition and the actual mathematics pull in different directions. We name the trap, say what is
really going on, and point you to the notebook that works it through.

> For the precise boundaries of what the course does and does not claim, see
> [`scientific_claims_guardrails.md`](scientific_claims_guardrails.md). For running on real IBM
> hardware, see [`hardware_notes.md`](hardware_notes.md).

---

## "PLV already measures the coupling — why bother with quantum optimal transport?"

**The intuition.** The phase-locking value is a clean, well-understood synchrony measure. If two
signals are coupled, PLV sees it — so a quantum measure feels like machinery for its own sake.

**What is really going on.** A coupling measure can only see what your *embedding* hands it. PLV
reads the first circular moment of the phase difference, `⟨e^{iΔθ}⟩`. Two oscillator ensembles can
share an *identical* PLV by construction yet differ in their second moment `⟨e^{2iΔθ}⟩` — and a
richer multi-frequency quantum embedding's mutual information separates them, where PLV and the naive
phase-qubit embedding are blind. The embedding is the dial; the transport objects are fixed.

**Where the course shows it.** `04/13`–`04/16` (the redundancy reveal, the richer embedding, the
discriminating experiment, and the honest accounting).

## "A reconstructed state with fidelity below 1 means the device decohered."

**The intuition.** You prepare `|+⟩`, run tomography, and get fidelity 0.96. The missing 0.04 must
be decoherence.

**What is really going on.** Two effects pull the number below 1, and a single run cannot separate
them: genuine device decoherence, *and* finite-shot estimation. With a few thousand shots per basis,
the estimated Bloch vector is noisy, and the projection back to a physical (positive, unit-trace)
state is a slightly biased estimator — so even a perfect, noiseless device would not return purity
exactly 1. More shots shrink the sampling part and isolate the physical gap.

**Where the course shows it.** `05/03` (tomography and fidelity on real hardware).

## "Quantum mutual information measures entanglement."

**The intuition.** A Bell pair has `I(A:B) = 2` bits and is maximally entangled, so QMI must be an
entanglement measure.

**What is really going on.** Quantum mutual information measures *total* correlation — classical plus
quantum. For a **pure** state it happens to equal twice the entanglement entropy, so for the Bell
pair the "2 bits = maximal entanglement" reading is exact. For a general **mixed** state, QMI also
counts classical correlations, so it is not an entanglement measure on its own.

**Where the course shows it.** `02/10` (quantum mutual information), `02/11` (negative conditional
entropy), `05/04` (QMI on hardware).

## "The entropic plan should always smooth out as the regularisation goes to zero."

**The intuition.** Entropic optimal transport interpolates smoothly; as `ε → 0` you recover the
unregularised plan.

**What is really going on.** Some state pairs have a *pinned* joint entropy. The pair `|+⟩⟨+|` versus
`I/2`, for instance, has its joint von Neumann entropy fixed at 1 bit — so its entropic plan is
`ε`-independent and rank-deficient, and a solver returns a non-finite objective. The cure is to run
entropic / operator-Sinkhorn / Umegaki demonstrations on a **commuting diagonal pair**, where the
smoothing behaves as expected.

**Where the course shows it.** `04/06`–`04/08` (entropic QOT, operator Sinkhorn, the Amari bridge).

## "If quantum optimal transport separates the two ensembles, then only quantum optimal transport can."

**The intuition.** The quantum measure found a difference PLV could not, so the difference is
intrinsically quantum.

**What is really going on.** The difference lives in the second circular moment, which a *classical*
higher-order statistic could also detect. The value of quantum optimal transport here is not unique
detecting power — it is the **unified geometric framework**: one geometry, one pair of measures, with
the embedding as the single dial that sets what the coupling measure can see.

**Where the course shows it.** `04/16` (the honest accounting).

## "More shots will fix the hardware error."

**The intuition.** Hardware results are noisy; averaging over more shots will converge to the right
answer.

**What is really going on.** Shots cure the *statistical* scatter (which shrinks like `1/√shots`),
but a *systematic* bias — read-out asymmetry, gate error — is a floor that no amount of sampling
removes. Telling the two apart is the whole craft of error characterisation; removing the floor is
the job of error mitigation.

**Where the course shows it.** `05/02` (the Born rule on hardware), `05/06` (the capstone and the
frontier).

## "Bures distance is the same as one minus the fidelity."

**The intuition.** Both go to zero as two states coincide, so they must be the same quantity.

**What is really going on.** The Bures distance is `d_B(ρ, σ) = √(2(1 − √F))`, the quantum
Fisher–Rao distance built from the Uhlmann fidelity `F` — not `1 − F`, and not its square. Keep track
of whether a function returns the distance or its square.

**Where the course shows it.** `02/12` (the Bures distance), `05/05` (Bures on hardware).

---

*Spotted another trap worth adding? Contributions are welcome — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).*
