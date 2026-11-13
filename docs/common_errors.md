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

**What is really going on.** A coupling measure reads only what your *embedding* hands it, the
embedding being the recipe that writes the two signals into a quantum state. PLV is the modulus of
the first circular moment of the phase difference, `|⟨e^{iΔθ}⟩|`. Two ensembles can share the same
first moment, hence an *identical* PLV by construction, and differ in their second moment
`⟨e^{2iΔθ}⟩`. The mutual information of a quantum embedding that carries both moments, the
multi-frequency qutrit of `04/14`, then differs between them, while PLV and the naive phase-qubit
embedding, which carries the first moment only, give the same value for both. PLV is not the only
classical statistic, though: the second-harmonic PLV, `|⟨e^{2iΔθ}⟩|`, reads the same difference, and
from fewer samples than the qutrit embedding with equal weights needs. So the reason to bother is not
sensitivity. It is that PLV and the second-harmonic PLV become two readings of one state, measured by
one distance from independence, with the embedding as the dial that sets what is read.

**Where the course shows it.** `04/13`–`04/17` (the redundancy reveal, the richer embedding, the
discriminating experiment, the test on coupled oscillators, and the weighted embedding).

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

**Where the course shows it.** `02/11` (quantum mutual information), `02/12` (negative conditional
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
statistic, the second-harmonic PLV, also reads, and from fewer samples. On the relative phase of the
two moments, the quantum measures computed in the course read less than a classical statistic does.
The value of quantum optimal transport here is the **unified geometric framework**, one state and one
distance from independence with the embedding as the single dial that sets what the coupling measure
reads. It is not an exclusive detecting power.

**Where the course shows it.** `04/15`–`04/17` (the discriminating experiment, the accounting of the
capstone, and the weighted embedding).

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

**Where the course shows it.** `02/13` (the Bures distance), `05/05` (Bures on hardware).

---

*Spotted another trap worth adding? Contributions are welcome — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).*
