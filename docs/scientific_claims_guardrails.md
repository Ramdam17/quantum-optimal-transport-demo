# Scientific Claims — Guardrails

This course makes a **precise, bounded** claim about quantum optimal transport. It is easy, in
summary or in conversation, to round that claim up into something stronger and false. This document
states what the course **does** and **does not** establish, so the science stays honest — and so an
AI tutor working alongside a learner does not overclaim.

Each entry gives a claim you **may** make, the nearby claim you **may not**, and **why**.

> Companion docs: [`common_errors.md`](common_errors.md) (learner-facing intuition traps) and
> [`hardware_notes.md`](hardware_notes.md) (running on real devices).

---

## 1. Quantum optimal transport vs PLV

**May:** *In simulation*, a richer multi-frequency quantum embedding's mutual information separates
two oscillator ensembles that have **identical PLV by construction** but a different second circular
moment — where PLV, and the naive phase-qubit embedding, cannot (`04/15`; the separation is
significant against an a2-matched null).

**May not:**
- "Quantum optimal transport beats PLV." — Too broad. The result is about *matched-PLV ensembles
  differing in a specific higher moment*, under a *specific embedding*, in *simulation*.
- "Only quantum optimal transport can see this difference." — False. The difference is a second-order
  phase statistic; a classical higher-order statistic could detect it too.

**Why:** The discriminating power comes from the **embedding** (what structure of the signal is
written into the state), not from quantum mechanics being uniquely able to see it. The value of QOT
here is a **unified geometric framework** — one geometry, one pair of measures, with the embedding as
the single dial. State it that way. (`04/13`–`04/16`.)

## 2. Quantum mutual information and entanglement

**May:** For the **pure** Bell pair, `I(A:B) = 2` bits — double the classical maximum — and this is a
legitimate signature of maximal entanglement (for a pure bipartite state, QMI = 2 × entanglement
entropy).

**May not:** "Quantum mutual information is an entanglement measure." — QMI measures **total**
correlation (classical + quantum). For mixed states it includes classical correlations and is not, on
its own, an entanglement measure.

**Why:** The pure-state identity is a special case, not the general meaning. (`02/10`, `02/11`,
`05/04`.) Refs: Nielsen & Chuang ch. 11; Wilde, *Quantum Information Theory*, ch. 11.

## 3. The capstone effect on real hardware

**May:** The capstone effect is **demonstrated in simulation** (exact density matrices). On a real
QPU it is **not resolvable today**, and the course quantifies by how much it falls short.

**May not:** "We demonstrated quantum-optimal-transport coupling detection on real quantum hardware."
— No. The simulated effect is ≈ 0.004 nats; a device's statistical band on a single absolute QMI is
already larger (≈ 0.012 nats on `ibm_marrakesh`), so one run cannot resolve it — and for the
*discrimination* the binding limit is the band of the difference, ≈ √2·σ_stat (see §5), not this
absolute band.

**Why:** Keep "the effect exists" (a simulation result) and "current hardware can measure it" (false,
today) as separate statements. Resolving the effect would need far more shots *and* error mitigation
to remove the systematic bias. This is the same structural wall described by De Palma, Marvian, Rouzé
& Stilck França, *Limitations of variational quantum algorithms: a quantum optimal transport
approach* (arXiv:2204.03455 / PRX Quantum 4, 010309, 2023). (`04/17`, `05/06`.)

## 4. Reading a "device below ideal" gap

**May:** A reconstructed device quantity below its ideal (fidelity < 1, purity < 1, `I(A:B)` < 2)
reflects **decoherence together with finite-shot and projection bias**.

**May not:** "The whole gap is decoherence." — Even a noiseless device at finite shots reconstructs
purity/fidelity below 1, because the Bloch/Pauli estimates are noisy and the projection to a physical
state is a biased estimator.

**Why:** A single run cannot separate the two contributions; more shots shrink the statistical part
and isolate the genuine physical (decoherence) gap. (`05/03`, `05/04`.)

## 5. Systematic bias cancels in a difference

**May:** For an **absolute** quantity (e.g. one Bell-pair `I(A:B)`), the device's systematic bias is
a real obstacle. For the capstone **discrimination** — a *difference* of two matched-PLV QMIs — a bias
that affects both ensembles alike **largely cancels**, so the binding limit is the statistical band of
the difference (≈ √2 · σ_stat).

**May not:** "The ~0.4-nat systematic bias is what blocks the discrimination." — It mostly cancels in
the difference; do not present it as the binding constraint for telling the ensembles apart.

**Why:** The conclusion (the effect is unresolvable today) holds either way, but the *reason* must be
correct: the statistical band, not the absolute bias, is the wall for the discrimination. (`05/06`.)

## 6. Two different sigmas — do not conflate them

**May:** Cite the **simulation** significance ("≈ 7.6σ against an a2-matched null") as a statement
about the simulated experiment, and the **device** statistical band (σ_stat) as a statement about
hardware resolution.

**May not:** Treat "7.6σ" as if it were a hardware result, or compare it directly to the device error
bar. They are different quantities living in different sentences.

**Why:** One is the separation of a simulated effect from a control distribution; the other is the
shot-noise on a single hardware estimate. Conflating them manufactures a false "significant on
hardware" claim. (`05/06`.)

---

## Quick reference

| Topic | Safe to say | Do not say |
|---|---|---|
| QOT vs PLV | richer embedding separates matched-PLV ensembles *in simulation* | "QOT beats PLV"; "only QOT can" |
| QMI | total correlation; = entanglement only for pure states | "QMI is an entanglement measure" |
| Hardware | effect demonstrated in simulation; unresolvable on current QPUs | "demonstrated on real hardware" |
| device < ideal | decoherence **and** finite-shot/projection bias | "purely decoherence" |
| the discrimination | statistical band is the wall (bias cancels in the difference) | "the systematic bias blocks it" |
| significance | 7.6σ is a *simulation* control comparison | "7.6σ on hardware" |

**One-line summary:** the course *demonstrates a real, embedding-driven effect in simulation*, frames
quantum optimal transport as a *unifying geometric language* (not a unique detector), and is *honest
that current hardware cannot yet resolve the effect* — quantified, not hand-waved.
