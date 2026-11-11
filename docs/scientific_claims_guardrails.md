# Scientific Claims — Guardrails

This course makes a **precise, bounded** claim about quantum optimal transport. It is easy, in
summary or in conversation, to round that claim up into something stronger and false. This document
states what the course **does** and **does not** establish, so the science stays honest — and so an
AI tutor working alongside a learner does not overclaim.

Each entry gives a claim you **may** make, the nearby claim you **may not**, and **why**.

> Companion docs: [`common_errors.md`](common_errors.md) (learner-facing intuition traps) and
> [`hardware_notes.md`](hardware_notes.md) (running on real devices).

## Terms used here

- **PLV** (phase-locking value) is the modulus of the first circular moment of the phase difference
  Δ between two signals, `|⟨e^{iΔ}⟩|`. It measures how steadily the two signals keep the same phase
  difference.
- The **second circular moment** is `⟨e^{2iΔ}⟩`, and its modulus is the **second-harmonic PLV**.
  "The two harmonics" are these two moments. Their **relative phase** is the angle of the second
  moment, counted from twice the angle of the first.
- An **embedding** is the recipe that writes the two phases into a quantum state. The naive
  phase-qubit embedding carries the first moment only. The multi-frequency qutrit embedding carries
  the first two, with equal weights or with chosen ones; "the weighted embedding" is the one of
  `04/17`, with weights (1, 0.2, 2).
- An **ensemble** is a set of sampled pairs of phases. The two **matched-PLV ensembles** of `04/15`
  share their first circular moment, hence their expected PLV, and differ in the second. The **gap**
  is the difference in quantum mutual information between them.
- A **nat** is the unit of information counted with natural logarithms; one nat is about 1.44 bits.
  Section 2 speaks in bits, and sections 3 and 5 in nats.
- The **area under the curve** (AUC) is the probability that an ensemble with the structure has a
  larger value of a statistic than an ensemble without it; it ignores the scale of the statistic. The
  **detection rate** is the fraction of observed differences that fall outside the central 95 % of
  the differences obtained when there is nothing to find. Both are defined in `04/15`.
- **σ_stat** is the shot-noise error bar of one hardware estimate of a mutual information. A gap is
  a difference of two estimates, so its **band** is √2·σ_stat.

---

## 1. Quantum optimal transport vs PLV

**May:** On synthetic signals, a quantum coupling measure reads coupling structure that PLV cannot,
**when its embedding carries the second harmonic**. The mutual information of the multi-frequency
qutrit embedding differs between two ensembles that have **identical PLV by construction** and a
different second circular moment (`04/15`), and it detects a coupling carried by the second harmonic
alone, where the long-run PLV is zero (`04/16`). On the naive phase-qubit embedding, the quantum
measures are functions of PLV (`04/13`).

**May not:**
- "Quantum optimal transport beats PLV." — PLV is one classical statistic among several. The
  second-harmonic PLV reads the same structure, from fewer samples (`04/15`) and shorter records
  (`04/16`) than the equal-weight embedding needs. The weighted embedding of `04/17` comes level
  with it and does not pass it.
- "Only quantum optimal transport can see this difference." — False: the second-harmonic PLV reads
  the same difference.
- "The quantum measure is the more sensitive one." — On the AUC, the score that ignores scale, the
  course found no case where a quantum measure ranks the records better than the matching classical
  statistic. Where the detection rate of the weighted measure comes out a little higher, `04/17`
  traces it to the scale of the statistic and not to a better ranking. On the hand-built ensembles of
  `04/15` the equal-weight quantum measure needed at least a hundred times more samples than the
  second-harmonic PLV.
- "The quantum measures read everything a classical statistic of the two harmonics reads." — They do
  not. The quantum mutual information and the Bures distance of the qutrit embedding read the cosine
  of the relative phase of the two harmonics, so they cannot tell a relative phase of +β from −β. A
  classical sine moment can (`04/16`). This holds for the measures computed in the course, which are
  functions of the state; `04/16` notes that the expectation value of a well-chosen observable could
  in principle read the sign.

**Why:** What a coupling measure reads is set by the **embedding**, that is, by which structure of the
signal is written into the state, and each thing it reads has a classical statistic beside it. The
value of quantum optimal transport here is a **unified geometric framework**: one state and one
distance from independence, so that choosing which statistic to compute becomes choosing which
embedding to use. That is a unification, and on the systems of the capstone it is not a gain in
sensitivity. The evidence is one hand-built pair of ensembles and one model of coupled oscillators,
all synthetic; the course did not test real recordings. (`04/13`–`04/17`.)

## 2. Quantum mutual information and entanglement

**May:** For the **pure** Bell pair, `I(A:B) = 2` bits — double the classical maximum — and this is a
legitimate signature of maximal entanglement (for a pure bipartite state, QMI = 2 × entanglement
entropy).

**May not:** "Quantum mutual information is an entanglement measure." — QMI measures **total**
correlation (classical + quantum). For mixed states it includes classical correlations and is not, on
its own, an entanglement measure.

**Why:** The pure-state identity is a special case, not the general meaning. (`02/10`, `02/11`,
`05/04`.) Refs: Nielsen & Chuang ch. 11; Wilde, *Quantum Information Theory*, ch. 11.

## 3. The capstone's gaps on real hardware

**May:** The two gaps are **computed on a classical computer**, with no read-out noise, from density
matrices built on 150,000 phase samples per ensemble: 0.0044 nat with equal weights (`04/15`; 0.0049
nat in the long run, `04/16`) and 0.0145 nat with the weighted embedding (`04/17`). The course then
measures how finely a device reads a mutual information, on a two-qubit Bell pair, and sets the two
gaps against that resolution (`05/06`). The first is about a quarter of the band of a difference and
the second about 0.85 of it, so one run would resolve neither.

**May not:**
- "We demonstrated quantum-optimal-transport coupling detection on real quantum hardware." — No. No
  state of the capstone was prepared on a device. `05/06` measures the error bar of one estimate on
  a Bell pair, σ_stat ≈ 0.012 nat on `ibm_marrakesh`, and the band of a difference is then at least
  √2·σ_stat ≈ 0.017 nat, since that figure counts shot noise only. The equal-weight gap is about a
  quarter of that band and the weighted gap about 0.85 of it, before the device shrinks them (§5).
  The weighted embedding brings the gap closer to the resolution of the device and does not reach it.
- "A quantum processor does better than a classical computer on these signals." — Nothing in the
  course shows it. The quantities are computed classically, a classical statistic reads the same
  structure, and the device adds noise to the reading.

**Why:** Keep "the gap exists" (a classical computation) and "hardware can measure it" (not shown, and
out of reach of one run on the device studied in `05/06`) as separate statements. Bringing the band
down to a third of a gap would take about 5.8 × 10⁵ shots per measurement setting and per ensemble
for the equal-weight gap, and about 5.3 × 10⁴ for the weighted gap, on the offline device model of
`05/06`; it would take more once the device has shrunk the gap (§5). These counts come from a
two-qubit Bell-pair probe, while the capstone's states are pairs of qutrits, which need four qubits
and a larger tomography. They are orders of magnitude, and neither an upper nor a lower bound.
(`04/17`, `05/06`.)

## 4. Reading a "device below ideal" gap

**May:** A reconstructed device quantity below its ideal (fidelity < 1, purity < 1, `I(A:B)` < 2)
reflects **decoherence together with finite-shot and projection bias**.

**May not:** "The whole gap is decoherence." — Even a noiseless device at finite shots reconstructs
purity/fidelity below 1, because the Bloch/Pauli estimates are noisy and the projection to a physical
state is a biased estimator.

**Why:** A single run cannot separate the two contributions; more shots shrink the statistical part
and isolate the genuine physical (decoherence) gap. (`05/03`, `05/04`.)

## 5. Systematic bias in a difference

**May:** For an **absolute** quantity (e.g. one Bell-pair `I(A:B)`), the device's systematic bias is
a real obstacle. For the capstone **discrimination** — a *difference* of two mutual informations —
the bias **rescales the gap**. On the offline device model and a family of two-qubit probe states,
the device returns between 0.60 and 0.81 nat for each nat of ideal mutual information, so a gap of
0.010 nat would arrive as 0.006 to 0.008 nat. The lowest slope belongs to the weakest correlations
that were measured. Closer to zero, where the capstone's states sit, the slope was not measured, and
the loss there may be larger. The binding limit for telling the two ensembles apart is the
statistical band of the difference.

**May not:**
- "The systematic bias is what blocks the discrimination." — On the probes measured, the bias shrinks
  a gap by a fifth to two fifths and does not erase it, so it is not what limits the discrimination.
- "The bias cancels in the difference." — It would if it were the same for every state. It grows
  with the mutual information of the prepared state, from about zero for a product state to 0.39 nat
  for the Bell pair on the offline model (0.18 nat on the real run), so it cancels only in part.

**Why:** The conclusion (one run would not resolve the gaps) holds either way, but the *reason* must
be correct: the statistical band is the wall for the discrimination, and the bias shrinks the gap
without hiding it. (`05/06`.)

## 6. Two different yardsticks: signal samples and device shots

**May:** Cite the **detection rates and areas under the curve** of `04/15` to `04/17` as statements
about how many samples of the *signals* a statistic needs to tell two ensembles apart, and the
**device** error bar (σ_stat) as a statement about hardware resolution.

**May not:** Set a detection rate beside the shot noise of the device as if the two were on one
scale, or present a detection rate computed classically on sampled signals as a hardware result. Do
not quote a significance in σ for the capstone either, such as the "7.6σ" of an earlier version of
the course: every statistic is scored with a detection rate and an area under the curve (`04/15`).

**Why:** The first yardstick comes from the finite number of signal samples, and has nothing to do
with a device. The second is the shot noise on a single hardware estimate of a state that is known
exactly. A demonstration on hardware would pay both. Conflating them manufactures a false
"significant on hardware" claim. (`05/06`.)

---

## Quick reference

| Topic | Safe to say | Do not say |
|---|---|---|
| QOT vs PLV | with an embedding that carries the second harmonic, reads past PLV on synthetic signals, no further than a classical statistic of the same harmonics | "QOT beats PLV"; "only QOT can"; "QOT is more sensitive" |
| relative phase | the QMI and Bures distance of the qutrit embedding read its cosine, not its sign | "the quantum measures read everything the classical ones do"; "no quantum measure could read the sign" |
| QMI | total correlation; = entanglement only for pure states | "QMI is an entanglement measure" |
| Hardware | gaps computed classically; no capstone state was run on a device; judged on a Bell-pair probe, one run would resolve neither gap | "demonstrated on real hardware" |
| quantum processor | it adds noise to the reading of quantities computed classically | "a quantum processor does better on these signals" |
| device < ideal | decoherence **and** finite-shot/projection bias | "purely decoherence" |
| the discrimination | statistical band is the wall; the bias rescales the gap | "the systematic bias blocks it"; "the bias cancels" |
| yardsticks | detection rates describe signal samples; σ_stat describes the device | "7.6σ"; a detection rate quoted as a hardware result |

**One-line summary:** the course *measures, on synthetic signals, that a quantum coupling measure reads
past PLV, no further than a classical statistic of the same harmonics and less far on their relative
phase*, and it frames quantum optimal transport as a *unifying geometric language*, not a unique or
more sensitive detector. It is also *honest that hardware would not resolve gaps of the size of the
capstone's in one run*, a statement quantified on a Bell-pair probe and not hand-waved.
