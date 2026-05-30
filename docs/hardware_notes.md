# Hardware Notes — Running on Real IBM Quantum

The course is **simulator-first**: every notebook runs offline with no account and no network. This
document is for when you want to run the module-05 notebooks on a **real IBM QPU** — how to set up
credentials, what the one switch does, the idioms that matter, and the traps to avoid.

> For the science (what the hardware results do and do not prove), see
> [`scientific_claims_guardrails.md`](scientific_claims_guardrails.md).

---

## 1. Credentials (never commit a token)

You need an API token from your IBM Quantum Platform dashboard. There are two ways to make it
available; the run-cells in the notebooks read it the same way either way.

**A — `save_account` (recommended).** One time, in a Python shell:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",   # current platform; the legacy "ibm_quantum" channel was retired mid-2025
    token="<YOUR_TOKEN>",
    overwrite=True,
)
```

This writes the token to `~/.qiskit/qiskit-ibm.json` — **outside the repository**. The run-cell is
then literally `QiskitRuntimeService()` with no token in sight.

**B — a git-ignored `.env`.** Put the token in a `.env` at the repository root (already in
`.gitignore`):

```
IBM_QUANTUM_TOKEN=<YOUR_TOKEN>
```

A `.env` file is **not** read automatically by Python — export it into your shell (or load it with
`python-dotenv`) so it lands in the environment. `select_backend` then picks it up via
`IBM_QUANTUM_TOKEN` (optionally `IBM_QUANTUM_CHANNEL`).

Either way the rule holds: **never hardcode or commit a token.** `save_account` keeps the secret out
of the repository tree entirely and is the lower-risk choice.

## 2. The one switch

Every module-05 notebook selects its backend in a single cell:

```python
from qot_course.hardware.runtime import select_backend

USE_HARDWARE = False  # flip to True once your credentials are set
backend, label, is_real = select_backend(use_hardware=USE_HARDWARE)
print(f"Running on: {label}  (real hardware = {is_real})")
```

- `USE_HARDWARE = False` (default) returns an offline `AerSimulator` carrying the `FakeManilaV2`
  noise model — runs with no network, no credentials, in CI.
- `USE_HARDWARE = True` returns the least-busy real QPU; if credentials are missing it falls back to
  a simulator and reports `is_real = False`, so the cell stays honest about where it actually ran.

The same primitive code runs in both cases — this is **local testing mode** (passing a fake or Aer
backend straight to a runtime primitive runs it locally).

## 3. The primitive path (not `backend.run`)

On today's IBM Quantum Platform, a real QPU is driven only through the runtime **primitives** —
`SamplerV2` (samples bitstrings → counts) and `EstimatorV2` (expectation values), optionally inside a
`Session` or batch. The local `backend.run(...)` idiom works on simulators but is **not** the path to
a QPU. Transpilation to an **ISA circuit** (the device's native gates and qubit connectivity) is
mandatory:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa = pm.run(qc)                                   # qc built with QuantumCircuit(n, n) + qc.measure(...)
counts = SamplerV2(mode=backend).run([isa], shots=4096).result()[0].data.c.get_counts()
```

The classical register is named `c` when you build the circuit with `QuantumCircuit(n, n)` — hence
`.data.c.get_counts()`. (A circuit built with `qc.measure_all()` names it `meas`.)

## 4. Traps worth knowing

- **The transpiler cancels identity padding.** Inserting `x` then `x` (an identity) to add depth and
  dial decoherence will be *optimised away* at `optimization_level ≥ 1` — the gates vanish and any
  depth sweep goes flat. Put a `barrier` around each gate (`qc.barrier(); qc.x(0); qc.barrier();
  qc.x(0)`) so the transpiler keeps them. (`05/05` does this.)
- **`seed_simulator` is simulator-only.** `backend.set_options(seed_simulator=...)` makes an offline
  run reproducible, but a real QPU has no such option and will raise. Guard it:
  `if not is_real: backend.set_options(seed_simulator=42)`.
- **Channel churn.** The legacy `ibm_quantum` channel was retired mid-2025; the current one is
  `ibm_quantum_platform`. Keeping the run-cell as a bare `QiskitRuntimeService()` (reading the saved
  account) is robust to this.

## 5. What to expect

- **A modern device is cleaner than the offline model.** The default `FakeManilaV2` is an old 5-qubit
  noise model. A current processor does noticeably better. For reference, a one-shot run on
  `ibm_marrakesh` (156-qubit Heron, 2026-05-30) gave, for a Bell pair:

  | quantity | FakeManila (offline) | ibm_marrakesh | ideal |
  |---|---|---|---|
  | fidelity | 0.90 | 0.96 | 1.00 |
  | `I(A:B)` | 1.38 bits | 1.74 bits | 2.00 bits |

  A consequence: a clean device can be *too good* for a noise demonstration — on `ibm_marrakesh` the
  `05/05` depth sweep sat within shot-noise (the gates barely move the state), so the offline
  FakeManila model is what makes the "more depth → more decoherence" principle visible.

- **Quota and queue.** Real runs consume your account's QPU allocation (the Open plan is limited) and
  pass through a queue. A small 1–2 qubit tomography job is seconds of QPU time, but be deliberate
  about how many you submit.

## 6. Reproducing the committed snapshots

Each `05/*` notebook records a real `ibm_marrakesh` run in an "A real run on IBM hardware" cell. To
reproduce: set `USE_HARDWARE = True` (with credentials configured) and re-run the notebook. Your
numbers will differ run to run (sampling) and device to device (the least-busy QPU varies), but the
story holds: below the ideal, above the classical bound where one exists.
