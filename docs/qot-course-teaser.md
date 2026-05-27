# From Moving Sand to Moving Matrices

### A hands-on journey into Optimal Transport and its quantum cousin

*A lab course — 16 weekly sessions of 2 hours, over about three months. No quantum background required. Bring Python curiosity.*

---

## Why this course?

Picture a pile of sand, and a hole you want to fill with it. What is the least-effort way to move the sand? That deceptively simple question is **optimal transport (OT)** — and over the last two decades it has quietly become one of the most useful ideas in data science. It gives us a principled way to measure *how far apart two distributions are* (the **Wasserstein distance**), and it shows up everywhere we compare things: morphing images, aligning word embeddings between languages, matching populations in genetics, comparing brain-activity patterns.

Now the twist that gives this course its name. What if the objects we compare are not piles of sand — not probability distributions — but **matrices**? Covariance matrices, connectivity matrices, diffusion tensors, kernels: the positive-definite objects we measure in the lab all the time. To transport *those* correctly, you need a generalization that mathematicians borrowed, of all places, from **quantum mechanics**: **quantum optimal transport (QOT)**.

Here is the honest punchline, and the spirit of the whole course: **your data does not need to be quantum for this to be useful.** The mathematics of quantum states — density matrices, von Neumann entropy, quantum channels — turns out to be exactly the right *language* for matrix-valued data. We use the quantum formalism because it is the correct geometry, not because we are claiming any "quantum magic."

Over sixteen weeks we will climb, slowly and by hand, from that first pile of sand to the research frontier. By the end you will have **built, in your own code, a genuine quantum-Wasserstein distance** — and, just as importantly, you will have the judgment to tell a real result from quantum hype.

---

## What makes it different

- **Hands-on, one concept at a time.** Each session is a short piece of intuition, then live code, then exercises *you* complete, then a one-page summary you keep. No firehose.
- **From zero on the quantum side.** We assume no quantum mechanics, no measure theory, no information theory beyond the basics. We build the vocabulary together.
- **Relentlessly honest.** We flag, out loud, what is solid versus what is fashionable. (Example: a "quantum mutual information" is rock-solid; a "quantum transfer entropy" is an open research question — and we will say so.)
- **You keep everything.** Reusable, documented code; exhaustive notebooks; typeset summaries; and two living documents we grow every week — a **classical-to-quantum dictionary** and a **glossary**.

---

## The journey — four movements

**Movement 1 · Quantum foundations (≈3 weeks).** The grammar: qubits, *density matrices*, entanglement, and quantum channels. We will even run a few cells on a **real IBM quantum computer** — and watch a state we prepared "perfectly" come back slightly imperfect, because the hardware is noisy. That is your first, visceral lesson in *why density matrices exist*.

**Movement 2 · Information theory & geometry (≈3 weeks).** The spine of the whole course. Entropy, the Kullback–Leibler divergence, **mutual information**, **transfer entropy** — and their quantum counterparts. One quantity (relative entropy) will keep reappearing for the rest of the course, so we give it the time it deserves. We will also meet the two great geometries of probability — the *information* geometry of Fisher and the *transport* geometry of Wasserstein — and the beautiful fact that they shake hands at a single, famous algorithm.

**Movement 3 · Classical optimal transport (≈4 weeks).** Monge's original problem, Kantorovich's relaxation, the Wasserstein distances, duality, and the workhorse **Sinkhorn** algorithm. We end on the one case we can solve with a clean formula — transport between Gaussians — which turns out to be the secret door into the quantum world.

**Movement 4 · Quantum optimal transport (≈5 weeks).** Why matrices that *do not commute* break our classical intuition; how to compute QOT honestly as a **convex optimization** problem; the **quantum Sinkhorn**; and a **capstone open-research project**: measuring the coupling between two interacting systems — think of two brains in a hyperscanning experiment — with the information-theoretic and transport tools we have built. It is a real, unsolved question, and by week sixteen you will have the tools to attack it.

---

## A taste of the punchline

Two quick pictures of *why QOT exists*:

- **The coin that is not a coin.** Two quantum systems can give the *exact same measurement statistics* — both look like a fair coin, 50/50 — yet be genuinely different objects, because one carries a hidden "coherence" the measurement throws away. A classical distance says "identical." QOT says "different." That gap is the entire subject.
- **Same energy, different wiring.** Two systems can have the *same per-channel power* but completely different *connectivity* between channels. Tools that look only at the power see twins; tools that look at the whole matrix — the quantum ones — see two different things.

If those two examples make you curious, this course is for you.

---

## What you will walk away with

- A working intuition for **optimal transport** and the **Wasserstein distance**, and the ability to compute them in Python.
- Fluency in the **language of quantum information** — density matrices, entropy, fidelity, channels — without needing a physics degree.
- A real grasp of **information theory** (entropy, KL, mutual information, transfer entropy) *and its quantum extensions*.
- Hands-on experience with **convex optimization** (semidefinite programming) and with **real quantum hardware** via Qiskit.
- A reusable, tested **codebase** and a stack of one-page summaries you can teach from.
- A genuine **research seed** you could turn into a methods project for the lab.

---

## Who it is for

Anyone in the lab — and beyond — who is comfortable with **Python and NumPy** and curious about the ideas. You do **not** need quantum mechanics, measure theory, or prior information theory. If you can write a for-loop and read a plot, you are ready. We set up the environment together in session one.

---

## The honest promise — what this is *not*

This is **not** a course claiming that brains are quantum computers, or that a quantum chip will speed up your analysis tomorrow. It is a course about a powerful piece of mathematics, taught carefully, with its limits stated plainly. One of the skills you will leave with is the ability to **read a "quantum X" paper and tell the substance from the marketing**. In a field full of hype, that alone is worth the sixteen weeks.

---

## Practical details *(to personalize)*

- **When:** [start date] — weekly, [day/time], ~2 hours each, for ~3 months.
- **Where:** [room / video link].
- **What to bring:** a laptop; we install everything together in session one (Python via `uv`, plus a few open-source libraries). A free IBM Quantum account if you want to run the hardware cells yourself (optional).
- **Cost:** free, open to the lab and collaborators.
- **Interested?** [how to sign up / contact].

---

*One line for the busy reader:* **a gentle, hands-on, sixteen-week climb from "moving piles of sand" to computing a real quantum-Wasserstein distance — building the code, the intuition, and the healthy skepticism along the way.**
