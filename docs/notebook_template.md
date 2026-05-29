# Notebook charter template (QOT course)

Every notebook is ONE concept (a "brick") or a "synthesis". Plain `.ipynb`, output-free in git.
Voice: warm, empowering, celebratory — AND rigorous. No decorative emojis. No "obviously /
simply / trivially / just". Celebrate what the learner just achieved; frame difficulty as growth.

## Cell order
1. **Header** (markdown): `# NN — Title`; one-line purpose; **Prerequisites** (notebook ids);
   **What you'll be able to do** (3-5 action-verb objectives). A warm one-line welcome is encouraged.
2. **Imports** (code): stdlib / third-party / `qot_course` local. `np.random.seed(...)`;
   `from qot_course import viz; viz.use_course_style()`. **Never** hardcode hex — use `qot_course.colors`.
3. **Body sections** (one concept), each: *intuition* (markdown) → *implementation* (code) →
   **"Read the figure / output"** (markdown — always explain what we see, kindly and concretely).
4. **Your turn** (markdown): 2-3 small exercises, tiered easy→harder (no emoji labels — words).
5. **Summary** (markdown): "What you built" bullets that *celebrate the accomplishment* + dictionary row.
6. **References** (markdown): cited papers; `Previous:` / `Next:` links within the module.

## Synthesis notebooks
Assume their bricks; integrate, deliver the punchline + dictionary row + (where relevant) the
hardware/application demo. Lighter than a from-scratch notebook.

## Figures
Use `qot_course.viz` helpers and `qot_course.colors`. Fixed dims/DPI from the charter. Every
figure is followed by a "Read the figure" paragraph.
