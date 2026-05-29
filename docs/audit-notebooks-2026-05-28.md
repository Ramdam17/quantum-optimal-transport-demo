# Audit des notebooks — Quantum Optimal Transport course

**Auteur de l'audit :** Claude (revue demandée par Rémy)
**Date :** 2026-05-28
**Portée :** les 16 notebooks `notebooks/s01…s16`, les modules `src/qot_course/`, les tests, l'état git.
**Méthode :** lecture intégrale des 16 notebooks + lecture des modules sources QOT + exécution
de la suite de tests (`uv run pytest` → **161 passed**) + exécution de `course_greatest_hits()`
+ une expérience de contrôle sur le capstone (détaillée en Annexe).

> **But du document.** Avant de reprendre les notebooks un par un, poser à plat : (1) ce qui
> est fait, (2) les étonnements, (3) les erreurs détectées, (4) des pistes d'amélioration, (5)
> les points où **ça va trop vite** et les étapes intermédiaires à introduire. C'est une base
> de discussion, pas un plan d'action figé. Aucune modification n'a été faite au code.

---

## Verdict en une page

**Le cours est, en grande majorité, de l'excellent matériel didactique.** s01→s14 sont riches,
progressifs, chaque figure est expliquée, les exercices « Your turn » sont pertinents, les
références primaires sont correctes. Le **framework calcule vraiment** : le SDP (s13) et le
quantum Sinkhorn (s14) sont du vrai code validé contre des formes closes ; les 161 tests
passent ; toutes les identités-phares de `course_greatest_hits()` tiennent.

**Donc le « ça ne marche pas » que tu as senti n'est PAS un bug et n'est PAS généralisé.**
C'est une **conclusion scientifique préfabriquée**, localisée dans **le capstone (s15)** et
reprise par **s16**. Le problème se concentre en trois foyers :

| Foyer | Gravité | Une phrase |
|---|---|---|
| **A1. Le capstone est construit pour ne pas pouvoir gagner** | 🔴 | QMI/Bures y sont des reparamétrisations monotones de PLV (Spearman ≈ 1.000) ; « est-ce que QOT bat PLV ? — ouvert » est un non-test, pas un résultat. |
| **A2. La moitié difficile n'est pas commitée** | 🔴 (process) | s06→s16 + tout `quantum_ot/`, `transport/`, `geometry/` + 12 tests sont untracked. Aucun historique d'effort sur la partie OT/QOT. |
| **A3. Le rythme et l'architecture** | 🟠 | Sauts conceptuels trop grands (surtout M4) ; **trois** objets « quantum Wasserstein » différents jamais réconciliés. |

Plus des points mineurs : s14 « Quantum Sinkhorn » ne lance jamais d'itération Sinkhorn (A4),
quelques artefacts non relus et erreurs de cross-référence (A5).

---

## Partie A — Constats transversaux

### A1. 🔴 Le capstone (s15) : un « c'est ouvert » préfabriqué

C'est le cœur de ton étonnement, et c'est vérifié, pas supposé.

**Le montage.** `capstone.py:joint_density_matrix` construit
`ρ_AB = E_t[ |ψ_A(t)⟩⟨ψ_A(t)| ⊗ |ψ_B(t)⟩⟨ψ_B(t)| ]` avec `|ψ(t)⟩ = (|0⟩ + e^{iθ(t)}|1⟩)/√2`.
Une moyenne temporelle d'**états produits purs** est un état **séparable par construction** —
intrication nulle à *tout* K. En développant les éléments de matrice, chaque entrée vaut
`(1/4)·E[e^{i(θ_A·Δm + θ_B·Δn)}]` avec Δ ∈ {−1,0,1} : `ρ_AB` est entièrement déterminée par les
moments de phase. Dans le régime « phases qui dérivent » (marginales → I/2, cohérence-somme → 0,
ce que le notebook confirme lui-même), il ne reste qu'**un** paramètre : `|E[e^{i(θ_A−θ_B)}]|`,
qui **est** la PLV (au facteur 4 près : `PLV = 4·|ρ_AB[01,10]|`).

**Mesure de contrôle** (200 dyades, 8 seeds × 25 K — voir Annexe) :
```
Spearman ρ(PLV, QMI)   = 0.9997      une f(PLV) lisse explique R² = 0.9997 de QMI
Spearman ρ(PLV, Bures) = 0.9999      une f(PLV) lisse explique R² = 0.9984 de Bures
```
⟹ Dans ce montage, QMI et Bures-coupling **classent chaque dyade exactement comme PLV**.
Ils ne *peuvent pas* la battre. « Whether they beat PLV is open » n'est pas un constat
empirique — c'est une tautologie du plongement choisi.

**Le test décisif est nommé puis esquivé.** Le « Your turn » item #3 (« même distribution de
différence de phase, structure d'ordre supérieur différente, montrer que les mesures quantiques
les distinguent ») est *exactement* l'expérience qui donnerait sa chance à QOT — refilée en
devoir. Et `tests/test_capstone.py` ne teste que « la mesure monte avec K » (`test_coupling_
measures_increase_with_K`), propriété que PLV partage trivialement. Aucun test ne vérifie une
quelconque discrimination que PLV ne ferait pas.

**Erreur visible en prime.** Cellule `2091922a`, dans le tableau « didactique » des 4 mesures,
la formule de QMI contient une auto-correction de modèle laissée verbatim :
> `$S(\rho_{AB}) + S(\rho_B) + S(\rho_A) - $ ... wait it's $S(\rho_A) + S(\rho_B) - S(\rho_{AB})$`

C'est le seul artefact de ce type dans les 16 notebooks (balayage effectué), mais il est
révélateur de génération non relue dans ce notebook précis.

**Ce n'est pas qu'un problème de paresse : c'est un défaut de design scientifique.** La
correction n'est pas d'accepter le négatif, c'est de rendre le capstone capable de trancher
(voir Partie C, options A/B/C).

### A2. 🔴 (process) La moitié difficile n'est pas commitée

- **HEAD = `cae0d24 docs(s5): …`**. Tout ce qui suit s05 est en working-tree, untracked :
  `notebooks/s06…s16`, `src/qot_course/{geometry,transport,quantum_ot}/`,
  `infotheory/quantum.py`, `summaries/build_s06…s16`, et 12 fichiers `tests/test_*`.
- s01→s05 ont chacun un **plan** (`docs/superpowers/plans/…sN…md`) et un historique discipliné
  (plan → module → notebook → summary → ligne de dictionnaire). **Aucun plan n'existe pour
  s06→s16.**
- Conséquence directe du « sans trop d'effort visible » : il n'y a littéralement aucune trace
  d'effort incrémental sur tout le cœur OT + QOT.

→ Action endossée : **commiter s06→s16 proprement**, par session, pour avoir une baseline
diffable avant de retravailler (traitée séparément).

### A3. 🟠 Rythme & architecture — où ça va trop vite

Tu l'as senti : « ça va trop vite ». Concrètement :

1. **Trois objets « quantum Wasserstein » jamais réconciliés.** Le cours promet un objet et en
   calcule trois différents :
   - **S11** construit le *pont de Bures–Wasserstein* (terme matriciel = distance de Bures sur
     densités) comme « le » Wasserstein quantique attendu.
   - **S13** calcule en réalité un *SDP de couplage* avec coût SWAP (`1−|⟨ψ|φ⟩|²`) ou coût
     quadratique de position — un objet **différent** du pont de S11.
   - **S15** utilise encore autre chose : `Bures-coupling = d_B(ρ_AB, ρ_A⊗ρ_B)` *et* QMI.

   L'étudiant arrive en M4 en attendant le pont de S11, et on lui sert un SDP SWAP, puis une
   Bures-to-product. Les liens (égalité ? cas particuliers ?) ne sont pas explicités. **C'est
   la discontinuité conceptuelle la plus importante du cours.**

2. **M4 enchaîne une machinerie majeure par séance de 2h** : SDP (s13) → Sinkhorn quantique
   (s14) → capstone multi-objets (s15) → frontière (s16). Le capstone seul demande : phases →
   qubit → ρ_AB bipartite → 4 mesures de couplage, tout nouveau, en une séance.

3. **Étape intermédiaire manquante = la cause racine du défaut A1.** Il n'y a pas de séance
   « comment plonger un signal classique/physiologique dans un état quantique, et qu'est-ce que
   ce plongement préserve/jette ». Du coup le capstone improvise le plongement phase-pure — qui
   est précisément celui qui rend QOT redondant avec PLV. **Une séance sur les plongements
   ralentirait ET réparerait la science.**

### A4. 🟠 « Quantum Sinkhorn » (s14) ne lance jamais de Sinkhorn

s14 résout le SDP entropique via l'atome `cvxpy.von_neumann_entr` et **affirme** l'équivalence
avec l'itération opérateur (matrix-exp + partial traces) sans jamais la lancer ; l'itération
est reléguée au « Your turn » #2 (« they should give the same plan » — asserté, non démontré).
C'est **transparent** dans le texte (donc pas trompeur), mais la séance qui porte le nom d'un
algorithme ne fait pas tourner l'algorithme. Le docstring de `sinkhorn.py:17` contient en plus
une dérivation du pont d'Amari **algébriquement fausse** (`tr(ρ log ρ) − tr(ρ log K)·ε/1`, ε
mal placé) — alors que l'identité finale et son test sont corrects.

### A5. 🟡 Artefacts & cross-références à corriger

- **CLARABEL** renvoie `Solution may be inaccurate` (warnings à l'exécution de la suite et de
  `greatest_hits`) ; le code accepte `optimal_inaccurate` alors que les docstrings annoncent
  « ~1e-9 accuracy ». Écart annoncé/livré (les valeurs restent justes en pratique).
- **s10** (cellule `c29f5ca8`) : « We will see in S15 that this duality survives the lift to
  density matrices » → le dual de Lipschitz / qubit-$W_1$ est en **S16**, pas S15.
- **Forward-ref pendante** : s08 (« comes back in S15 ») et s12 (dict. « Birkhoff … S15
  mention ») renvoient à Birkhoff/assignment « en S15 », mais le capstone n'en parle pas.
- **s14** (cellule `f0904ea8`) : un one-liner `np.linalg.eigvalsh(np.linalg.eigh(K)[1] @ …)`
  illisible pour vérifier `log(K) = −C/ε` — à simplifier (didactique).

---

## Partie B — Audit notebook par notebook

Légende : ✅ solide · ⚠️ bon mais à retoucher · 🔴 problème de fond.

### s01 — Roadmap ✅
- **Fait :** OT 1-D avec POT, plan naïf vs optimal, W₂ croît avec le déplacement, dictionnaire
  amorcé, « Your turn », références. Modèle du genre.
- **Étonnements/erreurs :** RAS. (« W₂ ≈ d » : le « ≈ » est honnête, dû à la discrétisation.)
- **Pistes :** —

### s02 — Qubits & states ✅
- **Fait :** amplitudes, Born, Bloch, shot noise (20/200/2000), circuit Qiskit + Aer, idiome
  matériel honnête. Excellent.
- **Erreurs :** RAS.
- **Pistes :** lien explicite shot-noise → estimation finie (rappelé en s05/s15).

### s03 — Density matrices ✅
- **Fait :** ρ, pur vs mixte, le punchline |+⟩ vs I/2 (la graine du cours), boule de Bloch,
  fidélité/trace distance, tomographie d'un |+⟩ bruité. Excellent.
- **Erreurs :** RAS.

### s04 — Composite & channels ✅
- **Fait :** tenseur, partial trace = marginale, intrication (Bell pur ↔ parts I/2), canaux
  CPTP/dépolarisant, `is_cptp`. Excellent.
- **Erreurs :** RAS.

### s05 — Classical information theory ✅
- **Fait :** entropie, KL asymétrique, MI (indep→corrélé), transfer entropy (copie laggée),
  notes d'honnêteté (biais d'estimation, PID non unique). Excellent.
- **Erreurs :** RAS.

### s06 — Information geometry ✅
- **Fait :** simplexe, métrique de Fisher (diverge aux coins), géodésiques Fisher–Rao
  (sphère des √p), punchline deux-géométries (mixture vs Wasserstein). Excellent.
- **Erreurs :** RAS (`bernoulli_fisher(0.5)=4`, `d_FR([1,0,0],[0,1,0])=π` vérifiés).

### s07 — Quantum information theory ✅
- **Fait :** Umegaki, QMI=2 bits (Bell), entropie conditionnelle négative, sweep de Werner,
  Bures ; caveat bound-entanglement, seuil PPT 2/3 en exercice. Niveau expert.
- **Erreurs :** RAS (`d_B = 2 sin(θ/4)` vérifié).

### s08 — Monge → Kantorovich ✅
- **Fait :** Monge vs Kantorovich, obstruction de splitting (2→3), LP, polytope de transport,
  Birkhoff–von Neumann, problème d'assignation, décomposition de Birkhoff, exos (sommets
  dégénérés, Marcus–Newman). Excellent.
- **Erreurs :** la forward-ref « comes back in S15 » est pendante (cf. A5).

### s09 — Wasserstein ✅
- **Fait :** axiomes métriques, forme close quantile 1-D (= LP vérifié), W₁ = aire entre CDF
  (Vallender), géodésique de McCann, démo W₂-linéaire vs KL-diverge. Excellent.
- **Erreurs :** RAS.

### s10 — Duality & Sinkhorn ✅
- **Fait :** dualité de Kantorovich (gap nul vérifié), régularisation entropique, Sinkhorn 5
  lignes, trade-off ε, **vérification numérique du pont d'Amari** par perturbations préservant
  les marges. Excellent.
- **Erreurs :** cross-ref « in S15 » → devrait être S16 (cf. A5).

### s11 — Gaussians & dynamics ✅
- **Fait :** Bures–Wasserstein closed form (1-D et multi-D vs LP), **terme matriciel = distance
  de Bures** (vérifié contre s07), carte de McCann affine / géodésique gaussienne,
  Benamou–Brenier / Otto. Excellent — et c'est le pont vers M4.
- **Erreurs :** RAS. **Étonnement d'architecture :** ce pont est présenté comme « le »
  Wasserstein quantique, mais s13 calcule un objet différent (cf. A3).

### s12 — Why QOT ✅
- **Fait :** problème du collapse diagonal, |+⟩ vs I/2 *et* non-commutant |+⟩ vs |+i⟩, principe
  de consistance, taxonomie de Trevisan, dictionnaire complété. Excellent.
- **Erreurs :** dict. « Birkhoff … S15 mention » pendante (cf. A5).

### s13 — Coupling QOT = SDP ✅ (cœur réussi)
- **Fait :** lift LP→SDP, cvxpy 5 lignes, collapse diagonal vérifié, punchline |+⟩ vs I/2
  délivré, validation pure-state `1−|⟨ψ|φ⟩|²` (5 paires aléatoires), visualisation du couplage.
  Vrai travail d'ingénierie, validé.
- **Erreurs :** warnings CLARABEL `optimal_inaccurate` vs « ~1e-9 » annoncé (A5).
- **Pistes :** réconcilier explicitement le coût SWAP / quadratique avec le pont de Bures de
  s11 (cf. A3) — sinon l'étudiant ne sait pas quel « quantum Wasserstein » il calcule.

### s14 — Quantum Sinkhorn ⚠️
- **Fait :** SDP entropique (von Neumann), noyau de Gibbs matriciel `expm(−C/ε)`, sweep ε
  (sharp→produit), **pont d'Amari quantique vérifié** (Umegaki). Bon contenu.
- **Erreurs/étonnements :** (i) l'itération Sinkhorn opérateur n'est **jamais** lancée — SDP
  + assertion d'équivalence (A4) ; (ii) dérivation Amari fausse dans le docstring `sinkhorn.py`
  (A4) ; (iii) one-liner illisible cell `f0904ea8` (A5).
- **Pistes :** implémenter l'itération (matrix-exp + partial traces) et la confronter au SDP —
  c'est la compétence annoncée de la séance.

### s15 — Capstone 🔴 (le foyer du problème)
- **Fait :** dyade de Kuramoto bruitée, plongement phase-qubit, ρ_AB, 4 mesures, sweep K, riches
  « honest caveats » (biais, leakage, comparaisons multiples, direct-sum/tensor, phase-only).
- **Problème de fond :** le plongement rend QMI/Bures redondants avec PLV (A1) ⟹ la question
  centrale ne *peut* pas être tranchée par ce montage ; l'expérience décisive est en devoir ;
  les tests ne vérifient que la monotonie en K.
- **Erreur visible :** artefact « … wait it's … » cellule `2091922a` (A1).
- **Pistes :** Partie C.

### s16 — Frontier & synthesis ⚠️
- **Fait :** récap 15 séances, théorème de limitation VQE (De Palma 2023), taxonomie QOT,
  problèmes ouverts, `course_greatest_hits()`, dictionnaire final, bibliographie. **Prose
  riche, cadrage « limites » légitime et voulu.**
- **Étonnements :** (i) une seule cellule de calcul (synthèse — défendable) ; (ii) le §8 « What
  this course did NOT prove » présente le non-test du capstone comme « question ouverte » sans
  distinguer *« structurellement impossible sous ce plongement »* de *« ouvert sur données
  réelles »* — il hérite donc de A1.
- **Pistes :** réécrire ce §8 une fois le capstone réparé (Partie C).

---

## Partie C — Pistes de ré-architecture (à discuter, non décidé)

Tu veux ralentir, ajouter des étapes intermédiaires, et « architecturer différemment ». Quatre
chantiers, à prioriser ensemble :

1. **Réparer le capstone (s15)** — trois directions, idéalement A/B *puis* C :
   - **(A) Expérience discriminante** (l'item #3) : deux systèmes à même distribution de
     différence de phase mais structure d'ordre supérieur / amplitude différentes → QOT
     sépare-t-il là où PLV est aveugle ?
   - **(B) Plongement plus riche** : ρ_AB état-mixte / sensible à l'amplitude / multi-fréquence,
     qui encode plus que `|PLV|`.
   - **(C) Théorème honnête** : « sous le plongement phase-pure-produit, QMI/Bures sont monotones
     en PLV — voici la preuve + la figure (Spearman ≈ 1) ». Transforme le non-test en résultat.

2. **Insérer une séance « plongements »** entre M3 et le capstone : comment passer d'un signal à
   un ρ, qu'est-ce que chaque plongement préserve. Ralentit *et* corrige la cause racine de A1.

3. **Réconcilier les trois objets QOT** (A3) : une demi-séance ou un encart explicite reliant
   pont de Bures (S11) ↔ SDP SWAP/quadratique (S13) ↔ Bures-coupling/QMI (S15) — quand sont-ils
   égaux, quand divergent-ils.

4. **Découper M4** si on ajoute des séances : S13 et S14 gagnent à respirer (le SDP, puis
   *séparément* l'itération Sinkhorn réellement codée).

---

## Partie D — Liste d'actions priorisée (proposition)

1. 🔴 **Commiter s06→s16 proprement** (baseline diffable) — endossé.
2. 🔴 **Décider la direction du capstone** (A/B/C) et l'implémenter.
3. 🟠 **Réécrire s16 §8** une fois le capstone tranché.
4. 🟠 **Réconcilier les 3 objets QOT** (encart S13 + renvoi S11/S15).
5. 🟡 **Corriger A5** : warnings CLARABEL vs tol annoncée, cross-refs S15→S16, forward-ref
   Birkhoff pendante, dérivation Amari du docstring `sinkhorn.py`, one-liner s14.
6. 🟡 **s14** : coder l'itération Sinkhorn opérateur et la confronter au SDP.
7. (discussion) **séance « plongements »** + **découpage M4**.

---

## Annexe — Journal de vérification

- `uv run pytest -q` → **161 passed, 14 warnings** (les warnings = CLARABEL `optimal_inaccurate`).
- `course_greatest_hits()` : toutes les identités tiennent — Bell QMI = 2.000, S(A|B) = −1.000,
  W₂ closed = W₂ LP = 1.303840, pont d'Amari `tr(CP)−εS(P) = εS_Umegaki(P‖K)` = 0.222177.
- **Contrôle capstone** (8 seeds × 25 K = 200 dyades, `duration=200`, ω=(1.0,1.2)) :
  Spearman ρ(PLV,QMI)=0.9997, ρ(PLV,Bures)=0.9999 ; régression `f(PLV)` lisse (degré 4) :
  R²(QMI)=0.9997 (résidu σ=0.0011 sur étendue 0.335), R²(Bures)=0.9984 (résidu σ=0.0036 sur
  étendue 0.489). ⟹ QMI et Bures sont des fonctions monotones de PLV dans ce montage.
- Balayage d'artefacts sur les 16 notebooks : seul `s15` contient un « wait it's ».
