# QOT research program — Phase 1 investigation (N-node network · learned embedding · the bound)

**Auteur :** Claude (Fable), enquête demandée par Rémy.
**Date :** 2026-06-10
**Statut :** note de recherche Phase 1 — **point de décision Phase 2 ensuite** (quelles pistes méritent
des notebooks « pour aller plus loin »). Rien n'est gravé ; un track négatif est un résultat honnête.
**Branche :** `feat/qot-network-investigation`. **Origine :** discussion stratégique post-PR #16.
**Reproductibilité :** scripts jetables `/tmp/qot_net_trackA.py`, `_trackA_q2.py`, `_trackB.py`,
`_trackC.py` ; réutilisent le src du cours ; seeds loggés ; builders validés contre formes closes.

> **Cadre.** La borne R2 (PR #16) a déjà tranché : sur données classiques déterministes, la QOT n'a
> **pas** d'avantage de *détection* — elle est *quantum-inspired*. La Phase 1 ne re-chasse donc pas
> la détection ; elle teste un **avantage méthodologique** (mesure principielle, géométrie,
> indépendance de représentation), toujours **face au baseline classique apparié**.

---

## Verdict en une page

| Track | Question | Verdict |
|---|---|---|
| **A — réseau N-nœuds** | Le formalisme apporte-t-il une valeur méthodologique sur le couplage multipartite ? | **PROMETTEUR, mais plus étroit que prévu.** Pas d'avantage de détection (confirmé). Le différenciateur réel est **la géométrie** : Bures donne une **métrique canonique sur les configurations de couplage** que le toolbox classique (total correlation, O-information) n'offre pas. |
| **B — embedding appris** | Peut-on *optimiser* l'embedding contre une cible ? | **PREUVE DE CONCEPT.** Marche (3.9× equal-weight, 1.59× le hand-picked), mais redécouvre l'intuition « tuer le canal PLV ». Sa vraie valeur exige un problème où le bon embedding n'est *pas* évident. |
| **C — borne structurelle** | La borne R2 tient-elle au multipartite ? | **CONFIRMÉ (négatif-utile).** Au N=3, intriquer un embedding d'ordre trop bas ne conjure aucun couplage triadique d'ordre supérieur. Quantum-inspired = unification, jamais détection exclusive — maintenant établi au-delà des dyades. |

**Recommandation préliminaire (à décider en Phase 2) :** la seule piste qui mérite vraiment des
notebooks est **A, recentrée sur la géométrie (Q2)** — *une métrique de transport/Bures sur les
configurations de couplage multipartite*, ce que le classique n'a pas canoniquement. B est un
add-on utile à A. C est un livrable d'intégrité (papier/encart), pas une série de notebooks.

---

## Track A — Réseau N-nœuds

**Dispositif make-or-break (vérité-terrain).** Analogue XOR de phase : `θ_C = θ_A + θ_B (+ bruit)`
avec θ_A, θ_B uniformes indépendantes → triple **entièrement couplé mais par-paires indépendant**.
Toutes les mesures bivariées sont aveugles par construction ; seul un objet multipartite peut voir.

**Builders validés** (formes closes) : joint density N-aire = bipartite pour N=2 (diff 0), trace=1 ;
total correlation classique d'indépendants ≈ 0.02 ; TC quantique d'un triple indépendant ≈ 0.

**Make-or-break (N=60k) :**
```
pairwise PLV / MI (A-B, A-C, B-C) : tous ~0.00 pour triadic ET null   (aveugle par construction)
classical total correlation       : triadic=2.21  null=0.02  gap=2.19   (voit)
quantum   total correlation       : triadic=0.23  null=0.00  gap=0.23   (voit aussi)
```
→ **Pas d'avantage de détection** : les deux voient le couplage triadique (R2 confirmé au multipartite).

**Q1 — « une mesure principielle vs un zoo » : NON, ce n'est pas un avantage.** Honnêteté : le
classique possède **déjà** des mesures multivariées uniques et principielles — total correlation
(Watanabe 1960), interaction information (McGill 1954), et surtout l'**O-information** (Rosas,
Mediano, Gastpar & Jensen 2019, *Phys. Rev. E* 100, 032305, DOI:10.1103/PhysRevE.100.032305 —
activement utilisée en neuro). La TC quantique ne fait que les dupliquer. **À ne pas vendre comme
un atout quantique.**

**Q2 — la géométrie : le différenciateur réel (WIN honnête).** Test décisif : deux configurations à
**total correlation appariée** mais structure différente (une *triadique*, une *par-paires*, accordées
à la même TC quantique).
```
triadic TC = 0.147   vs   pairwise TC = 0.186   (|Δ| = 0.039, le scalaire les croit identiques)
Bures(triadic, pairwise) = 0.347   (~9× la différence de TC)
```
→ Bures **sépare nettement** deux configs que le scalaire TC confond. Le point n'est pas « Bures voit
plus » (trivial : matrice 8×8 vs scalaire) — c'est que **le cadre fournit *une* métrique canonique et
sans-binning sur les configurations de couplage**, là où TC et O-information sont des scalaires *par*
config, **sans distance inter-config canonique**. Sweep : Bures(config, null) est **monotone** dans la
force triadique (0.36→0.035), donc une vraie métrique utilisable. **C'est l'objet qui justifierait le
programme.**

**Q3 — indépendance de représentation (avantage réel, à nuancer).** La TC classique **dérive
fortement avec le binning** (n_bins 6/12/24/48 → 1.45/2.21/2.71/3.13, croissance non bornée — fléau
connu de l'estimation d'entropie continue). La mesure quantique **n'a pas de paramètre de binning**
(embedding continu fixe). Nuance honnête : elle *échange* le binning contre un **choix d'embedding**
(aussi un cadran), et l'O-information a des estimateurs continus ; l'avantage est « représentation
continue interprétable » plutôt que « zéro arbitraire ».

**Verdict A :** prometteur, **recentré sur Q2** (la géométrie/transport sur les configurations).
Aucun avantage de détection ; Q1 n'en est pas un ; Q2 est le cœur, Q3 un appoint.

---

## Track B — Embedding appris (preuve de concept)

Optimisation sans gradient (Nelder-Mead, restarts) des 3 poids du `weighted_multifreq_state` pour
maximiser le gap rich-QMI sur les ensembles matched-PLV (a2=0 vs 0.3) :
```
equal (1,1,1)      gap = 0.0060
hand-picked(1,0.2,2) gap = 0.0149
LEARNED (1,0,1)    gap = 0.0236   (3.9× equal, 1.59× hand-picked)
```
→ l'optimiseur **conduit le poids PLV à 0** et boost h2 : il **redécouvre** l'intuition de brick 17 et
la pousse plus loin. **Preuve de concept valide**, mais sur ce jouet à cible unique il automate une
intuition connue. Sa valeur réelle serait sur un problème où le bon embedding n'est *pas* évident
(p. ex. cible comportementale/clinique) — à démontrer en Phase 3. Risque flaggé : peut se réduire à
du feature-learning ordinaire.

---

## Track C — Borne structurelle, étendue au multipartite (confirmé)

Couplage triadique visible **seulement à la 2e harmonique** : `θ_C = θ_A + θ_B + {0 ou π}` →
`⟨e^{i(a+b−c)}⟩|≈0` (1er ordre aveugle), `|⟨e^{2i(a+b−c)}⟩|≈1` (le couplage y vit).
```
classical 1st-harmonic triadic resultant = 0.001   (aveugle)
classical 2nd-harmonic triadic resultant = 1.000   (le lit)
quantum TC, qubit embedding (1er moment)  = 0.0000  (aveugle)
quantum TC, qubit embedding ENTANGLED     = 0.0000  (toujours aveugle -> l'intrication n'ajoute rien)
quantum TC, qutrit embedding (2e moment)  = 0.0743  (le voit)
```
→ La borne R2 **s'étend au multipartite** : la détection est fixée par les moments que l'embedding
capte ; intriquer un embedding d'ordre trop bas ne conjure pas la structure d'ordre supérieur ; un
statistique classique 2e-harmonique la lit. **Quantum-inspired = unification, jamais détection
exclusive, sur données classiques — confirmé à N=3.** C'est le garde-fou d'intégrité du programme.

---

## Implications pour la Phase 2 (décision)

1. **Track A, recentrée Q2** = le seul candidat solide pour des notebooks : une **géométrie de
   transport/Bures sur les configurations de couplage multipartite**, positionnée *honnêtement*
   contre l'O-information (le standard classique multivarié) — l'apport est la **métrique canonique
   inter-configurations**, pas la détection ni un premier scalaire multivarié.
2. **Le test à passer avant tout notebook A** : montrer que la métrique de Bures fait quelque chose
   d'*utile* qu'un classique ne fait pas — p. ex. classer/clusteriser des configurations de couplage
   de façon stable, ou définir une géodésique entre régimes de couplage. Sinon A reste « joli mais
   sans usage ».
3. **Track B** = encart/extension de A (embedding appris pour une cible), pas une série autonome.
4. **Track C** = un *encart d'intégrité* (ou un court write-up méthodo), pas des notebooks.
5. **Le SDP de transport multipartite** (vraie distance de Wasserstein quantique entre états de
   réseau, au-delà de Bures) reste le gros chantier ouvert — à n'ouvrir que si A-Q2 convainc.

## Références (vérifiées)

- F. Rosas, P. A. M. Mediano, M. Gastpar & H. J. Jensen, "Quantifying high-order interdependencies
  via multivariate extensions of the mutual information", *Phys. Rev. E* **100**, 032305 (2019).
  DOI:10.1103/PhysRevE.100.032305 (arXiv:1902.11239). *(vérifiée par recherche web)* — l'O-information :
  le standard classique multivarié dont Track A doit se distinguer (et donc la raison d'abandonner Q1).
- S. Watanabe, "Information theoretical analysis of multivariate correlation", *IBM J. Res. Dev.* **4**,
  66–82 (1960) — total correlation (multi-information). *(classique ; confirmer le DOI avant usage externe.)*
- W. J. McGill, "Multivariate information transmission", *Psychometrika* **19**, 97–116 (1954) —
  interaction information. *(classique ; confirmer le DOI avant usage externe.)*
- M. M. Wilde, *Quantum Information Theory*, 2nd ed. (CUP, 2017). DOI:10.1017/9781316809976 — entropies
  quantiques, QMI, total correlation quantique.

> Note : refs O-information vérifiée en ligne ; Watanabe/McGill sont des classiques bien établis mais
> leurs DOI exacts restent à confirmer avant toute citation externe (pas de DOI inventé ici).
