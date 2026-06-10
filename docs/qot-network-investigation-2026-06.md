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
| **A — réseau N-nœuds** | Le formalisme apporte-t-il une valeur méthodologique sur le couplage multipartite ? | **GATE FRANCHI (valeur modeste, honnête).** Pas d'avantage de détection. Le différenciateur réel = **la géométrie** : Bures clusterise les régimes de couplage (silhouette 0.474) nettement mieux que le descripteur classique naïf (0.191) et au niveau du descripteur expert (0.405) — un objet **canonique, sans-paramètre**, pas une capacité que le classique n'a pas. |
| **B — embedding appris** | Peut-on *optimiser* l'embedding contre une cible ? | **PREUVE DE CONCEPT.** Marche (3.9× equal-weight, 1.59× le hand-picked), mais redécouvre l'intuition « tuer le canal PLV ». Sa vraie valeur exige un problème où le bon embedding n'est *pas* évident. |
| **C — borne structurelle** | La borne R2 tient-elle au multipartite ? | **CONFIRMÉ (négatif-utile).** Au N=3, intriquer un embedding d'ordre trop bas ne conjure aucun couplage triadique d'ordre supérieur. Quantum-inspired = unification, jamais détection exclusive — maintenant établi au-delà des dyades. |

**Recommandation préliminaire (à décider en Phase 2), après le deep dive trajectoire :** le bilan
honnête est **plus sobre** qu'à la première lecture. La géométrie de Bures donne un avantage
**modeste et borné** : un win sur le **clustering statique** des régimes de couplage (vs le
descripteur classique naïf ; à parité avec le descripteur expert), mais **AUCUN** avantage en
**détection** (R2) ni en **détection temporelle de reconfiguration** (deep dive ci-dessous :
la vitesse-TC classique gagne). La valeur réelle est donc : *un langage géométrique unifié,
canonique et sans-paramètre pour le couplage multipartite* — conceptuellement propre et
pédagogiquement net, **empiriquement marginal**. Ce n'est pas un breakthrough ; c'est un reframing
méthodologique honnête. À Rémy de juger si cela mérite une série de notebooks « pour aller plus
loin » (cadrée sans surclamer) ou seulement un encart. B reste un add-on ; C un livrable d'intégrité.

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
classique possède **déjà** une mesure multivariée unique et principielle, et incontestée —
la **total correlation** (Watanabe 1960 ; = interaction information de McGill 1954 dans le cas
multivarié). C'est exactement ce que Track A calcule (gap 2.19, bedrock de la théorie de
l'information). La TC quantique ne fait que la dupliquer. **À ne pas vendre comme un atout quantique.**

> **Mise en garde (Rémy, 2026-06-10).** On **n'ancre pas** ce point sur l'**O-information** ni sur le
> programme *synergie / émergence / information intégrée* (Rosas, Mediano, Luppi… ; p. ex. l'application
> à la conscience dans eLife 2024, DOI:10.7554/eLife.88173). La *formule* O-information est bien
> définie, mais son **overlay interprétatif est contesté** (l'IIT a été publiquement qualifiée de
> pseudoscience par ~120 chercheurs en 2023), et Rémy n'a pas confiance dans ces travaux. Le point Q1
> tient **sans eux**, sur la total correlation de Watanabe (60 ans, non contestée). On ne construit
> rien sur le programme contesté.

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

**Phase-2 gate — « la métrique de Bures sert-elle à quelque chose ? » : OUI (modeste, honnête).**
Test de récupération de régimes : 5 régimes de couplage (indépendant, par-paires, chaîne,
triadique 1er ordre, triadique 2e ordre), 6 instances chacun, distance de Bures entre les `ρ`
qutrit, score de silhouette sur les labels de régime. Comparé à deux baselines classiques. **Très
stable (8 seeds, std 0.002) :**
```
Bures geometry (qutrit, sans paramètre)                  silhouette 0.474
classique NAIF  (MI par paires x3 + total correlation)   silhouette 0.191
classique EXPERT (PLV1x3, PLV2x3, résultants triadiques x2, TC) silhouette 0.405
```
→ Bures **≫ naïf** (le descripteur classique par défaut rate la structure triadique) et
**≈ légèrement > expert**. **Ce n'est PAS un fossé de capacité** : un classique pleinement informé
y arrive presque (0.405). La valeur de Bures est : **un objet géométrique unique, canonique, sans
paramètre**, qui capte automatiquement la structure multipartite (triadique incluse) *sans que
l'analyste ait à savoir quels features d'ordre supérieur calculer* — et qui ouvre des opérations
géométriques natives (clustering de régimes, géodésiques, trajectoire de couplage dans le temps)
qu'un vecteur de features ne fournit pas. L'edge marginal (~17 %) est réel mais modeste.

**Verdict A :** **gate franchi** — candidat notebooks, sur une proposition de valeur *honnête et
modeste* : une géométrie de transport/Bures canonique et sans-paramètre sur les configurations de
couplage multipartite. Aucun avantage de détection ; Q1 n'en est pas un (Watanabe) ; Q2/Q3 sont le
cœur ; l'avantage est *méthodologique* (canonicité, opérations géométriques), pas une capacité que
le classique n'a pas.

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

## Deep dive (2026-06-10) — la trajectoire temporelle de couplage : NÉGATIF honnête

L'angle le plus distinctif de la géométrie : l'hyperscanning est résolu dans le temps, donc une
métrique sur les configurations transforme la *dynamique* du couplage en un **chemin** (vitesse de
Bures `d_B(ρ(t), ρ(t−1))` sur fenêtres glissantes). Test décisif : une transition entre deux régimes
**à total correlation appariée** mais structure différente (chaîne → triadique) — la vitesse de
Bures devrait y piquer là où la vitesse-TC reste plate.

**Résultat (mené proprement, après correction d'un premier non-test où les époques n'étaient pas
appariées) :**
```
transition TC-appariée, couplage FAIBLE (TC~0.45) : Bures 1.3x   vs  TC-velocity 3.5x
transition TC-appariée, couplage FORT  (TC~1.4)  : Bures 4.5x   vs  TC-velocity 22x
```
→ **Bures ne bat PAS la vitesse-TC pour détecter la transition**, dans les deux cas. **Diagnostic
honnête :** même à TC moyenne appariée entre régimes, la fenêtre qui *chevauche* la transition
mélange les deux régimes, et la TC d'un mélange diffère de celle des époques pures → la vitesse-TC
pique quand même (artefact de mélange), et la fenêtre (W=2000) est trop bruitée pour résoudre le pas
de Bures structure-préservant. **La détection de point de changement ne donne aucun avantage à la
géométrie** — le scalaire classique fait le travail.

**Ce que ce négatif tempère :** la géométrie aide pour la **caractérisation/clustering statique** des
configurations de couplage (gate Phase-2, win modeste), **pas** pour la **détection temporelle** de
reconfiguration. Le programme « trajectoire de couplage » ne rescussite pas un avantage mesurable.

## Implications pour la Phase 2 (décision)

1. **Track A, recentrée Q2** = le seul candidat solide pour des notebooks : une **géométrie de
   transport/Bures sur les configurations de couplage multipartite**, positionnée *honnêtement*
   contre la **total correlation classique (Watanabe)** — l'apport est la **métrique canonique
   inter-configurations**, pas la détection ni un premier scalaire multivarié. (On se positionne
   contre le standard incontesté, **pas** contre l'O-information / le programme synergie-émergence.)
2. **Le test à passer avant tout notebook A : PASSÉ** (gate Phase-2 ci-dessus) — Bures clusterise
   les régimes de couplage nettement mieux que le descripteur classique naïf et au niveau (un peu
   au-dessus) du descripteur expert, de façon stable. La valeur est *méthodologique* (canonique,
   sans-paramètre, opérations géométriques natives), pas une capacité de détection. Donc A est un
   candidat-notebooks **légitime mais à cadrer honnêtement** (ne pas survendre l'edge).
3. **Track B** = encart/extension de A (embedding appris pour une cible), pas une série autonome.
4. **Track C** = un *encart d'intégrité* (ou un court write-up méthodo), pas des notebooks.
5. **Le SDP de transport multipartite** (vraie distance de Wasserstein quantique entre états de
   réseau, au-delà de Bures) reste le gros chantier ouvert — à n'ouvrir que si A-Q2 convainc.

## Références (vérifiées)

**Le baseline classique sur lequel on s'appuie (incontesté) :**
- S. Watanabe, "Information theoretical analysis of multivariate correlation", *IBM J. Res. Dev.* **4**,
  66–82 (1960) — total correlation / multi-information : la mesure multivariée principielle unique que
  Track A calcule. *(classique ; confirmer le DOI exact avant usage externe.)*
- W. J. McGill, "Multivariate information transmission", *Psychometrika* **19**, 97–116 (1954) —
  interaction information. *(classique ; confirmer le DOI exact avant usage externe.)*
- M. M. Wilde, *Quantum Information Theory*, 2nd ed. (CUP, 2017). DOI:10.1017/9781316809976 — entropies
  quantiques, QMI, total correlation quantique.

**Programme contesté — cité pour le situer, PAS pour s'appuyer dessus :**
- F. Rosas, P. A. M. Mediano, M. Gastpar & H. J. Jensen, "Quantifying high-order interdependencies via
  multivariate extensions of the mutual information", *Phys. Rev. E* **100**, 032305 (2019).
  DOI:10.1103/PhysRevE.100.032305 — l'O-information (formule bien définie ; overlay synergie/émergence contesté).
- A. I. Luppi, P. A. M. Mediano, F. E. Rosas et al., "A synergistic workspace for human consciousness
  revealed by Integrated Information Decomposition", *eLife* **12** (2024). DOI:10.7554/eLife.88173 —
  application à la conscience (IIT/Global Workspace). **Rémy : faible confiance ; on ne construit pas dessus.**
  (Métadonnées via PubMed.)
