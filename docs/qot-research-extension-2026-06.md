# QOT research extension — amplifying the effect & the structural bound

**Auteur :** Claude (Fable), recherche demandée par Rémy.
**Date :** 2026-06-09
**Statut :** note de recherche — **à valider par Rémy avant toute modification de notebook** (Phase P).
**Origine :** suite de l'audit `docs/audit-notebooks-2026-06-09.md` (Annexe D) et des 3 questions de Rémy
(QOT atteignable ? QMI améliorable ? a-t-on tout exploré ?).
**Reproductibilité :** scripts jetables `/tmp/qot_r1_amplify.py`, `/tmp/qot_r1_confirm.py`,
`/tmp/qot_r2_exclusive.py` ; réutilisent le src du cours ; seeds loggés ; N=150 000 sauf indication.

> **But.** Répondre par la mesure (pas par le raisonnement) à : (R1) peut-on *agrandir* l'effet
> QOT>PLV pour le rendre mesurable ? (R2) existe-t-il un régime où QOT voit ce qu'**aucune**
> statistique classique appariée ne voit, ou doit-on *démontrer* la borne qui l'en empêche ?
> Toutes les conclusions sont comparées au baseline classique apparié, pour ne jamais surclamer.

---

## Verdict en une page

- **R1 — l'effet s'agrandit, beaucoup.** Baisser le poids du canal PLV (1ère harmonique) dans le
  plongement multi-fréquence fait passer la significativité de la séparation de **7.6σ à ≫1000σ**
  (en supprimant la source de bruit dominante), et le gap QMI de 0.0044 à 0.0145 nat — **au-dessus
  du plancher de bruit hardware** (~0.012 nat, 05/06). Le plongement *amplitude* ouvre un canal de
  couplage **phase-amplitude** que la PLV ne voit pas. → la **mesurabilité** est un vrai gain.
- **R2 — la borne structurelle est démontrée (pas supposée).** Une unitaire *authentiquement
  intriquante* (1.5 bit d'intrication) sur un plongement d'ordre trop bas **reste aveugle** au
  moment d'ordre supérieur. Pouvoir de détection QOT = moments classiques que *l'embedding capte* ;
  aucun avantage exclusivement quantique depuis un plongement déterministe de données classiques.
- **Donc, sur les 3 questions de Rémy :** la QOT est atteignable (et amplifiable) ; la QMI
  s'améliore en *mesurabilité* mais pas en *exclusivité* ; on n'avait pas tout exploré, et
  l'exploration menée ici **ferme honnêtement la porte de l'avantage exclusif** pour des données
  classiques tout en **ouvrant** la voie pratique (plongements pondérés, amplitude, réseaux).

---

## R1 — Agrandir l'effet mesurable

Tous les essais sur les ensembles à PLV appariée (a₁=0.4 ; a₂_low=0.0 vs a₂_high=0.3 ;
PLV gap = 0.0017, PLV2 gap = 0.298).

### R1.1 — Pondération du plongement multi-fréquence (le levier décisif)

`multifreq_state` utilise des poids égaux `1/√d`. En pondérant les canaux (w_DC, w_h1, w_h2) :

| Pondération | gap QMI (nat) | vs equal-weight |
|---|---|---|
| equal (1,1,1) — baseline | 0.00440 | ×1.0 |
| boost h2 (1,1,2) | 0.00708 | ×1.6 |
| drop h1 (1,0.5,2) | 0.01270 | ×2.9 |
| **DC+h2 (1,0.2,2)** | **0.01450** | **×3.3** |

**Significativité (null a₂-apparié, 12 seeds) :**

| Pondération | gap | null mean ± std | z |
|---|---|---|---|
| equal (1,1,1) | 0.00440 | 0.00071 ± 0.00048 | **7.6σ** |
| DC+h2 (1,0.2,2) | 0.01450 | 0.00001 ± 0.00001 | **≫1000σ** |

Mécanisme : le canal h1 *est* la PLV ; il porte le couplage dominant **et** le bruit dominant.
Le baisser concentre la QMI sur le canal h2 discriminant et **fait s'effondrer la variance du
null** → la séparation devient quasi sans recouvrement. (Le z exact ≈1700 est fragile — null std
minuscule estimé sur 12 seeds — mais le **saut de plusieurs ordres de grandeur est robuste**.)
C'est un résultat de *denoising par design de plongement*, légèrement surprenant et utile.

### R1.2 — Quel objet QOT comme readout ?

Sur la même `ρ_AB` qutrit (equal-weight) : **QMI 0.00440 > Bures 0.00201 ≫ SDP-entre-marginales 0.0**.
La mesure doit lire le **joint** (cohérences), pas les marginales : le coût-transport entre
marginales est ~0 ici (les marginales sont ~I/d pour les deux ensembles) — c'est le mauvais outil
pour un *readout de couplage*. QMI domine Bures.

### R1.3 — Plongement amplitude : un canal de couplage nouveau

Deux ensembles à **phases identiques** (donc PLV *et* qutrit-de-phase appariés, PLV=0.4003) mais
couplage *phase-amplitude* différent (amplitude indépendante vs amplitude ∝ cos(Δθ)) :
plongement amplitude → **gap QMI = 0.00739**. Le canal amplitude expose un couplage que la PLV est
aveugle à par construction — pertinent pour la neuro (couplage phase-amplitude / amplitude-amplitude).

### R1.4 — Plus d'harmoniques

(1,2)→0.00440 · (1,2,3)→0.01132 · (1,2,3,4)→0.01743. **Non-artefactuel** : les ensembles ont
a₃,a₄ ≈ 0.002 pour les *deux* (vérifié) ; le gain vient de la non-linéarité globale de la QMI, pas
d'un moment non-apparié.

**Bilan R1.** L'effet est amplifiable ~3× en gap et de plusieurs ordres de grandeur en
significativité, **au-dessus du plancher hardware**. Mais cela reste **non-exclusif** : la PLV2
classique sépare toujours à 0.298 (≫ tous ces gaps). R1 achète la *mesurabilité/faisabilité*, pas
l'exclusivité quantique.

---

## R2 — Avantage exclusivement quantique ? Non : la borne structurelle, démontrée

**Test décisif.** Deux ensembles appariés sur a₁ **et** a₂, différant **uniquement** par a₃
(PLV3 gap = 0.299, le seul écart classique) :

| Mesure | gap (nat) | lecture |
|---|---|---|
| QMI qutrit (1,2) | 0.00016 | **aveugle à a₃** (ne capte que a₁,a₂) |
| QMI qudit (1,2,3) | 0.00374 | voit a₃ — **mais la PLV3 classique aussi** |
| **QMI plongement INTRIQUANT (1,2)** | **0.00010** | **toujours aveugle**, malgré 1.515 bit d'intrication |

Converse — appariés sur a₁,a₂,a₃ (tout le classique) → toutes les mesures QOT appariées
(gaps 0.0002–0.0005), l'intrication n'ajoute rien.

**Conclusion (mesurée).** Une unitaire authentiquement intriquante sur un plongement d'ordre trop
bas **ne peut pas conjurer** l'accès à un moment d'ordre supérieur. Énoncé :
> **Pouvoir de détection QOT ⊆ moments classiques de l'ordre que le plongement capte.** Égal, pas
> supérieur, en détection ; l'intrication du plongement ne change rien. Pour tout plongement
> *déterministe* de données de phase *classiques*, `ρ_AB` est une moyenne empirique de fonctions
> déterministes de (θ_A,θ_B) — ses entrées sont des moments classiques — donc une statistique
> classique également informée matche toute mesure QOT.

**Quand cela changerait (frontière, hors EEG/MEG) :** (i) donnée *intrinsèquement quantique* /
ressource quantique partagée (Bell/contextualité → aucune jointe classique n'existe) ; (ii)
plongement par *mesure quantique* (POVM) ou *canal quantique*, dont la sortie n'est plus un moment
classique de l'entrée. Les deux sont hors d'atteinte pour de l'EEG/MEG enregistré (voir
positionnement).

---

## Positionnement : une méthode *quantum-inspired*, à valeur méthodologique

La QOT-sur-EEG est **quantum-inspired** : on emprunte le formalisme quantique (matrice densité,
info quantique, transport optimal) comme outil d'analyse de données *classiques*, sans hardware ni
revendication d'avantage quantique. Sa valeur (confirmée par R1/R2) est **méthodologique**, pas une
supériorité de détection :

- **Cadre géométrique unifié** : PLV, couplage d'ordre supérieur, couplage d'amplitude, multi-canaux
  vivent dans un seul `ρ_AB` avec une famille de distances (QMI/Bures/transport).
- **Opérations composables** : trace partielle = marginalisation, entropie = incertitude, transport
  = comparaison.
- **Généralisation naturelle aux réseaux N-nœuds** (hyperscanning multi-sujets / multi-régions).
- **Accès *principé* à l'ordre supérieur** par le choix (et la *pondération*, cf. R1) de plongement,
  plutôt qu'une statistique choisie à la main.

**Capteur quantique ≠ donnée quantique.** L'OPM-MEG utilise des magnétomètres à pompage optique
(effets quantiques dans une vapeur atomique) mais « convertit les propriétés quantiques des atomes
en un signal électrique **classique** » (Brookes et al. 2019). Il aide la *mesurabilité* (SNR,
localisation → estimation plus fiable de l'ordre supérieur, donc utile pour R1), **pas**
l'exclusivité quantique : la donnée reste classique, la borne R2 tient.

### Littérature (références réelles à vérifier/citer en Phase P — ne pas inventer de DOI)

- Quantum-inspired density-matrix sur séries cérébrales : *EPJ Special Topics* (2026),
  « Quantum-inspired density-matrix recurrence analysis of brain time series ».
- Hyperscanning quantum-inspired (très proche de notre thèse — termes off-diagonaux ≈ synchronie de
  phase) : *PNAS* (2026), « Quantum-inspired entanglement between collaborating brains during human
  memory encoding ». **À positionner comme related work.**
- Représentation « quantum-like » de réseaux neuronaux : *Front. Hum. Neurosci.* (2025) /
  arXiv:2509.16253.
- Info-géométrie sur EEG : PMC10969156 (2024).
- OPM-MEG (capteur quantique → signal classique) : Brookes et al., *NeuroImage* (2019), PMC6988110 ;
  revue *Trends in Neurosciences* (2022), S0166-2236(22)00102-3.

---

## Implications pour la Phase P (notebooks — gated, après validation Rémy)

1. **04/16 `capstone_synthesis`** : (a) séparer avantage sim modeste/non-exclusif vs sous-bruit
   hardware ; (b) remplacer le caveat empirique par la **borne structurelle démontrée** (R2) ;
   (c) adopter le label **quantum-inspired** + valeur méthodologique + frontière « vraiment
   quantique ».
2. **Nouveau brick possible (R1)** : `embedding_design_amplifies_the_effect` — la pondération du
   canal PLV porte l'effet au-dessus du bruit hardware ; le plongement amplitude ouvre un canal
   nouveau. (one-concept-one-notebook, commit, STATE.md.)
3. **Nouveau brick possible (R2)** : `what_no_classical_data_embedding_can_buy` — la borne
   structurelle et sa frontière. Renforce l'honnêteté du capstone sans changer sa conclusion.
4. **05/06** : recalculer le 7.6σ inline (ou l'adoucir) ; il est confirmé reproductible (R1.1).
5. **CLARABEL** + **01/03 prereq** : corrections de l'audit.

> Rien de cela n'est appliqué. La science ci-dessus est la base de discussion ; les notebooks ne
> bougent qu'après ton feu vert.
