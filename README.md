# Bimodal colloids highlight the structural mirror of rigidity percolation and yielding
Results from this project are [available on arXiV](https://doi.org/10.48550/arXiv.2607.22901) and are currently under review.

This is a simulation and analysis pipeline for identifying the mechanically relevant structures that produce elastic response and yielding behavior in colloidal depletion gels.

Simulations can be run using the [sim-scripts](./sim-scripts) and the DPDMorse extension for HOOMD-blue v4.2.1, available as [hoomd4.2.1-mod](https://github.com/procf/hoomd4.2.1-mod).

To confirm that a system is at equilibrium before gelation, use [analysis-eq](./analysis-scripts/analysis-eq)

Shear simulation script is available for the [bimodal system](./sim-scripts/bi-colloid_50-50/shear-const-0.5-allinit-bi-DPD) 

## What to expect
For systems of ~10,000 colloidal particles the following scripts are available:

[analysis-GMM](./analysis-scripts/analysis-GMM)
- Gaussian Mixture Model (GMM) based mesoscale clustering

[analysis-topology](./analysis-scripts/analysis-topology)
- Construct a network representation of colloid-colloid bonds
- Identify tetrahedral structures and their hierarchical motifs (NOTE: motif data not used in this study)
- Calculate edge-betweenness centrality (EBC) for all contacts and isolate edges with the top 10% of values
- Classify cluster-cluster bridges as "singly-connected bridges" (SB) or "multi-connected bridges" (MB)
- In a bimodal system, save the bond-type composition of tetrahedral motifs, bridges, and the EBC>=P90 backbone
- Save ECDF and CCDF data for the EBC values of different structures
- Save data comparing all three structural classes (tetrahedra, bridges, high-EBC backbone)

[analysis-mech](./analysis-scripts/analysis-mech)
- Quench the system to remove negative-curvature artifacts
- Calculate the relative contribution of different bonds to the affine (Born) contribution to the static shear modulus
- Calculate per-bond non-affine projection 

[analysis-yielding](./analysis-scripts/analysis-yielding)
- Extract shear stress from simulation gsd-properties
- Measure bond orientation from the fabric tensor
- Calculate per-particle non-affine motion using Falk-Langer $\langle D^2_{min} \rangle$

plot-script
- Calculate relative localization of the non-affine response using a per-bond harmonic relaxation energy
- Classify bond breaks that occur during shear
- Calculate cumulative mean strain per bond



Note: GMM clustering and FIRE quenching can each take ~9hrs. Other analyses typically take less than 1min. 

In bimodal depletion gels, colloid-colloid interactions scale with particle size. Therefore, the attraction strength for S-S, S-L, and L-L contacts will be different.
For size ratio 1:2 this scales roughly with the arithmetic mean, such that $D_0^{SL} = 1.5 D_0^{SS}$ and $D_0^{LL} = 2 D_0^{SS}$.

## Software/package requirements
In this project, the following packages are actively used:
1. GNU Fortran (GCC) 11.4.1
2. `python` v3.10.16 
3. `gsd` v3.2.1
4. `numpy` v1.24.40
5. `pandas` v2.2.3
6. `networkx` v3.4.2
7. `scipy` v1.15.3
8. `node2vec` v0.5.0
9. `umap-learn` v0.5.9
10. `scikit-learn` v1.7.2
11. `freud-analysis` v3.4.0
12. `joblib` v1.5.2

## Hardware/OS tested
The program was tested on a single HPC-node running Rocky Linux 9.3 (kernel 5.14).

## Background

The fact that rigidity percolation and yielding can both be described as non-linear phase transitions makes it appealing to
believe that these processes are mirror images that form a single continuous transition. However, in amorphous materials, 
these transitions usually occur through complex multiscale interactions among a variety of structural features. We use large-scale 
simulations to compare 3 different structural classes commonly studied in colloidal gels: locally rigid tetrahedra, mesoscale 
cluster-cluster bridges, and connectivity-critical bonds with high values of edge-betweenness centrality (EBC). We find that bridges 
and high-EBC bonds appear to perform similar mechanical roles in both bulk elastic response and yielding behavior, which suggests that
the high-EBC subset describes the structural mirror that carries these transitions. We also find that the large particles 
present in a bimodal colloidal gel effectively label these features.

Here's the abstract: 

*In metastable particulate gels, it is tempting to believe that the dynamic similarities between
the fluid-to-solid non-linear phase transition of rigidity percolation and the solid-to-fluid transition
that occurs during yielding represent mirror images of the same continuous process. Even though
these behaviors are clearly dynamically similar, their multi-scale nature makes it difficult to deter-
mine if they could also follow a unified structural pathway. We know from model monodisperse
colloidal gels that both yielding and the elastic modulus seem to be heavily influenced by a small
subset of topologically distinct singly-connected bridges linking mesoscale features. Here we use
particle simulations to examine the participation of different classes of particle-level bonds and
their contributions to the bulk mechanical response. We find that rigidity is disproportionately
supported by singly connected intercluster bridges, whereas yielding localizes at bonds with high
edge-betweenness centrality (EBC); strikingly, these independently identified populations substan-
tially overlap and perform comparable mechanical roles. Bimodality exposes this correspondence by
concentrating large-particle contacts in both populations, thereby providing a compositional label
for the common backbone. Thus, rigidity and yielding are opposing mechanical manifestations of
the same mesoscale structure: the intercluster bottlenecks that establish rigidity are also the sites
at which rigidity is preferentially lost.*


## Contributors
This project was in collaboration with the [Soft Matter Engineering Laboratory](https://smel.eng.uci.edu/) at the University of California, Irvine. \
This work was done by [Rob Campbell](https://scholar.google.com/citations?user=i8S54zYAAAAJ&hl=en), Calvin (Ziye) Zhuang, 
[Ali Mohraz](https://scholar.google.com/citations?user=pW80NaAAAAAJ&hl=en), and [Safa Jamali](https://scholar.google.com/citations?user=D1asaYIAAAAJ&hl=en).\
Authors acknowledge support from the National Science Foundation (PMP-2025613) and NASA ROSES FINESST (80NSSC23K0015).
