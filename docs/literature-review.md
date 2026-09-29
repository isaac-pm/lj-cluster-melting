# Literature Review: Molecular Model of Melting in Lennard-Jones Clusters

**Project 6, Computational Methods (Université du Luxembourg, 2026)**
Stage 2 deliverable: literature survey. Compiled 2026-09-29.

---

## Contents

0. [Scope, search strategy and verification](#0-scope-search-strategy-and-verification)
1. [The project in one page](#1-the-project-in-one-page)
2. [Core papers: detailed reviews](#2-core-papers-detailed-reviews)
3. [Supporting literature: compact reviews](#3-supporting-literature-compact-reviews)
4. [Cross-paper synthesis by research direction](#4-cross-paper-synthesis-by-research-direction)
5. [Research-gap map](#5-research-gap-map)
6. [Concrete research directions](#6-concrete-research-directions)
7. [Consequences for our implementation](#7-consequences-for-our-implementation)
8. [Bibliography](#8-bibliography)

---

## 0. Scope, search strategy and verification

### 0.1 Guiding question

The course slides ask us to simulate an N = 13 Lennard-Jones (LJ) cluster with velocity Verlet and to follow the "crossover from solid-like vibrations to liquid-like structural rearrangements" as the energy is raised, using the relative bond-length fluctuation (Lindemann-type index) δ as the indicator. This review is organised around one question, which the literature has so far answered only in part:

> **What makes a finite cluster "solid" or "liquid", and how can we measure that from a single constant-energy trajectory without an arbitrary threshold and without expensive canonical sampling?**

Everything below is read against that question. For each paper we ask how it decides whether the cluster has melted, what counts as ground truth, and whether its indicator is checked against an independent observable.

### 0.2 Topics searched

1. Foundations: the LJ potential, early molecular dynamics (MD), the Verlet and velocity-Verlet integrators.
2. Numerical analysis of MD: symplectic integration, energy conservation, shadow Hamiltonians, choosing the timestep.
3. Simulations of rare-gas (LJ) cluster melting: MD and Monte Carlo (MC) in the microcanonical and canonical ensembles.
4. Thermodynamics of finite systems: solid–liquid coexistence, caloric curves, negative heat capacity, microcanonical temperature.
5. Order parameters and melting criteria: Lindemann/δ, bond-orientational order (Q_l, W_l), common-neighbour analysis (CNA), inherent structures.
6. Energy-landscape theory: global minima, disconnectivity graphs, basin-hopping, isomerisation.
7. Enhanced sampling: J-walking, parallel tempering / replica exchange, histogram reweighting, Wang–Landau.
8. Dynamical-systems views: Lyapunov exponents and chaos at the melting crossover.
9. Size effects, quantum effects and experiments on free clusters.
10. Recent machine-learning (ML) and data-driven order parameters.

### 0.3 How the references were checked

- Candidate works came from domain knowledge of the cluster-melting literature, from the reference trails of the major reviews [23, 77–80], and from targeted web searches for recent (2017–2025) work.
- **Every entry with a DOI in Section 8 was checked against Crossref metadata** (api.crossref.org) on 2026-09-29: title, authors, year, journal, volume and pages. Several details in our initial recollection were wrong and have been corrected. For example, the paper on the influence of saddles on chaos is *Hinde & Berry 1993*, J. Chem. Phys. 99, 2942 [66]; the 1992 Hinde–Berry–Wales paper is titled *"Chaos in small clusters of inert gas atoms"* [65]. Tsai & Jordan's jump-walking paper is J. Chem. Phys. **99**, 6957 (1993) [33], not J. Phys. Chem. 97.
- Two references have no DOI and were not machine-checked: Lindemann (1910) [2] and Wales's *Energy Landscapes* monograph (2003) [62]. Frenkel & Smit (3rd ed., 2023) [13] is cited as a book, because Crossref indexes only its chapters.
- A machine-readable bibliography built from the same Crossref records is in [`references.bib`](references.bib).

### 0.4 Honesty note on the level of reading

The findings in the per-paper reviews come from abstracts, from the papers' standard reception in later reviews [23, 62, 77–80], and from textbook knowledge. **We have not yet re-read every full text.** Statements marked **(verify)** are paraphrases whose exact wording or numbers should be checked against the original before we quote them in the final report or presentation. "Author-stated gaps" are paraphrased, not quoted.

---

## 1. The project in one page

| Item | What the course specifies |
|---|---|
| System | N = 13 particles in free space (no periodic boundaries), pairwise LJ potential V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] |
| Dynamics | Newton's equations, integrated with velocity Verlet |
| Tasks | (1) LJ forces; (2) velocity Verlet; (3) verify energy conservation; (4) timestep convergence study; (5) repeat for increasing initial kinetic energy; (6) increase N |
| Indicator | Relative bond-length fluctuation δ = 2/(N(N−1)) Σ_{i<j} √(⟨r_ij²⟩ − ⟨r_ij⟩²) / ⟨r_ij⟩ |
| Expected physics | No sharp phase transition at N = 13; a *crossover* from solid-like to liquid-like behaviour |

Known reference values from the literature that we can validate against:

- **Global minimum of LJ13**: a Mackay icosahedron with E_min = −44.326801 ε [56, 57]. This is the natural initial condition and a unit test for the energy routine.
- **Landscape size**: LJ13 has on the order of 10³ distinct local minima (≈1,500 reported by Doye, Miller & Wales [58] **(verify exact count)**). The icosahedron sits at the bottom of a single, well-defined funnel, which is why LJ13 melts "cleanly" compared with LJ38 [36, 58].
- **Melting region**: the canonical heat capacity of LJ13 peaks at k_BT/ε ≈ 0.28–0.30, about 34–35 K for argon [33, 34] **(verify)**. In the microcanonical ensemble, which is what our NVE code samples, the crossover appears as a change of slope, or a weak S-bend, in the caloric curve T(E), together with a jump in δ [19, 26].
- **Magnitude of δ**: values of a few per cent in the solid, rising steeply through ~0.1 and levelling off at ~0.2–0.3 in the liquid [16, 19, 23, 46]. The slides say δ ≈ 1 for a liquid. The literature does not support this: see Section 7.

---

## 2. Core papers: detailed reviews

Each entry follows the same template: **(1) citation; (2) problem; (3) methodology; (4) evaluation and ground truth; (5) critical analysis; (6) gaps, split into author-stated and inferred; (7) future work and extensions relevant to us.**

### Theme A: integrators and energy conservation

---

#### A1. Verlet (1967): *Computer "experiments" on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules* [6]

1. **Citation.** L. Verlet, *Phys. Rev.* 159, 98–103 (1967). doi:10.1103/PhysRev.159.98
2. **Problem.** Whether classical MD with an LJ potential reproduces the thermodynamics of a real fluid (argon), and how to integrate the equations of motion efficiently.
3. **Method.** Periodic LJ fluid of 864 particles. Positions are advanced with the central-difference (Störmer–Verlet) scheme r(t+h) = 2r(t) − r(t−h) + h²a(t). A tabulated list of neighbours (the "Verlet list") avoids O(N²) force evaluations. Fully deterministic; no thermostat.
4. **Evaluation.** The equation of state and thermodynamic properties are compared with experimental argon data. Here the ground truth is experiment. Integrator accuracy is judged by how well the total energy is conserved.
5. **Critical analysis.** Velocities are not stepped explicitly: they are recovered from finite differences, which makes the kinetic energy (and so the temperature) less accurate than the positions. This is the historical reason for the velocity form of the algorithm [7]. Periodic bulk results say nothing about free clusters, where surface atoms dominate (12 of the 13 atoms in LJ13 are surface atoms).
6. **Gaps.** *Author-stated:* extending to time-dependent correlation functions, done in later papers of the series. *Inferred:* no theory of *why* energy stays bounded. That came 30 years later with backward error analysis [8].
7. **Extensions for us.** Use this paper as the historical baseline in the report. Neighbour lists are pointless at N = 13 (all 78 pairs interact) but become relevant if we scale to N ≳ 100 (Task 6).

---

#### A2. Swope, Andersen, Berens & Wilson (1982): the velocity-Verlet paper [7]

1. **Citation.** W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, *J. Chem. Phys.* 76, 637–649 (1982). doi:10.1063/1.442716
2. **Problem.** The main goal is computing equilibrium constants for forming *physical clusters* of molecules in a vapour. It is directly relevant to us because it concerns small bound clusters. The paper is best known for introducing the **velocity-Verlet** formulation.
3. **Method.** The velocity-Verlet update: v(t+h/2) = v(t) + (h/2)a(t); r(t+h) = r(t) + h·v(t+h/2); recompute forces; v(t+h) = v(t+h/2) + (h/2)a(t+h). This is algebraically equivalent to position Verlet but carries synchronised positions and velocities with one force evaluation per step. Cluster formation is quantified with a cluster definition and a statistical-mechanical formulation.
4. **Evaluation.** Equivalence to the Verlet trajectory, energy conservation, and statistical estimates of cluster equilibrium constants.
5. **Critical analysis.** Cluster statistics depend on the chosen geometric definition of "a cluster". This is the same definition problem we face with evaporation at high energy (Section 7).
6. **Gaps.** *Inferred:* no error analysis of the energy in the velocity form; no guidance on timestep selection for strongly anharmonic (hot, near-evaporating) clusters.
7. **Extensions for us.** This is *the* algorithm we implement (Task 2). Cite it as the primary reference, with [8] as the theoretical justification.

---

#### A3. Hairer, Lubich & Wanner (2003): *Geometric numerical integration illustrated by the Störmer–Verlet method* [8]

1. **Citation.** E. Hairer, C. Lubich, G. Wanner, *Acta Numerica* 12, 399–450 (2003). doi:10.1017/S0962492902000144
2. **Problem.** Why the second-order Verlet scheme outperforms higher-order general-purpose integrators (for example RK4) in long-time Hamiltonian simulations.
3. **Method.** Mathematical analysis. Störmer–Verlet is symplectic and time-reversible. Backward error analysis shows that the numerical trajectory is the *exact* flow of a modified ("shadow") Hamiltonian H̃ = H + h²H₂ + h⁴H₄ + … .
4. **Evaluation.** Theorems plus numerical illustrations. The ground truth is exact conservation laws: the energy error stays **O(h²) and bounded, with no secular drift, over exponentially long times**, provided the potential is smooth and h is below a stability threshold.
5. **Critical analysis.** The guarantees assume a smooth H and exact arithmetic. At high energies LJ13 has close encounters in the steep r⁻¹² wall, so the effective stiffness and the stable h depend on the energy. The bounds are asymptotic and give no numerical constant for a given system.
6. **Gaps.** *Author-stated (paraphrase):* variable step sizes destroy symplecticity unless handled specially. *Inferred:* no practical recipe for choosing h for a specific molecular system. That is exactly our Task 4.
7. **Extensions for us.** This paper gives the *prediction* our timestep study should test: the amplitude of the energy fluctuation ΔE(h) should scale as **h²**, a slope of 2 on a log–log plot, with no drift. RK4 would instead drift, typically dissipating energy **(verify for our system)**. A short Verlet-vs-RK4 comparison would make a strong figure for the Stage 3 presentation.

---

#### A4. Engle, Skeel & Drees (2005): *Monitoring energy drift with shadow Hamiltonians* [9]; with Skeel (2009) [10] and Toxvaerd et al. (2012) [11]

1. **Citations.** R. D. Engle, R. D. Skeel, M. Drees, *J. Comput. Phys.* 206, 432–452 (2005), doi:10.1016/j.jcp.2004.12.009. R. D. Skeel, *SIAM J. Sci. Comput.* 31, 1363–1378 (2009), doi:10.1137/070683660. S. Toxvaerd, O. J. Heilmann, J. C. Dyre, *J. Chem. Phys.* 136, 224106 (2012), doi:10.1063/1.4726728.
2. **Problem.** Total-energy fluctuations of size O(h²) hide small systematic drifts. How can we detect loss of accuracy reliably, and why does MD "work" at all given that trajectories are chaotic?
3. **Method.** Cheaply computable approximations of the shadow Hamiltonian H̃ from quantities along the trajectory [9]. Because H̃ fluctuates far less than H, drift becomes visible much earlier. Skeel [10] argues that MD is reliable for *statistical* properties, not individual trajectories, via shadowing and backward-error arguments. Toxvaerd et al. [11] examine energy conservation for LJ systems and attribute much of the apparent drift to non-smooth potential truncation and round-off **(verify)**.
4. **Evaluation.** Numerical experiments on molecular systems, comparing the drift in H and in H̃.
5. **Critical analysis.** The shadow-Hamiltonian estimators require stored history (several consecutive steps), which adds bookkeeping. The results for periodic liquids with cut-offs do not transfer fully to our untruncated free cluster. That is actually good news for us: with no cut-off, the smoothness assumption of [8] holds exactly.
6. **Gaps.** *Inferred:* no systematic study of how integrator error propagates into *derived* thermodynamic quantities such as the caloric curve, the melting energy or δ. This is a real, testable gap (Direction D2 in Section 6).
7. **Extensions for us.** Report the fluctuation σ_E(h), the drift slope dE/dt(h) and, optionally, H̃. Test whether the *melting indicators* themselves (T(E), δ(E)) converge as h → 0 at the expected O(h²) rate.

---

### Theme B: simulations of LJ-cluster melting

---

#### B1. Briant & Burton (1975): *Molecular dynamics study of the structure and thermodynamic properties of argon microclusters* [16]

1. **Citation.** C. L. Briant, J. J. Burton, *J. Chem. Phys.* 63, 2045–2058 (1975). doi:10.1063/1.431542
2. **Problem.** Do small argon clusters have distinct solid-like and liquid-like states, and how do their thermodynamics differ from the bulk?
3. **Method.** Constant-energy MD of free argon clusters, including the 13- and 55-atom icosahedral sizes. Measured quantities: caloric curves (temperature against total energy), structure, and the rms bond-length fluctuation, which is essentially the δ in our slides.
4. **Evaluation.** Qualitative: a change in the caloric-curve slope and in bond-length fluctuations signals melting. Results are compared with bulk argon. There is no independent ground truth for "melted" in a finite system.
5. **Critical analysis.** Trajectories that were short by modern standards limit sampling of the rare solid ↔ liquid switching. Temperature was defined from the kinetic energy without the careful finite-system corrections introduced later [31] **(verify their dof convention)**.
6. **Gaps.** *Inferred:* no account of coexistence or of the observation-time dependence of δ. Both were central to the Berry group's later work [19, 23].
7. **Extensions for us.** This is the prototype for our Task 5 protocol. Reproducing its qualitative caloric-curve kink for LJ13 is a sensible first milestone.

---

#### B2. Jellinek, Beck & Berry (1986): *Solid–liquid phase changes in simulated isoenergetic Ar13* [19]

1. **Citation.** J. Jellinek, T. L. Beck, R. S. Berry, *J. Chem. Phys.* 84, 2783–2794 (1986). doi:10.1063/1.450303
2. **Problem.** How does Ar13 change from solid to liquid at *constant energy*, which is exactly our setting? Is there a coexistence regime?
3. **Method.** Long microcanonical MD at a sequence of total energies. Diagnostics: the caloric curve T(E); the rms bond-length fluctuation δ; **distributions of short-time-averaged kinetic energy**; and visual and structural analysis.
4. **Evaluation.** Three regimes are identified: a solid at low E, a liquid at high E, and an intermediate range where the cluster **switches back and forth** between solid-like and liquid-like forms on picosecond-to-nanosecond timescales. The switching shows up as a *bimodal* short-time temperature distribution. The ground truth for "phase" is operational (bimodality), not thermodynamic.
5. **Critical analysis.** (a) δ averaged over the whole run hides coexistence, because it mixes both forms. The paper's own evidence shows that a single δ value is not a sufficient descriptor. (b) What counts as bimodal depends on the averaging window. (c) One system size and one potential. (d) Very high energies risk evaporation, which must be excluded or controlled.
6. **Gaps.** *Author-stated (paraphrase):* coexistence in small clusters differs fundamentally from bulk first-order coexistence and needs a finite-system thermodynamic theory, later provided by [26–28]. *Inferred:* no systematic study of how conclusions depend on the averaging window τ; no uncertainty quantification.
7. **Extensions for us.** **This is the closest match to our project.** Beyond reproducing δ(E), we should also compute **short-time kinetic-energy histograms** at each E and look for bimodality. Doing so turns the vague "crossover" in the slides into a measurable coexistence band.

---

#### B3. Honeycutt & Andersen (1987): *Molecular dynamics study of melting and freezing of small Lennard-Jones clusters* [20]

1. **Citation.** J. D. Honeycutt, H. C. Andersen, *J. Phys. Chem.* 91, 4950–4963 (1987). doi:10.1021/j100303a014
2. **Problem.** The kinetics and structure of melting and freezing in small LJ clusters, including hysteresis (superheating and supercooling).
3. **Method.** MD of LJ clusters heated and cooled through the transition. Introduces **pair (common-neighbour) analysis**: each pair of neighbours is classified by the topology of the neighbours they share, which identifies icosahedral, fcc-like and hcp-like local order.
4. **Evaluation.** Hysteresis loops in energy against temperature; structural fingerprints of the ordered and disordered states. The ground truth is structural (local symmetry), not thermodynamic.
5. **Critical analysis.** Common-neighbour classification needs a distance cut-off, and results are sensitive to it. Hysteresis depends on heating and cooling rates, so the result is kinetic rather than equilibrium.
6. **Gaps.** *Inferred:* no direct link between the structural classifier and thermodynamic signatures such as a heat-capacity peak.
7. **Extensions for us.** CNA (or its adaptive variant [49]) gives a *structural* order parameter that is independent of δ, for example the fraction of 1551 pairs (icosahedral). Cross-checking δ against CNA answers "is δ measuring melting or just large vibrations?"

---

#### B4. Davis, Jellinek & Berry (1987) [21] and Beck, Jellinek & Berry (1987) [22]: isothermal Ar13 and "slush"

1. **Citations.** H. L. Davis, J. Jellinek, R. S. Berry, *J. Chem. Phys.* 86, 6456–6464 (1987), doi:10.1063/1.452436. T. L. Beck, J. Jellinek, R. S. Berry, *J. Chem. Phys.* 87, 545–554 (1987), doi:10.1063/1.453602.
2. **Problem.** Does coexistence survive in the canonical (constant-T) ensemble? How does melting behaviour depend on cluster size, magic versus non-magic?
3. **Method.** Constant-temperature MD of Ar13 [21]. MD across several rare-gas cluster sizes [22].
4. **Evaluation.** Bimodal distributions in the canonical ensemble for Ar13 [21]. Magic-number clusters such as 13 show well-separated solid and liquid forms, whereas some non-magic sizes show an intermediate, soft **"slush"** state without clear coexistence [22].
5. **Critical analysis.** Thermostat artefacts (the choice of thermostat alters the dynamics) and small sample sizes.
6. **Gaps.** *Inferred:* the "slush" is a qualitative category without a quantitative order parameter.
7. **Extensions for us.** For Task 6 (increase N), a comparison of **N = 13 (magic) against N = 14 or 12 (non-magic)** is cheap and tests a well-documented qualitative prediction.

---

#### B5. Berry, Beck, Davis & Jellinek (1988): *Solid–liquid phase behavior in microclusters* [23]

1. **Citation.** R. S. Berry, T. L. Beck, H. L. Davis, J. Jellinek, *Adv. Chem. Phys.* 70 (Part 2), 75–138 (1988). doi:10.1002/9780470122693.ch3
2. **Problem.** A review and synthesis: can "phases" be defined for systems of ~10–100 atoms?
3. **Method.** A synthesis of MD, MC and theory. Phases are defined as *locally stable* regions of phase space in which the system stays for times much longer than vibrational periods.
4. **Evaluation.** Compiles caloric curves, δ, and short-time temperature distributions for several clusters.
5. **Critical analysis.** The phase definition is inherently tied to *observation time*, which is conceptually clean but makes every indicator time-scale dependent.
6. **Gaps.** *Author-stated (paraphrase):* a statistical-mechanical foundation for coexistence in finite systems. *Inferred:* no consensus threshold for δ; the commonly quoted ~0.1 is inherited from bulk Lindemann practice [2].
7. **Extensions for us.** Use this review as the conceptual reference for the report's "what does melting mean for N = 13?" section.

---

#### B6. Wales & Berry (1990) [24, 25]: melting, freezing and spinodals of small argon clusters

1. **Citations.** D. J. Wales, R. S. Berry, *J. Chem. Phys.* 92, 4283–4295 (1990), doi:10.1063/1.457788; and *J. Chem. Phys.* 92, 4473–4482 (1990), doi:10.1063/1.457758.
2. **Problem.** Establish the *melting* and *freezing* limits, the bounds of the coexistence band, in small argon clusters, and relate them to spinodals.
3. **Method.** MD and statistical models based on densities of states of the solid-like and liquid-like forms.
4. **Evaluation.** The coexistence range in energy or temperature is bounded by the stability limits of each form.
5. **Critical analysis.** Relies on a two-state (solid/liquid) picture, which becomes questionable for non-magic sizes [22] and larger clusters [27].
6. **Gaps.** *Inferred:* the two-state picture does not extend to multi-state landscapes such as LJ38 [36].
7. **Extensions for us.** Suggests measuring the *lower* and *upper* energies at which δ starts and stops being bimodal, rather than a single "melting energy".

---

### Theme C: thermodynamics of finite systems

---

#### C1. Labastie & Whetten (1990): *Statistical thermodynamics of the cluster solid–liquid transition* [26]

1. **Citation.** P. Labastie, R. L. Whetten, *Phys. Rev. Lett.* 65, 1567–1570 (1990). doi:10.1103/PhysRevLett.65.1567
2. **Problem.** A statistical-mechanical explanation of cluster coexistence, and of how canonical and microcanonical descriptions differ for small N.
3. **Method.** MC sampling of LJ clusters and extraction of the density of states Ω(E). Canonical and microcanonical caloric curves are obtained from Ω(E).
4. **Evaluation.** Bimodal canonical energy distributions, and a microcanonical caloric curve with an **S-bend (a "van der Waals loop")**, that is, a region of *negative* microcanonical heat capacity **(verify for which N the S-bend is predicted)**. The ground truth is Ω(E), exact up to sampling error.
5. **Critical analysis.** Depends on MC reaching equilibrium across the barrier (quasi-ergodicity) [33]. Whether the S-bend is present for LJ13 specifically is subtle: it is small and sensitive to the dof convention and to how evaporation is constrained.
6. **Gaps.** *Inferred:* no direct dynamical (MD) test of the S-bend at the time; experimental confirmation came only later, for sodium [73].
7. **Extensions for us.** Our NVE runs give T(E) directly. A careful T(E) with error bars around the crossover is a direct test of whether LJ13 shows an S-bend. This is a sharp, falsifiable question (Direction D4).

---

#### C2. Kunz & Berry (1993) [27] and Wales & Berry (1994) [28]: coexistence in finite systems

1. **Citations.** R. E. Kunz, R. S. Berry, *Phys. Rev. Lett.* 71, 3987–3990 (1993), doi:10.1103/PhysRevLett.71.3987. D. J. Wales, R. S. Berry, *Phys. Rev. Lett.* 73, 2875–2878 (1994), doi:10.1103/PhysRevLett.73.2875.
2. **Problem.** Under what conditions can a finite system show two or more phases in dynamic equilibrium, and how does this relate to the Gibbs phase rule?
3. **Method.** Theory based on free-energy functions of an order parameter with several local minima. Kunz & Berry find *multiple* coexisting forms (for example surface-melted states) in larger clusters.
4. **Evaluation.** Consistency with simulated distributions and caloric curves.
5. **Critical analysis.** Requires choosing an order parameter; the conclusions are only as good as that choice.
6. **Gaps.** *Inferred:* no principled, automatic way to *choose* the order parameter. This gap reappears in the ML literature [82, 83].
7. **Extensions for us.** Motivates building a Landau-type free-energy profile F(δ_τ) = −k_BT ln P(δ_τ) from the short-window δ distribution at each energy (Direction D1).

---

#### C3. Lynden-Bell & Wales (1994) [29] and Doye & Wales (1995) [30]: free-energy barriers and order-parameter coexistence

1. **Citations.** R. M. Lynden-Bell, D. J. Wales, *J. Chem. Phys.* 101, 1460–1476 (1994), doi:10.1063/1.467771. J. P. K. Doye, D. J. Wales, *J. Chem. Phys.* 102, 9673–9688 (1995), doi:10.1063/1.468786.
2. **Problem.** Quantify the free-energy barrier between solid-like and liquid-like forms, and how it scales with cluster size.
3. **Method.** Landau free energies as functions of an order parameter (potential energy or bond-orientational order Q₄/Q₆ [47]), computed from simulations and from the superposition (harmonic-basin) approximation.
4. **Evaluation.** Double-well free-energy profiles in the coexistence range; barriers that grow with N.
5. **Critical analysis.** Order-parameter dependence: different order parameters give different barriers, and none is uniquely "correct".
6. **Gaps.** *Inferred:* no comparison of *which* order parameter best separates the phases for LJ13 under NVE.
7. **Extensions for us.** Compute δ_τ, Q₆ and potential energy on the same trajectory and compare their free-energy profiles (Direction D3).

---

#### C4. Pearson, Halicioglu & Tiller (1985): *Laplace-transform technique for … the classical microcanonical ensemble* [31]

1. **Citation.** E. M. Pearson, T. Halicioglu, W. A. Tiller, *Phys. Rev. A* 32, 3030–3039 (1985). doi:10.1103/PhysRevA.32.3030
2. **Problem.** Correct expressions for temperature, heat capacity and related quantities in the *microcanonical* ensemble of a finite system with conserved total momentum and angular momentum.
3. **Method.** Laplace-transform derivation giving, for example, T in terms of ⟨K⟩ and C_V in terms of ⟨K⟩ and ⟨K⁻¹⟩, with the number of degrees of freedom reduced by the conserved quantities.
4. **Evaluation.** Analytical, checked against simulations.
5. **Critical analysis.** Assumes ergodicity within the energy shell, which fails for the solid at very low E on short runs.
6. **Gaps.** *Inferred:* rarely applied carefully in student-level or introductory MD, which leads to systematically biased T for small N.
7. **Extensions for us.** **Directly implementable.** Zero the total linear and angular momentum, use f = 3N − 6 = 33 degrees of freedom for T, and compute the microcanonical C_V from kinetic-energy fluctuations. This produces a heat-capacity curve *without* any thermostat.

---

#### C5. Bixon & Jortner (1989): *Energetic and thermodynamic size effects in molecular clusters* [32]

1. **Citation.** M. Bixon, J. Jortner, *J. Chem. Phys.* 91, 1631–1642 (1989). doi:10.1063/1.457123
2. **Problem.** How the width of the coexistence region and the melting temperature scale with cluster size.
3. **Method.** A two-state thermodynamic model (solid ↔ liquid) with size-dependent energies and entropies.
4. **Evaluation.** Comparison with simulation data for rare-gas clusters.
5. **Critical analysis.** A two-state model; it does not capture structural transitions or surface melting.
6. **Gaps.** *Inferred:* model parameters are fitted, not predicted.
7. **Extensions for us.** Offers a simple analytical fit for our δ(E) or T(E) data, giving a "melting energy" and "transition width" with uncertainties.

---

### Theme D: sampling methods and heat capacities

---

#### D1. Tsai & Jordan (1993): *Use of the histogram and jump-walking methods for overcoming slow barrier crossing behavior in Monte Carlo simulations: applications to the phase transitions in the (Ar)13 and (H2O)8 clusters* [33]

1. **Citation.** C. J. Tsai, K. D. Jordan, *J. Chem. Phys.* 99, 6957–6970 (1993). doi:10.1063/1.465442
2. **Problem.** Standard Metropolis MC equilibrates poorly near the transition because solid ↔ liquid barriers are rarely crossed (quasi-ergodicity), which gives wrong heat capacities.
3. **Method.** J-walking: occasional jumps to configurations sampled at a higher temperature. Histogram reweighting to interpolate between temperatures.
4. **Evaluation.** Converged canonical C_V(T) for Ar13 with a well-defined peak **(verify: peak near 34–35 K)**. The ground truth is the converged partition-function estimate.
5. **Critical analysis.** J-walking needs a stored high-T distribution and breaks detailed balance unless implemented carefully; parallel tempering later replaced it [36, 39].
6. **Gaps.** *Inferred:* canonical results cannot be compared directly with the microcanonical MD caloric curves without an ensemble transformation.
7. **Extensions for us.** **The benchmark curve.** Our microcanonical C_V (via [31]) or T(E) can be converted to the canonical ensemble, or compared qualitatively, against this reference.

---

#### D2. Frantz (1995): *Magic numbers for classical Lennard-Jones cluster heat capacities* [34]

1. **Citation.** D. D. Frantz, *J. Chem. Phys.* 102, 3747–3768 (1995). doi:10.1063/1.468557
2. **Problem.** How the melting heat-capacity peak depends on N for small LJ clusters.
3. **Method.** J-walking MC of heat capacities for a range of small N.
4. **Evaluation.** Sizes with closed-shell or highly symmetric structures, notably N = 13 and 19, show pronounced and sharp C_V peaks ("magic numbers"). Other sizes have weaker, broader features **(verify size range)**.
5. **Critical analysis.** Classical statistics only (quantum effects matter for Ne; see [68, 69]). Canonical ensemble.
6. **Gaps.** *Inferred:* no systematic dynamical (MD) counterpart; microcanonical signatures for non-magic sizes are less well documented.
7. **Extensions for us.** Provides predictions to test for Task 6. Our microcanonical crossover should be sharpest at N = 13 and 19.

---

#### D3. Neirotti, Calvo, Freeman & Doll (2000): *Phase changes in 38-atom Lennard-Jones clusters. I. A parallel tempering study in the canonical ensemble* [36]

1. **Citation.** J. P. Neirotti, F. Calvo, D. L. Freeman, J. D. Doll, *J. Chem. Phys.* 112, 10340–10349 (2000). doi:10.1063/1.481671
2. **Problem.** LJ38 has a double-funnel landscape (fcc truncated octahedron against icosahedral structures) [58]. Is there a solid–solid transition before melting?
3. **Method.** Parallel tempering (replica exchange) MC over a temperature ladder.
4. **Evaluation.** A low-temperature C_V feature (the solid–solid transition) well below the melting peak **(verify values)**.
5. **Critical analysis.** Even parallel tempering struggles with the LJ38 double funnel. Results depend on the number of replicas and the run length.
6. **Gaps.** *Inferred:* δ cannot distinguish a solid–solid transition from partial melting, because both raise δ.
7. **Extensions for us.** If we "increase the number of particles" to 38, a single δ(E) will be *misleading*. This makes the case for structural indicators [47, 49].

---

### Theme E: order parameters and melting criteria

---

#### E1. Lindemann (1910) [2] and Zhou, Karplus, Ball & Berry (2002): *The distance fluctuation criterion for melting* [46]

1. **Citations.** F. A. Lindemann, *Phys. Z.* 11, 609–612 (1910) (no DOI). Y. Zhou, M. Karplus, K. D. Ball, R. S. Berry, *J. Chem. Phys.* 116, 2323–2329 (2002), doi:10.1063/1.1426419.
2. **Problem.** Lindemann proposed that a crystal melts when the rms vibration amplitude reaches a fixed fraction (~0.1) of the interatomic spacing. For finite systems the *pairwise distance-fluctuation* version δ (our slide formula) is used, since there is no lattice. Zhou et al. ask how reliable δ is as a melting criterion across models.
3. **Method.** Zhou et al. compare δ with thermodynamic signatures (heat capacity) for clusters and homopolymers modelled with square-well and Morse potentials of different ranges **(verify details)**.
4. **Evaluation.** δ tracks the thermodynamic transition qualitatively, but its value at the transition and the sharpness of its jump depend on the potential and the system **(verify exact conclusions)**.
5. **Critical analysis, directly relevant to us.**
   - **No universal threshold.** The "δ ≈ 0.1" rule is empirical. Sarkar, Jana & Bagchi [50] show that the universal Lindemann criterion breaks down even for bulk polydisperse LJ solids.
   - **Time-window dependence.** In the liquid, permutational isomerisation makes ⟨r_ij⟩ drift, so δ grows with averaging time until every pair has sampled the whole cluster. δ is therefore not a state function unless τ is specified.
   - **Evaporation.** If one atom leaves, its r_ij grows without bound and δ diverges. This is *not* melting.
   - **Not structural.** Large-amplitude anharmonic vibrations, or isomerisation between two *solid* forms, also raise δ [36, 84].
6. **Gaps.** *Author-stated (paraphrase):* need for a criterion that does not depend on the model. *Inferred:* no standard protocol for choosing τ, handling evaporation, or reporting uncertainty on δ.
7. **Extensions for us.** Always report **δ together with the averaging window τ**. Detect evaporation explicitly. Pair δ with at least one structural parameter (Q₆ or CNA).

---

#### E2. Steinhardt, Nelson & Ronchetti (1983): *Bond-orientational order in liquids and glasses* [47]; Lechner & Dellago (2008) [48]

1. **Citations.** P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* 28, 784–805 (1983), doi:10.1103/PhysRevB.28.784. W. Lechner, C. Dellago, *J. Chem. Phys.* 129, 114707 (2008), doi:10.1063/1.2977970.
2. **Problem.** Rotationally invariant measures of local orientational order that distinguish crystal, icosahedral and liquid-like environments.
3. **Method.** Bond directions are expanded in spherical harmonics Y_lm, then the invariants Q_l (and W_l) are formed. Icosahedral order has a distinctive Q₆ and W₆ signature. Lechner & Dellago average over neighbours to sharpen the distinction.
4. **Evaluation.** Separation of known reference structures (fcc, hcp, bcc, icosahedral, liquid) in simulations of LJ systems.
5. **Critical analysis.** Needs a neighbour definition (cut-off); for a 13-atom cluster, statistics per atom are poor. Surface atoms dominate small clusters and have incomplete neighbour shells.
6. **Gaps.** *Inferred:* global Q₆ for a 13-atom cluster is noisy. It is better used on the *central* atom (whose 12 neighbours form the icosahedron) or on quenched inherent structures [54].
7. **Extensions for us.** A cheap structural order parameter: Q₆ of the central atom, or of whichever atom has the most neighbours. It is expected to fall sharply when the icosahedron breaks up.

---

#### E3. Stillinger & Weber (1982, 1984): inherent structures [54, 55]

1. **Citations.** F. H. Stillinger, T. A. Weber, *Phys. Rev. A* 25, 978–989 (1982), doi:10.1103/PhysRevA.25.978; *Science* 225, 983–989 (1984), doi:10.1126/science.225.4666.983.
2. **Problem.** Separate vibrational motion from structural rearrangement by mapping each configuration to the local minimum it drains into, its "inherent structure".
3. **Method.** Periodically quench (steepest-descent or conjugate-gradient minimisation) snapshots of the trajectory. Record the energy and identity of the resulting minima.
4. **Evaluation.** In the solid, quenches return the global minimum. In the liquid, many higher-energy minima appear.
5. **Critical analysis.** Each quench costs an energy minimisation; frequent quenching is expensive for large N (trivial for N = 13).
6. **Gaps.** *Inferred:* relatively rarely combined with δ in cluster-melting studies to test *why* δ rises.
7. **Extensions for us.** **Excellent, cheap, threshold-free diagnostic for N = 13.** Define the "liquid fraction" as the fraction of quenches that do not return E_min = −44.3268 ε, after identifying permutational isomers. This gives a structural ground truth against which δ can be calibrated (Directions D3 and D6).

---

### Theme F: energy landscapes

---

#### F1. Wales & Doye (1997) [57] and Doye, Miller & Wales (1999) [58]: global minima and landscapes of LJ clusters

1. **Citations.** D. J. Wales, J. P. K. Doye, *J. Phys. Chem. A* 101, 5111–5116 (1997), doi:10.1021/jp970984n. J. P. K. Doye, M. A. Miller, D. J. Wales, *J. Chem. Phys.* 111, 8417–8428 (1999), doi:10.1063/1.480217.
2. **Problem.** Locate the global minima of LJ_N up to N = 110 [57], and describe how the organisation of the landscape (funnels) changes with N [58].
3. **Method.** Basin-hopping global optimisation [57]. Enumeration of minima and transition states, and disconnectivity graphs [58].
4. **Evaluation.** Lowest-known energies, now the standard Cambridge Cluster Database benchmark. Graphs show single-funnel landscapes for magic sizes such as LJ13 and multi-funnel ones such as LJ38.
5. **Critical analysis.** Global minima are "putative": optimality is not proven for larger N. Landscape databases are incomplete for large N.
6. **Gaps.** *Inferred:* the link between landscape topology and the *microcanonical* melting signatures observed in MD is mostly qualitative.
7. **Extensions for us.** Use E_min(LJ13) = −44.326801 ε as a unit test. The single-funnel picture explains why LJ13 is the textbook "clean" crossover.

---

### Theme G: chaos and dynamics

---

#### G1. Hinde, Berry & Wales (1992) [65]; Hinde & Berry (1993) [66]; Nayak, Ramaswamy & Chakravarty (1995) [67]

1. **Citations.** R. J. Hinde, R. S. Berry, D. J. Wales, *J. Chem. Phys.* 96, 1376–1390 (1992), doi:10.1063/1.462173. R. J. Hinde, R. S. Berry, *J. Chem. Phys.* 99, 2942–2963 (1993), doi:10.1063/1.465201. S. K. Nayak, R. Ramaswamy, C. Chakravarty, *Phys. Rev. E* 51, 3376–3380 (1995), doi:10.1103/PhysRevE.51.3376.
2. **Problem.** How chaotic is cluster dynamics, and does chaoticity change across the solid–liquid crossover?
3. **Method.** Lyapunov exponents from tangent-space (linearised) dynamics or from the divergence of nearby trajectories. Local Lyapunov exponents are related to regions near saddles of the potential energy surface.
4. **Evaluation.** The maximal Lyapunov exponent changes characteristically with energy through the melting region **(verify: [67] reports a feature near the transition)**.
5. **Critical analysis.** Lyapunov exponents need long runs and careful renormalisation. They are sensitive to the integrator: a symplectic scheme preserves phase-space volume, but finite h still perturbs the exponents.
6. **Gaps.** *Inferred:* rarely compared quantitatively with δ and C_V *on the same trajectories*.
7. **Extensions for us.** An optional "dynamical" order parameter. It also explains, for the report, *why* two runs with almost identical initial conditions diverge, which is relevant when we discuss reproducibility and energy conservation versus trajectory accuracy [10].

---

### Theme H: quantum and experimental reality checks

---

#### H1. Calvo, Doye & Wales (2001): *Quantum partition functions from classical distributions: application to rare-gas clusters* [68]

1. **Citation.** F. Calvo, J. P. K. Doye, D. J. Wales, *J. Chem. Phys.* 114, 7312–7329 (2001). doi:10.1063/1.1359768
2. **Problem.** How large are quantum effects on the thermodynamics (including melting) of rare-gas clusters such as Ne13 and Ar13?
3. **Method.** Quantum partition functions reconstructed from classically sampled distributions via harmonic or anharmonic superposition.
4. **Evaluation.** Quantum effects noticeably shift the melting behaviour of Ne clusters and matter much less for Ar and heavier gases **(verify magnitude)**. Compared with path-integral results.
5. **Critical analysis.** Approximations in the superposition treatment.
6. **Gaps.** *Inferred:* our purely classical model is appropriate for Ar13 and Xe13 but not for Ne13. This should be stated as a limitation.
7. **Extensions for us.** State in the report which element the reduced units map onto (argon is the standard choice) and why classical mechanics suffices.

---

#### H2. Schmidt, Haberland et al. (1997, 1998, 2001): melting and negative heat capacity of free Na clusters [71–73]

1. **Citations.** *Phys. Rev. Lett.* 79, 99 (1997), doi:10.1103/PhysRevLett.79.99. *Nature* 393, 238 (1998), doi:10.1038/30415. *Phys. Rev. Lett.* 86, 1191 (2001), doi:10.1103/PhysRevLett.86.1191.
2. **Problem.** Experimental measurement of caloric curves of size-selected free clusters.
3. **Method.** Photofragmentation of mass-selected, thermalised Na_N⁺ clusters; the number of evaporated atoms maps to internal energy.
4. **Evaluation.** Melting points depressed relative to bulk, with *irregular* size dependence [72]. A **negative microcanonical heat capacity** for Na147⁺ [73], the experimental counterpart of the S-bend in [26].
5. **Critical analysis.** Metallic (Na) clusters, not rare-gas ones; the potential and the physics differ.
6. **Gaps.** *Inferred:* no direct experiment on the caloric curve of Ar13.
7. **Extensions for us.** Motivation and context for the report's introduction: finite-size melting is measurable and non-trivial, and it is not merely a simulation curiosity.

---

### Theme I: recent data-driven directions

---

#### I1. Smidt, Geiger & Miller (2021) [82]; Takahashi (2023) [83]; Zeng, Hsu & Wu (2022) [52]

1. **Citations.** T. E. Smidt, M. Geiger, B. K. Miller, *Phys. Rev. Research* 3, L012002 (2021), doi:10.1103/PhysRevResearch.3.L012002. K. Z. Takahashi, *Phys. Chem. Chem. Phys.* 25, 658–672 (2023), doi:10.1039/D2CP03696G. S.-Y. Zeng, C.-H. Hsu, T.-M. Wu, *J. Phys. Chem. A* 126, 2018–2030 (2022), doi:10.1021/acs.jpca.1c09527.
2. **Problem.** Choosing order parameters automatically instead of by hand.
3. **Method.** Euclidean-equivariant neural networks that learn symmetry-breaking order parameters [82]. ML-based *selection* from a large library of local order parameters [83]. Systematic classification of solid-like LJ clusters with Q_l and W_l [52].
4. **Evaluation.** Separation of labelled phases or structures, via classification accuracy on simulations with known labels.
5. **Critical analysis.** Labels come from the simulation protocol (for example "this run was at high T"), which is circular for melting. Supervised approaches need a ground truth that finite clusters do not have.
6. **Gaps.** *Inferred:* not yet applied to the *microcanonical crossover* of LJ13, where the absence of a sharp transition makes the labels ambiguous. Unsupervised or self-consistent approaches are the natural fit.
7. **Extensions for us.** An optional stretch goal: unsupervised clustering (for example PCA or a Gaussian mixture) of sorted-distance fingerprints from our trajectories. It should discover the solid/liquid split *without* a δ threshold (Direction D8).

---

#### I2. Hoffenberg, Khrabry, Barsukov, Kaganovich & Graves (2025): *Size-dependent second-order-like phase transitions in Fe nanocluster melting from low-temperature structural isomerization* [84]

1. **Citation.** L. E. S. Hoffenberg, A. Khrabry, Y. Barsukov, I. D. Kaganovich, D. B. Graves, *J. Chem. Phys.* 162, 134305 (2025). doi:10.1063/5.0236122
2. **Problem.** In some metal nanoclusters, "melting" indicators rise gradually from low temperature because of structural isomerisation, not because of a sharp first-order-like jump.
3. **Method.** MD of Fe clusters across sizes, analysed with Lindemann-type and structural measures **(verify details)**.
4. **Evaluation.** A size-dependent change between sharp and gradual ("second-order-like") behaviour.
5. **Critical analysis.** An Fe-specific interatomic potential; how far the results carry over to LJ is unclear.
6. **Gaps.** *Inferred:* reinforces that δ conflates isomerisation with melting. A threshold-based melting temperature is therefore ill-defined when isomerisation starts early.
7. **Extensions for us.** Supports separating "isomerisation onset" from "melting" in our LJ analysis using inherent structures [54].

---

## 3. Supporting literature: compact reviews

Each row gives the problem, the approach, how it is validated, its main limitation, and how it helps our project.

| Ref. | Work | Problem / approach | Validation | Key limitation | Use for our project |
|---|---|---|---|---|---|
| [1] | Jones 1924 | Derives the (12-6 family) intermolecular potential from gas equation-of-state data | Fit to experimental virial data | Empirical form; r⁻¹² repulsion has no physical basis | Primary citation for the potential |
| [3] | Pawlow 1909 | Theory of melting-point depression with particle size via surface energy | Thermodynamic argument | Continuum theory; fails at N ~ 10 | Historical root of size-dependent melting |
| [4] | Alder & Wainwright 1959 | First MD method (hard spheres) | Comparison with theory | Hard spheres, not continuous potentials | Historical context |
| [5] | Rahman 1964 | First MD of liquid argon with a continuous (LJ) potential | Comparison with neutron data | Periodic bulk | Historical context; argon parameters |
| [12–15] | Allen & Tildesley; Frenkel & Smit; Leimkuhler & Matthews; Tuckerman | Standard MD/MC textbooks: integrators, ensembles, error analysis | n/a | General, not cluster-specific | Implementation reference, reduced units, dof counting |
| [17, 18] | Etters & Kaelberer 1975, 1977 | MC thermodynamics and phase transitions of small rare-gas clusters | Energy and structural distributions | Early MC; limited sampling | Early independent evidence of cluster melting |
| [35] | Calvo & Labastie 1995 | Configurational density of states obtained from MD (multiple-histogram) | Consistency with MC | Needs overlapping histograms | Lets our NVE runs yield Ω(E) and hence canonical curves |
| [37] | Mandelshtam, Frantsuzov & Calvo 2006 | Adaptive exchange MC for LJ74–78 structural transitions | C_V features | Larger N, heavy computation | Example of structural vs melting transitions at larger N |
| [38] | Noya & Doye 2006 | LJ309 structural transitions and melting | Free-energy and C_V analysis | Very large-scale sampling | Upper end of "increase N" |
| [39, 40] | Hukushima & Nemoto 1996; Earl & Deem 2005 | Parallel tempering (replica exchange) and its review | Spin glasses; many applications | Replica count grows with system size | Recommended sampler if we add canonical MC |
| [41] | Ferrenberg & Swendsen 1989 | Multiple-histogram reweighting | Ising benchmarks | Needs histogram overlap | Combine runs at many energies into one curve |
| [42] | Wang & Landau 2001 | Flat-histogram estimation of the density of states | Ising/Potts | Convergence tuning in continuous systems | Alternative route to Ω(E) and an S-bend test |
| [43–45] | Nosé 1984; Hoover 1985; Bussi et al. 2007 | Deterministic and stochastic thermostats for canonical MD | Canonical distribution checks | Nosé–Hoover can be non-ergodic for small, stiff systems; the thermostat perturbs dynamics | Needed only if we compare NVE against NVT (Direction D5) |
| [49] | Stukowski 2012 | Review of structure identification (CNA, adaptive CNA, polyhedral template matching) | Crystalline benchmarks | Designed for bulk crystals | Choosing a structural classifier for N ≳ 55 |
| [50] | Sarkar, Jana & Bagchi 2017 | Breakdown of a universal Lindemann criterion in polydisperse LJ solids | MD | Bulk, polydisperse | Evidence against a fixed δ threshold |
| [51] | Guardiola & Navarro 2011 | Lindemann criteria for quantum clusters at low T | Quantum MC | Quantum regime | Shows δ-type criteria need care even in their definition |
| [53] | Hoare & Pal 1971 | Early catalogue of cluster structures and energy surfaces | Enumeration | Small N, limited computers | Historical landscape work |
| [56] | Northby 1987 | Lattice-based search for LJ_N minima, 13 ≤ N ≤ 147 | Energies | Lattice-restricted search | Global-minimum reference values |
| [59] | Doye & Wales 1998 | Why global optimisation is hard: thermodynamic competition between funnels | LJ38 etc. | Specific to multi-funnel sizes | Explains LJ38 difficulty if we scale N |
| [60] | Ball & Berry 1999 | Master-equation dynamics from statistical samples of minima and saddles | Comparison with MD | Needs a landscape database | Theory for isomerisation rates (Direction D6) |
| [61, 63, 62] | Wales 2002, 2018; Wales 2003 book | Discrete path sampling; review of landscape methods; monograph | Many systems | Methodologically heavy | Background for landscape interpretation |
| [64] | Doye & Calvo 2001 | Entropy can change the favoured cluster structure with temperature | LJ clusters | Harmonic/anharmonic approximations | Why structural indicators matter at finite T |
| [69] | Frantsuzov & Mandelshtam 2004 | Variational Gaussian wave-packet quantum thermodynamics of vdW clusters | Comparison with path integrals | Approximate quantum method | Quantifies when classical LJ13 fails (Ne) |
| [70] | Buffat & Borel 1976 | Experimental melting-point depression of Au nanoparticles | Electron diffraction | Large particles (nm) | Experimental context for size effects |
| [74] | Breaux et al. 2003 | Ga clusters stay solid *above* the bulk melting point | Ion calorimetry | Covalent/metallic | Counter-example to monotonic size depression |
| [75] | Cleveland, Luedtke & Landman 1998 | Au cluster melting via icosahedral precursors | MD + structure analysis | Au potential | Structural transitions before melting |
| [76] | Sabo et al. 2004 | Phase changes of binary LJ X₁₃₋ₙYₙ clusters | Parallel tempering | Many parameters | Easy extension: mixed clusters with two σ/ε |
| [77–80] | Baletto & Ferrando 2005; Proykova & Berry 2006; Berry & Smirnov 2009; Aguado & Jarrold 2011 | Reviews of nanocluster structure, phase changes and metal-cluster melting | n/a | Reviews | Entry points for the report introduction |
| [81] | Behler & Parrinello 2007 | Neural-network interatomic potentials | DFT comparison | Needs training data | Route to "real" materials beyond LJ (out of scope) |

---

## 4. Cross-paper synthesis by research direction

### 4.1 Integrator-centred work (symplectic MD, energy conservation) [6–11, 14]

- **Dominant approach.** Velocity Verlet with a fixed timestep; energy conservation is used as the accuracy check.
- **Assumptions.** A smooth potential; h well below the fastest vibrational period; statistical rather than trajectory-level accuracy is what matters.
- **Strengths.** Rigorous theory (a bounded O(h²) energy error with no drift) [8]; cheap; time-reversible.
- **Limitations.** Theory covers the *energy*, not derived quantities such as δ, T(E) or the melting energy. Chaos makes individual trajectories meaningless over long times [10, 65].
- **Unsolved.** How integration error propagates into *melting diagnostics*. Is the melting energy converged at the h we use?

### 4.2 Microcanonical MD of cluster melting [16, 19, 20, 24, 25]

- **Dominant approach.** Run NVE at a ladder of energies. Plot T(E) and δ(E). Look for kinks and bimodality.
- **Assumptions.** Ergodicity on the timescale of the run; temperature defined from ⟨K⟩ with the correct dof; no evaporation.
- **Strengths.** Natural for isolated clusters (experiments are close to microcanonical); shows dynamics and coexistence directly.
- **Limitations.** Poor sampling of rare switching near the crossover; δ depends on averaging time; evaporation at high E.
- **Unsolved.** A standardised, threshold-free, uncertainty-quantified definition of the crossover energy from NVE data alone.

### 4.3 Finite-system thermodynamics and coexistence [26–32]

- **Dominant approach.** Density of states Ω(E) or Landau free energies F(Q), and two-state models.
- **Assumptions.** A suitable order parameter exists; the two-state picture (for magic sizes).
- **Strengths.** Explains coexistence, the S-bend and ensemble inequivalence; confirmed experimentally for Na [73].
- **Limitations.** The results depend on the chosen order parameter. The S-bend is small or unclear for LJ13 and hard to resolve.
- **Unsolved.** An unambiguous yes/no answer on the S-bend (negative heat capacity) of LJ13 from *direct MD*, with error bars.

### 4.4 Order parameters and melting criteria [2, 20, 46–52, 54, 55]

- **Dominant approach.** Lindemann-type δ, bond-orientational Q_l/W_l, common-neighbour analysis, inherent-structure quenching.
- **Assumptions.** A fixed threshold separates solid from liquid (δ); neighbour cut-offs (Q_l, CNA).
- **Strengths.** Cheap and intuitive; δ needs no neighbour definition and works for any N.
- **Limitations.** δ conflates vibrations, isomerisation and evaporation; it depends on the time window; there is no universal threshold [50]. Structural parameters are noisy for N = 13.
- **Unsolved.** A calibrated mapping between δ_τ and an independent structural ground truth, such as the fraction of quenches away from the global minimum.

### 4.5 Enhanced sampling and heat capacities [33–42]

- **Dominant approach.** J-walking, then parallel tempering, plus histogram reweighting; canonical C_V(T) peaks.
- **Assumptions.** Canonical ensemble; converged replica exchange.
- **Strengths.** Converged, reproducible benchmark heat capacities, including for multi-funnel N.
- **Limitations.** Canonical, not microcanonical; MC loses the dynamics; the cost grows with N.
- **Unsolved.** Cheap, direct comparison between NVE dynamics-derived C_V (via [31]) and parallel-tempering benchmarks.

### 4.6 Energy landscapes [53–64]

- **Dominant approach.** Global optimisation, stationary-point databases, disconnectivity graphs, superposition thermodynamics.
- **Assumptions.** The landscape is well sampled; harmonic or anharmonic basin approximations.
- **Strengths.** Explains *why* LJ13 melts cleanly (single funnel) and LJ38 does not.
- **Limitations.** Expensive for large N; mostly equilibrium.
- **Unsolved.** A quantitative bridge between landscape features (barrier heights, numbers of minima) and the δ(E) curve seen in MD.

### 4.7 Dynamical-systems view (chaos) [65–67]

- **Dominant approach.** Maximal Lyapunov exponents as functions of energy.
- **Strengths.** An order-parameter-free, dynamical indicator.
- **Limitations.** Expensive; noisy; integrator-sensitive.
- **Unsolved.** Consistent comparison with δ and C_V on the same data.

### 4.8 Quantum and experimental context [68–75]

- **Dominant approach.** Path integrals or superposition corrections; photofragmentation calorimetry.
- **Unsolved (for LJ).** Experimental caloric curves of rare-gas clusters of ~13 atoms are, to our knowledge, not available with the precision of the Na work [71–73] **(verify)**.

### 4.9 Data-driven order parameters [52, 81–83]

- **Dominant approach.** Supervised or ML-selected local descriptors.
- **Assumption.** Labels are available.
- **Limitation.** A circular labelling problem for finite-size crossovers.
- **Unsolved.** Unsupervised discovery of the solid/liquid distinction in LJ13 NVE trajectories, validated against inherent structures.

### 4.10 Answering the guiding question

*Do existing methods tell us what makes a finite cluster "liquid" and how to measure it without an arbitrary threshold or canonical sampling?* **Only partially.**

- The **conceptual answer** exists: a phase is a long-lived region of phase space [23]; a cluster is "liquid" when it explores many permutational isomers rather than one basin [54, 58].
- The **measurement answer** is fragmented. δ is simple but threshold- and window-dependent. C_V needs ensemble conversions. Structural parameters are noisy at N = 13. Inherent-structure quenching is closest to a ground truth but is rarely used to *calibrate* δ.
- Hardly any work checks whether conclusions about melting in NVE MD are *converged in the timestep*. For a numerical-methods course this is the most natural contribution.

---

## 5. Research-gap map

Scores (★ low to ★★★ high) on five criteria: **(a)** under-studied, **(b)** technically meaningful, **(c)** experimentally (numerically) testable within the course, **(d)** relevant to the project's assessment (numerics plus physics), **(e)** capable of supporting a new contribution at our level.

| # | Gap | Source | (a) | (b) | (c) | (d) | (e) | Priority |
|---|---|---|---|---|---|---|---|---|
| G1 | Propagation of integrator (timestep) error into melting diagnostics (T(E), δ, crossover energy) | Inferred from [8–11] vs [16, 19] | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ | **1** |
| G2 | δ depends on averaging window τ and is reported without it; no uncertainty quantification | Inferred from [19, 23, 46] | ★★★ | ★★★ | ★★★ | ★★★ | ★★ | **2** |
| G3 | Calibration of δ against a structural ground truth (inherent-structure quenches) | Inferred from [46, 54] | ★★ | ★★★ | ★★★ | ★★★ | ★★ | **3** |
| G4 | Microcanonical C_V and S-bend of LJ13 from direct MD with error bars; comparison with canonical benchmarks | Author-stated in part [26, 28]; inferred | ★★ | ★★★ | ★★ | ★★★ | ★★ | 4 |
| G5 | Standard handling of evaporation at high E (constraining sphere, detection, effect on δ) | Inferred from [7, 19] | ★★★ | ★★ | ★★★ | ★★ | ★★ | 5 |
| G6 | Separating isomerisation onset from melting (δ conflates them) | Inferred from [36, 84] | ★★ | ★★★ | ★★★ | ★★ | ★★ | 6 |
| G7 | Size dependence of microcanonical δ(E) for magic vs non-magic N (13 vs 12/14, 19, 38) | Inferred from [22, 34] | ★★ | ★★ | ★★★ | ★★★ | ★★ | 7 |
| G8 | Unsupervised order parameters for finite-size crossovers without labels | Inferred from [82, 83] | ★★★ | ★★ | ★★ | ★ | ★★ | 8 |
| G9 | Lyapunov exponent vs δ vs C_V on the same trajectories | Inferred from [65–67] | ★★ | ★★ | ★★ | ★★ | ★★ | 9 |
| G10 | Quantum corrections for light rare gases (Ne13) | Author-stated in [68, 69] | ★ | ★★ | ★ | ★ | ★ | 10 (out of scope) |

---

## 6. Concrete research directions

Directions D1–D5 fit inside the course project, and D1–D3 are the recommended core. D6–D9 are stretch goals.

### D1. Window-resolved Lindemann index and a threshold-free coexistence band

- **Research question.** Can the solid–liquid coexistence band of LJ13 be identified from NVE data *without* a δ threshold, using the averaging-window dependence of δ?
- **Hypothesis.** For short windows τ (a few vibrational periods), the distribution P(δ_τ) is unimodal and low in the solid, unimodal and high in the liquid, and **bimodal** in the coexistence band. The band's edges are insensitive to τ over a plateau of τ values.
- **Method.** Long NVE runs (≥10⁶ steps) at 30–50 energies. Compute δ over sliding windows τ ∈ {1, 2, 5, 10, 50} ps. Fit Gaussian mixtures to P(δ_τ) and use a bimodality coefficient. Build Landau-type profiles −ln P(δ_τ).
- **Data.** Our own trajectories; LJ13 icosahedron as the initial condition.
- **Evaluation.** Agreement of the band with the kink in T(E), with short-time kinetic-energy bimodality [19], and with the inherent-structure liquid fraction (D3).
- **Contribution.** A reproducible, threshold-free protocol with uncertainties.
- **Difference from existing work.** [16, 19, 46] report δ at one implicit τ; [19] used kinetic-energy bimodality but not a δ_τ sweep.

### D2. Timestep convergence of *physical* observables, not just energy

- **Research question.** How does the timestep h bias the melting diagnostics (T(E), δ(E), crossover energy E_m)?
- **Hypothesis.** All equilibrium averages converge as O(h²), as backward error analysis predicts: trajectories sample the shadow Hamiltonian H̃ ≈ H + O(h²) [8]. There is a threshold h* above which close collisions in the liquid cause energy drift *before* they cause visible errors in the solid, so the high-energy liquid side limits h.
- **Method.** h ∈ {0.0005, 0.001, 0.002, 0.005, 0.01} τ_LJ. Measure σ_E and drift, and T(E), δ(E) and E_m. Log–log convergence plots; Richardson extrapolation to h → 0. Optional: compare against RK4 at equal cost (force evaluations).
- **Data.** Our own trajectories.
- **Evaluation.** Slope-2 convergence; a stable E_m under extrapolation; drift detection via the shadow Hamiltonian [9].
- **Contribution.** Connects the numerical-methods part of the course (Tasks 3–4) to the physics (Task 5) in one quantitative result. This is the strongest candidate for the "solution strategy" presentation (Stage 3).
- **Difference from existing work.** [8–11] analyse energy conservation; [16–25] take h as given. We did not find prior work linking them for cluster melting **(verify with a final targeted search)**.

### D3. Calibrating δ against inherent structures

- **Research question.** What value (or range) of δ_τ corresponds to a given probability that the cluster is out of the icosahedral basin?
- **Hypothesis.** The liquid fraction f_liq(E), the fraction of quenched snapshots not returning to E_min, is a sigmoid in E. The δ jump coincides with f_liq ≈ 0.5. δ values near 0.1 correspond to an intermediate f_liq rather than a sharp boundary.
- **Method.** Every M steps, quench a snapshot (conjugate gradient or FIRE) and record the minimum energy. Identify permutational isomers by energy. Plot f_liq against E next to δ(E) and Q₆ of the central atom.
- **Data.** Our own trajectories; E_min = −44.326801 ε as the reference [56, 57].
- **Evaluation.** Logistic fits; comparison of transition midpoints and widths across indicators.
- **Contribution.** A physically grounded "ground truth" for δ in the most studied cluster.
- **Difference from existing work.** Inherent structures [54, 55] and δ [46] are well established separately; they are rarely used together as a calibration.

### D4. Is there an S-bend in the microcanonical caloric curve of LJ13?

- **Research question.** Does direct NVE MD, with the correct dof and conserved-momentum corrections [31], resolve a negative heat-capacity region for LJ13?
- **Hypothesis.** The S-bend is either absent or smaller than the achievable statistical error for LJ13. It becomes clearer for larger N (for example 55), consistent with growing free-energy barriers [29, 30].
- **Method.** Dense energy grid across the crossover; block-averaged T(E) with standard errors; microcanonical C_V from kinetic-energy fluctuations [31]; optional Ω(E) via multiple histograms [35, 41].
- **Evaluation.** Statistical significance of dT/dE < 0 over any sub-interval; comparison with canonical C_V benchmarks [33, 34].
- **Contribution.** A clear yes, no, or "below resolution" answer with error bars.

### D5. Evaporation-aware protocol for the high-energy regime

- **Research question.** How do evaporation and its treatment (a constraining sphere of radius R_c, or discarding runs) affect δ(E) and T(E) on the liquid side?
- **Hypothesis.** Without confinement, δ diverges above some E because of dissociating atoms, not melting. With a soft wall, results depend weakly on R_c inside a plateau.
- **Method.** Evaporation detection (distance from the centre of mass > R_c, or loss of all neighbours within 2σ). Runs with and without a harmonic wall at several R_c.
- **Evaluation.** Sensitivity of δ(E) and T(E) to R_c.
- **Contribution.** A documented protocol; this treatment is standard but often unreported in introductory work.

### D6. Isomerisation rates as a melting indicator (stretch)

- **Question.** Does the rate of basin changes k(E) show Arrhenius/RRKM-like behaviour below the crossover, and a change of regime at melting?
- **Method.** Time series of inherent structures (from D3); count transitions; fit k(E) to RRKM-like forms [60].
- **Difference.** Links D3 to landscape kinetics [58, 60], and separates isomerisation from melting as motivated by [84].

### D7. Size scan: magic vs non-magic clusters (Task 6)

- **Question.** How do δ(E) and the bimodality band from D1 change for N = 12, 13, 14, 19, 26, 38, 55?
- **Hypothesis.** Sharp bands at magic N (13, 19, 55). A broad, "slushy" crossover at non-magic N [22, 34]. Solid–solid features for N = 38 [36] that δ misreports as melting.
- **Method.** Same pipeline as D1–D3; parallelise over energies (embarrassingly parallel), which also covers the course's sequential-vs-parallel comparison.
- **Data.** Global minima from the Cambridge Cluster Database or from our own basin-hopping [57].

### D8. Unsupervised order parameter from trajectories (stretch)

- **Question.** Can an unsupervised method (PCA or diffusion maps on sorted pair-distance vectors, plus a Gaussian mixture) recover the solid/liquid split without thresholds or labels?
- **Evaluation.** Agreement with f_liq from D3 (adjusted Rand index); stability across N.
- **Difference.** [82, 83] are supervised or selection-based and target bulk phases.

### D9. Lyapunov-exponent cross-check (stretch)

- **Question.** Does the maximal Lyapunov exponent change at the same E_m as δ and f_liq?
- **Method.** Two-trajectory divergence with periodic renormalisation (Benettin-style) on the D1 trajectories.
- **Difference.** Compares dynamical and structural indicators on identical data, which [65–67] did not do together with δ calibration.

---

## 7. Consequences for our implementation

Concrete checklist distilled from the review:

1. **Reduced units.** σ = ε = m = 1, time unit τ = σ√(m/ε). For argon (σ = 3.405 Å, ε/k_B = 119.8 K, m = 39.95 u), τ ≈ 2.16 ps. Report temperatures as k_BT/ε, and optionally in kelvin for argon.
2. **Initial condition.** Mackay icosahedron. **Unit test:** E_pot = −44.326801 ε after local minimisation [56, 57].
3. **Force routine.** Vectorised over all 78 pairs; no cut-off (a free cluster, so no truncation artefacts [11]). Unit test: compare with finite-difference gradients of E_pot.
4. **Conserved quantities.** Remove the centre-of-mass velocity *and* the total angular momentum from the initial velocities. Monitor both, together with the energy.
5. **Temperature.** T = 2⟨K⟩ / (f k_B) with **f = 3N − 6 = 33** [31]. Microcanonical C_V from kinetic-energy fluctuations [31].
6. **Energy ladder (Task 5).** Scale velocities from the equilibrated low-E state in small increments. Equilibrate, then average. Run both heating and cooling sweeps to check for hysteresis [20].
7. **Timestep study (Task 4).** Log–log plot of σ_E and drift against h, where the expected slope for σ_E is 2 [8]. Additionally check the convergence of T(E) and δ(E) (D2).
8. **δ definition.** Use the slide formula, but **always state the averaging window τ**, and compute δ over sliding windows (D1).
9. **About "δ ≈ 1 = liquid" in the slides.** The literature reports δ of a few per cent for the solid, rising through ~0.1 and levelling off at roughly 0.2–0.3 for a liquid-like cluster [16, 19, 23, 46]. δ approaches 1 only if pair distances fluctuate by as much as their mean, which in practice indicates **evaporation or dissociation**, not melting. We should say this explicitly in the report and treat the slide's statement as a loose upper bound rather than a threshold (**verify our own numbers first**).
10. **Evaporation.** Detect it (distance from the centre of mass > R_c) and either discard the run or apply a weak confining wall (D5). Report which option was used.
11. **Independent indicators.** Add at least one of: short-time kinetic-energy histograms [19], periodic quenches giving f_liq [54], or Q₆ of the central atom [47].
12. **Scaling (Task 6 and course requirement).** Energies are independent, so parallelise over the energy grid (for example with `multiprocessing` or on the HPC cluster). Report sequential vs parallel timings. For N ≳ 100, consider neighbour lists [6].

---

## 8. Bibliography

All entries with a DOI were checked against Crossref on 2026-09-29. BibTeX: [`references.bib`](references.bib).

**Foundations and integrators**

1. J. E. Jones, "On the determination of molecular fields. II. From the equation of state of a gas," *Proc. R. Soc. Lond. A* **106**, 463–477 (1924). https://doi.org/10.1098/rspa.1924.0082
2. F. A. Lindemann, "Über die Berechnung molekularer Eigenfrequenzen," *Phys. Z.* **11**, 609–612 (1910). *(no DOI; not machine-verified)*
3. P. Pawlow, "Über die Abhängigkeit des Schmelzpunktes von der Oberflächenenergie eines festen Körpers," *Z. Phys. Chem.* **65U**, 1–35 (1909). https://doi.org/10.1515/zpch-1909-6502
4. B. J. Alder, T. E. Wainwright, "Studies in molecular dynamics. I. General method," *J. Chem. Phys.* **31**, 459–466 (1959). https://doi.org/10.1063/1.1730376
5. A. Rahman, "Correlations in the motion of atoms in liquid argon," *Phys. Rev.* **136**, A405–A411 (1964). https://doi.org/10.1103/PhysRev.136.A405
6. L. Verlet, "Computer 'experiments' on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules," *Phys. Rev.* **159**, 98–103 (1967). https://doi.org/10.1103/PhysRev.159.98
7. W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, "A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: application to small water clusters," *J. Chem. Phys.* **76**, 637–649 (1982). https://doi.org/10.1063/1.442716
8. E. Hairer, C. Lubich, G. Wanner, "Geometric numerical integration illustrated by the Störmer–Verlet method," *Acta Numerica* **12**, 399–450 (2003). https://doi.org/10.1017/S0962492902000144
9. R. D. Engle, R. D. Skeel, M. Drees, "Monitoring energy drift with shadow Hamiltonians," *J. Comput. Phys.* **206**, 432–452 (2005). https://doi.org/10.1016/j.jcp.2004.12.009
10. R. D. Skeel, "What makes molecular dynamics work?," *SIAM J. Sci. Comput.* **31**, 1363–1378 (2009). https://doi.org/10.1137/070683660
11. S. Toxvaerd, O. J. Heilmann, J. C. Dyre, "Energy conservation in molecular dynamics simulations of classical systems," *J. Chem. Phys.* **136**, 224106 (2012). https://doi.org/10.1063/1.4726728
12. M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford University Press (2017). https://doi.org/10.1093/oso/9780198803195.001.0001
13. D. Frenkel, B. Smit, *Understanding Molecular Simulation: From Algorithms to Applications*, 3rd ed., Academic Press/Elsevier (2023).
14. B. Leimkuhler, C. Matthews, *Molecular Dynamics: With Deterministic and Stochastic Numerical Methods*, Springer (2015). https://doi.org/10.1007/978-3-319-16375-8
15. M. E. Tuckerman, *Statistical Mechanics: Theory and Molecular Simulation*, 2nd ed., Oxford University Press (2023). https://doi.org/10.1093/oso/9780198825562.001.0001

**Rare-gas cluster melting simulations**

16. C. L. Briant, J. J. Burton, "Molecular dynamics study of the structure and thermodynamic properties of argon microclusters," *J. Chem. Phys.* **63**, 2045–2058 (1975). https://doi.org/10.1063/1.431542
17. R. D. Etters, J. Kaelberer, "Thermodynamic properties of small aggregates of rare-gas atoms," *Phys. Rev. A* **11**, 1068–1079 (1975). https://doi.org/10.1103/PhysRevA.11.1068
18. J. B. Kaelberer, R. D. Etters, "Phase transitions in small clusters of atoms," *J. Chem. Phys.* **66**, 3233–3239 (1977). https://doi.org/10.1063/1.434298
19. J. Jellinek, T. L. Beck, R. S. Berry, "Solid–liquid phase changes in simulated isoenergetic Ar13," *J. Chem. Phys.* **84**, 2783–2794 (1986). https://doi.org/10.1063/1.450303
20. J. D. Honeycutt, H. C. Andersen, "Molecular dynamics study of melting and freezing of small Lennard-Jones clusters," *J. Phys. Chem.* **91**, 4950–4963 (1987). https://doi.org/10.1021/j100303a014
21. H. L. Davis, J. Jellinek, R. S. Berry, "Melting and freezing in isothermal Ar13 clusters," *J. Chem. Phys.* **86**, 6456–6464 (1987). https://doi.org/10.1063/1.452436
22. T. L. Beck, J. Jellinek, R. S. Berry, "Rare gas clusters: Solids, liquids, slush, and magic numbers," *J. Chem. Phys.* **87**, 545–554 (1987). https://doi.org/10.1063/1.453602
23. R. S. Berry, T. L. Beck, H. L. Davis, J. Jellinek, "Solid–liquid phase behavior in microclusters," *Adv. Chem. Phys.* **70**, 75–138 (1988). https://doi.org/10.1002/9780470122693.ch3
24. D. J. Wales, R. S. Berry, "Melting and freezing of small argon clusters," *J. Chem. Phys.* **92**, 4283–4295 (1990). https://doi.org/10.1063/1.457788
25. D. J. Wales, R. S. Berry, "Freezing, melting, spinodals, and clusters," *J. Chem. Phys.* **92**, 4473–4482 (1990). https://doi.org/10.1063/1.457758

**Thermodynamics of finite systems**

26. P. Labastie, R. L. Whetten, "Statistical thermodynamics of the cluster solid–liquid transition," *Phys. Rev. Lett.* **65**, 1567–1570 (1990). https://doi.org/10.1103/PhysRevLett.65.1567
27. R. E. Kunz, R. S. Berry, "Coexistence of multiple phases in finite systems," *Phys. Rev. Lett.* **71**, 3987–3990 (1993). https://doi.org/10.1103/PhysRevLett.71.3987
28. D. J. Wales, R. S. Berry, "Coexistence in finite systems," *Phys. Rev. Lett.* **73**, 2875–2878 (1994). https://doi.org/10.1103/PhysRevLett.73.2875
29. R. M. Lynden-Bell, D. J. Wales, "Free energy barriers to melting in atomic clusters," *J. Chem. Phys.* **101**, 1460–1476 (1994). https://doi.org/10.1063/1.467771
30. J. P. K. Doye, D. J. Wales, "An order parameter approach to coexistence in atomic clusters," *J. Chem. Phys.* **102**, 9673–9688 (1995). https://doi.org/10.1063/1.468786
31. E. M. Pearson, T. Halicioglu, W. A. Tiller, "Laplace-transform technique for deriving thermodynamic equations from the classical microcanonical ensemble," *Phys. Rev. A* **32**, 3030–3039 (1985). https://doi.org/10.1103/PhysRevA.32.3030
32. M. Bixon, J. Jortner, "Energetic and thermodynamic size effects in molecular clusters," *J. Chem. Phys.* **91**, 1631–1642 (1989). https://doi.org/10.1063/1.457123

**Sampling and heat capacities**

33. C. J. Tsai, K. D. Jordan, "Use of the histogram and jump-walking methods for overcoming slow barrier crossing behavior in Monte Carlo simulations: Applications to the phase transitions in the (Ar)13 and (H2O)8 clusters," *J. Chem. Phys.* **99**, 6957–6970 (1993). https://doi.org/10.1063/1.465442
34. D. D. Frantz, "Magic numbers for classical Lennard-Jones cluster heat capacities," *J. Chem. Phys.* **102**, 3747–3768 (1995). https://doi.org/10.1063/1.468557
35. F. Calvo, P. Labastie, "Configurational density of states from molecular dynamics simulations," *Chem. Phys. Lett.* **247**, 395–400 (1995). https://doi.org/10.1016/0009-2614(95)01226-5
36. J. P. Neirotti, F. Calvo, D. L. Freeman, J. D. Doll, "Phase changes in 38-atom Lennard-Jones clusters. I. A parallel tempering study in the canonical ensemble," *J. Chem. Phys.* **112**, 10340–10349 (2000). https://doi.org/10.1063/1.481671
37. V. A. Mandelshtam, P. A. Frantsuzov, F. Calvo, "Structural transitions and melting in LJ74–78 Lennard-Jones clusters from adaptive exchange Monte Carlo simulations," *J. Phys. Chem. A* **110**, 5326–5332 (2006). https://doi.org/10.1021/jp055839l
38. E. G. Noya, J. P. K. Doye, "Structural transitions in the 309-atom magic number Lennard-Jones cluster," *J. Chem. Phys.* **124**, 104503 (2006). https://doi.org/10.1063/1.2173260
39. K. Hukushima, K. Nemoto, "Exchange Monte Carlo method and application to spin glass simulations," *J. Phys. Soc. Jpn.* **65**, 1604–1608 (1996). https://doi.org/10.1143/JPSJ.65.1604
40. D. J. Earl, M. W. Deem, "Parallel tempering: Theory, applications, and new perspectives," *Phys. Chem. Chem. Phys.* **7**, 3910–3916 (2005). https://doi.org/10.1039/b509983h
41. A. M. Ferrenberg, R. H. Swendsen, "Optimized Monte Carlo data analysis," *Phys. Rev. Lett.* **63**, 1195–1198 (1989). https://doi.org/10.1103/PhysRevLett.63.1195
42. F. Wang, D. P. Landau, "Efficient, multiple-range random walk algorithm to calculate the density of states," *Phys. Rev. Lett.* **86**, 2050–2053 (2001). https://doi.org/10.1103/PhysRevLett.86.2050
43. S. Nosé, "A unified formulation of the constant temperature molecular dynamics methods," *J. Chem. Phys.* **81**, 511–519 (1984). https://doi.org/10.1063/1.447334
44. W. G. Hoover, "Canonical dynamics: Equilibrium phase-space distributions," *Phys. Rev. A* **31**, 1695–1697 (1985). https://doi.org/10.1103/PhysRevA.31.1695
45. G. Bussi, D. Donadio, M. Parrinello, "Canonical sampling through velocity rescaling," *J. Chem. Phys.* **126**, 014101 (2007). https://doi.org/10.1063/1.2408420

**Order parameters and melting criteria**

46. Y. Zhou, M. Karplus, K. D. Ball, R. S. Berry, "The distance fluctuation criterion for melting: Comparison of square-well and Morse potential models for clusters and homopolymers," *J. Chem. Phys.* **116**, 2323–2329 (2002). https://doi.org/10.1063/1.1426419
47. P. J. Steinhardt, D. R. Nelson, M. Ronchetti, "Bond-orientational order in liquids and glasses," *Phys. Rev. B* **28**, 784–805 (1983). https://doi.org/10.1103/PhysRevB.28.784
48. W. Lechner, C. Dellago, "Accurate determination of crystal structures based on averaged local bond order parameters," *J. Chem. Phys.* **129**, 114707 (2008). https://doi.org/10.1063/1.2977970
49. A. Stukowski, "Structure identification methods for atomistic simulations of crystalline materials," *Modelling Simul. Mater. Sci. Eng.* **20**, 045021 (2012). https://doi.org/10.1088/0965-0393/20/4/045021
50. S. Sarkar, P. K. Jana, B. Bagchi, "Breakdown of universal Lindemann criterion in the melting of Lennard-Jones polydisperse solids," *J. Chem. Sci.* **129**, 833–840 (2017). https://doi.org/10.1007/s12039-017-1245-y
51. R. Guardiola, J. Navarro, "On the Lindemann criterion for quantum clusters at very low temperature," *J. Phys. Chem. A* **115**, 6843–6850 (2011). https://doi.org/10.1021/jp1111313
52. S.-Y. Zeng, C.-H. Hsu, T.-M. Wu, "Bond orientational order parameters for classifying solid-like clusters in a Lennard-Jones system near liquid–solid transition and at solid states," *J. Phys. Chem. A* **126**, 2018–2030 (2022). https://doi.org/10.1021/acs.jpca.1c09527

**Energy landscapes**

53. M. R. Hoare, P. Pal, "Physical cluster mechanics: Statics and energy surfaces for monatomic systems," *Adv. Phys.* **20**, 161–196 (1971). https://doi.org/10.1080/00018737100101231
54. F. H. Stillinger, T. A. Weber, "Hidden structure in liquids," *Phys. Rev. A* **25**, 978–989 (1982). https://doi.org/10.1103/PhysRevA.25.978
55. F. H. Stillinger, T. A. Weber, "Packing structures and transitions in liquids and solids," *Science* **225**, 983–989 (1984). https://doi.org/10.1126/science.225.4666.983
56. J. A. Northby, "Structure and binding of Lennard-Jones clusters: 13 ≤ N ≤ 147," *J. Chem. Phys.* **87**, 6166–6177 (1987). https://doi.org/10.1063/1.453492
57. D. J. Wales, J. P. K. Doye, "Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms," *J. Phys. Chem. A* **101**, 5111–5116 (1997). https://doi.org/10.1021/jp970984n
58. J. P. K. Doye, M. A. Miller, D. J. Wales, "Evolution of the potential energy surface with size for Lennard-Jones clusters," *J. Chem. Phys.* **111**, 8417–8428 (1999). https://doi.org/10.1063/1.480217
59. J. P. K. Doye, D. J. Wales, "Thermodynamics of global optimization," *Phys. Rev. Lett.* **80**, 1357–1360 (1998). https://doi.org/10.1103/PhysRevLett.80.1357
60. K. D. Ball, R. S. Berry, "Dynamics on statistical samples of potential energy surfaces," *J. Chem. Phys.* **111**, 2060–2070 (1999). https://doi.org/10.1063/1.479474
61. D. J. Wales, "Discrete path sampling," *Mol. Phys.* **100**, 3285–3305 (2002). https://doi.org/10.1080/00268970210162691
62. D. J. Wales, *Energy Landscapes: Applications to Clusters, Biomolecules and Glasses*, Cambridge University Press (2003). *(no DOI; not machine-verified)*
63. D. J. Wales, "Exploring energy landscapes," *Annu. Rev. Phys. Chem.* **69**, 401–425 (2018). https://doi.org/10.1146/annurev-physchem-050317-021219
64. J. P. K. Doye, F. Calvo, "Entropic effects on the size dependence of cluster structure," *Phys. Rev. Lett.* **86**, 3570–3573 (2001). https://doi.org/10.1103/PhysRevLett.86.3570

**Chaos and dynamics**

65. R. J. Hinde, R. S. Berry, D. J. Wales, "Chaos in small clusters of inert gas atoms," *J. Chem. Phys.* **96**, 1376–1390 (1992). https://doi.org/10.1063/1.462173
66. R. J. Hinde, R. S. Berry, "Chaotic dynamics in small inert gas clusters: The influence of potential energy saddles," *J. Chem. Phys.* **99**, 2942–2963 (1993). https://doi.org/10.1063/1.465201
67. S. K. Nayak, R. Ramaswamy, C. Chakravarty, "Maximal Lyapunov exponent in small atomic clusters," *Phys. Rev. E* **51**, 3376–3380 (1995). https://doi.org/10.1103/PhysRevE.51.3376

**Quantum effects**

68. F. Calvo, J. P. K. Doye, D. J. Wales, "Quantum partition functions from classical distributions: Application to rare-gas clusters," *J. Chem. Phys.* **114**, 7312–7329 (2001). https://doi.org/10.1063/1.1359768
69. P. A. Frantsuzov, V. A. Mandelshtam, "Quantum statistical mechanics with Gaussians: Equilibrium properties of van der Waals clusters," *J. Chem. Phys.* **121**, 9247–9256 (2004). https://doi.org/10.1063/1.1804495

**Experiments and other materials**

70. Ph. Buffat, J.-P. Borel, "Size effect on the melting temperature of gold particles," *Phys. Rev. A* **13**, 2287–2298 (1976). https://doi.org/10.1103/PhysRevA.13.2287
71. M. Schmidt, R. Kusche, W. Kronmüller, B. von Issendorff, H. Haberland, "Experimental determination of the melting point and heat capacity for a free cluster of 139 sodium atoms," *Phys. Rev. Lett.* **79**, 99–102 (1997). https://doi.org/10.1103/PhysRevLett.79.99
72. M. Schmidt, R. Kusche, B. von Issendorff, H. Haberland, "Irregular variations in the melting point of size-selected atomic clusters," *Nature* **393**, 238–240 (1998). https://doi.org/10.1038/30415
73. M. Schmidt, R. Kusche, T. Hippler, J. Donges, W. Kronmüller, B. von Issendorff, H. Haberland, "Negative heat capacity for a cluster of 147 sodium atoms," *Phys. Rev. Lett.* **86**, 1191–1194 (2001). https://doi.org/10.1103/PhysRevLett.86.1191
74. G. A. Breaux, R. C. Benirschke, T. Sugai, B. S. Kinnear, M. F. Jarrold, "Hot and solid gallium clusters: Too small to melt," *Phys. Rev. Lett.* **91**, 215508 (2003). https://doi.org/10.1103/PhysRevLett.91.215508
75. C. L. Cleveland, W. D. Luedtke, U. Landman, "Melting of gold clusters: Icosahedral precursors," *Phys. Rev. Lett.* **81**, 2036–2039 (1998). https://doi.org/10.1103/PhysRevLett.81.2036
76. D. Sabo, C. Predescu, J. D. Doll, D. L. Freeman, "Phase changes in selected Lennard-Jones X₁₃₋ₙYₙ clusters," *J. Chem. Phys.* **121**, 856–867 (2004). https://doi.org/10.1063/1.1759625

**Reviews**

77. F. Baletto, R. Ferrando, "Structural properties of nanoclusters: Energetic, thermodynamic, and kinetic effects," *Rev. Mod. Phys.* **77**, 371–423 (2005). https://doi.org/10.1103/RevModPhys.77.371
78. A. Proykova, R. S. Berry, "Insights into phase transitions from phase changes of clusters," *J. Phys. B* **39**, R167–R202 (2006). https://doi.org/10.1088/0953-4075/39/9/R01
79. R. S. Berry, B. M. Smirnov, "Phase transitions in various kinds of clusters," *Phys. Usp.* **52**, 137–164 (2009). https://doi.org/10.3367/UFNe.0179.200902b.0147
80. A. Aguado, M. F. Jarrold, "Melting and freezing of metal clusters," *Annu. Rev. Phys. Chem.* **62**, 151–172 (2011). https://doi.org/10.1146/annurev-physchem-032210-103454

**Machine learning and recent work**

81. J. Behler, M. Parrinello, "Generalized neural-network representation of high-dimensional potential-energy surfaces," *Phys. Rev. Lett.* **98**, 146401 (2007). https://doi.org/10.1103/PhysRevLett.98.146401
82. T. E. Smidt, M. Geiger, B. K. Miller, "Finding symmetry breaking order parameters with Euclidean neural networks," *Phys. Rev. Research* **3**, L012002 (2021). https://doi.org/10.1103/PhysRevResearch.3.L012002
83. K. Z. Takahashi, "Molecular cluster analysis using local order parameters selected by machine learning," *Phys. Chem. Chem. Phys.* **25**, 658–672 (2023). https://doi.org/10.1039/D2CP03696G *(see also the published correction, doi:10.1039/D2CP90240K)*
84. L. E. S. Hoffenberg, A. Khrabry, Y. Barsukov, I. D. Kaganovich, D. B. Graves, "Size-dependent second-order-like phase transitions in Fe nanocluster melting from low-temperature structural isomerization," *J. Chem. Phys.* **162**, 134305 (2025). https://doi.org/10.1063/5.0236122
