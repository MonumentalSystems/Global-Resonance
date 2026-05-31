# Critical Learning Spec

## Thesis

Self-organizing systems with feedback generically evolve toward the Kosterlitz-Thouless critical stiffness `J_c = 2/pi` because this is the regime that best balances:

- coherence: the ability to sustain structure, preserve memory, and transmit information
- adaptability: the ability to respond to perturbation, explore new states, and avoid rigid lock-in

Below `J_c`, dynamics are too disordered to support robust organization. Above `J_c`, dynamics are too synchronized to remain responsive. Near `J ~= J_c`, the system should maximize useful computation.

This project treats that claim as an engineering target, not just a philosophical statement.

## Governing Principle

The lawful core is fixed by Clifford dynamics and bivector non-commutativity:

`d_t F = lawful_dynamics(F, grad F, forcing; kappa=1, J)`

with:

- `kappa = 1` fixed by the algebra
- `J` the primary control parameter across scales
- the Lohe dynamics treated as the KT boundary constraint in the 2D reduction
- task-specific learning restricted to forcing, readout, and narrow control surfaces around the lawful core

## Parameter Intuition

For collaborators, the most useful plain-language reading of the core parameters is:

- `J`: stiffness, coupling strength, or phase-locking pressure
- `Omega`: the local intrinsic rotation tendency
- `gamma`: damping or dissipation
- `forcing`: external perturbation or input drive

The most important quantity is `J`.

`J` is called stiffness because it measures how strongly the system resists relative phase twist or local mismatch across neighboring sites.

In practical terms:

- low `J`: sites evolve more independently, so the system is flexible but struggles to sustain coherent structure
- high `J`: sites strongly constrain one another, so the system preserves coherence but risks becoming rigid
- near-critical `J`: the system is coherent enough to carry memory and structure, but adaptive enough to respond and reorganize

So `J` should not be read as a learning rate or generic energy scale. It is better understood as the strength of neighborhood influence in the lawful dynamics.

In the current MVP implementation, `J` directly scales the neighbor-coupling commutator term, which means it answers the question:

"How strongly should local state evolution be shaped by adjacent states rather than only by local rotation, damping, and forcing?"

## Design Goal

Build models where the dynamics carry most of the computational weight.

This means:

- the state is rich
- the update law is mostly fixed
- learnable capacity is small and interpretable
- the model is explicitly measured for how close it remains to the useful critical regime

## Critical Learning Hypothesis

A lawful dynamical system governed by Clifford/Lohe/Kuramoto-compatible updates becomes a better learning substrate when controlled to remain near `J_c = 2/pi`.

Predictions:

1. A sweep over `J` should reveal a regime window near `J_c` that outperforms lower-`J` and higher-`J` settings on tasks requiring both memory and adaptation.
2. Near-critical systems should show stronger tradeoffs than either:
   - subcritical systems: flexible but incoherent
   - supercritical systems: coherent but rigid
3. Diagnostics such as synchronization order, susceptibility, defect persistence, and recovery after perturbation should jointly peak or balance near the same regime.

## Biological Model

The motivating claim is stronger than analogy. The proposal is that biological discharge is a realization of the same KT boundary mechanism already formalized in the Coulomb-gas and vortex language.

On this reading, firing does not create organization from nothing. Organization builds first, and firing is a thresholded phase-shift or release event.

In cortical terms:

- charge accumulates across local assemblies
- partial synchrony builds as neighboring units reinforce one another
- inhibition prevents weak or noisy coordination from propagating
- firing occurs when accumulated coherence becomes strong enough to cross a symmetry-breaking or KT-boundary transition

This suggests that useful learning systems should not be optimized for raw synchronization alone. They should be optimized for synchronized accumulation under constraint, near the threshold where coordinated phase transition or release becomes possible.

Under this view:

- forcing contributes local charge or drive
- `J` determines whether that drive remains isolated or becomes collective coherence
- `gamma` removes stale or weak structure
- inhibition or thresholding governs whether a coordinated assembly can discharge into action

In the theorem language, this biological picture is expected to correspond to:

- vortex suppression above the boundary
- vortex proliferation below the boundary
- fugacity relevance and irrelevance across the boundary
- screening-driven redistribution of stiffness

So the biological model is:

- charge accumulates
- coherence builds
- the assembly approaches the KT boundary
- a phase shift / discharge occurs when the accumulated state crosses the effective threshold

The computational burden therefore lives in the buildup, regulation, and release of coherent assemblies, not only in the final event itself.

## Multi-Frequency Stabilization

Near-critical systems can become turbulent. In the KT picture this appears as vortex-rich, defect-mediated dynamics. The hypothesis here is that such turbulence need not be eliminated to become useful. It can be partially tamed by structured multi-frequency reinforcement.

The biological claim is that brain activity across multiple reinforcing frequency bands may be using this same control mechanism. Different oscillatory bands can:

- stabilize coordination over distinct scales
- gate when local assemblies can bind or release
- reduce destructive turbulence without collapsing the system into rigid lockstep
- preserve adaptability while improving coherence transfer

For the model family, this suggests that a good near-critical substrate may require not only the right average `J`, but also lawful multi-scale rhythm or forcing structure that helps regulate defect turbulence.

This leads to an additional design prediction:

- multi-frequency or multi-timescale coupling should improve stability and controllability near `J_c` without simply pushing the system into over-synchronization

Current executable sweeps suggest a sharper refinement: temporal `2/pi` behaves as the strongest controller for remaining near the critical manifold in this prototype. `phi` remains a plausible rotational companion constant, but it is not currently the strongest critical controller on the measured objectives.

Those objectives should be kept separate:

- `balance` asks for the best overall coherence/adaptability tradeoff
- `critical release` asks for the cleanest boundary-locking regime, emphasizing low critical gap, high release, and low suppression

The present results favor temporal `2/pi` most clearly on the second objective, which is the one most tightly tied to the KT claim.

## Architecture

### 1. Lawful Core

The core state should be a local Clifford-valued field, lattice, or graph.

Required properties:

- explicit `J`
- explicit damping
- explicit forcing
- explicit inhibition / thresholded discharge structure
- local coupling and transport
- normalization or conservation rules justified by the formal theory

### 2. Criticality Controller

The controller is intentionally small.

Its job is not to invent dynamics, but to steer the lawful system toward the useful critical regime.

Permitted learnable roles:

- local or global modulation of `J`
- forcing gain
- readout temperature
- sparse regime-control variables

Disallowed by default:

- large embedding stacks
- large MLP shells that can solve the task without the dynamics
- arbitrary attention-style replacement of the lawful update

### 3. Task Interface

Inputs act as forcing on the lawful state.

Outputs are read out from observables of that state:

- local or global Clifford projections
- defect cache / constellation statistics
- field averages
- synchronization observables
- topological event traces

### 4. Diagnostics Layer

Criticality must be measured directly.

Core metrics:

- synchronization order parameter
- susceptibility / perturbation response
- defect density or cache activity
- correlation / dispersion across sites
- adaptation rate
- dwell time / metastability
- distance to `J_c`

Derived metric:

- coherence-adaptability balance score

## Minimal Viable Variant

The first implementation should be intentionally small but structurally correct.

Recommended MVP:

- a small ring or lattice of Clifford states
- shared lawful update with explicit `J`
- global `Omega`, `gamma`, `J`, inhibition threshold, discharge gain, forcing gain, and readout temperature
- shared defect cache readout
- explicit criticality diagnostics at every step

In the current implementation, inhibition and discharge are a first executable bridge from the proved vortex/fugacity boundary to the biological parameterization:

- local coherence and forcing contribute to an accumulated drive proxy
- a threshold models inhibition
- a discharge term only amplifies collective release once the threshold is exceeded

This should be treated as an executable first mapping of the formal KT/Coulomb boundary into a biological learning model, not yet as the final completed derivation.

The point of the MVP is not to be state of the art. The point is to encode the theory faithfully enough to test whether near-critical steering improves useful computation.

## Training Objective

The loss should not be task-only.

Suggested form:

`L = L_task + lambda_gap * L_critical_gap + lambda_sync * L_sync_balance + lambda_stab * L_stability`

where:

- `L_task` measures prediction quality
- `L_critical_gap` penalizes distance from `J_c`
- `L_sync_balance` penalizes both over-synchronization and under-synchronization
- `L_stability` prevents collapse or divergence

The exact weighting can evolve, but the regime objective must remain explicit.

## Benchmark Ladder

### Regime validation

- sweep `J`
- verify distinct subcritical / near-critical / supercritical behaviors
- measure whether the critical window is stable across resolution and sequence length

### Substrate quality

- memory under perturbation
- recovery after shock
- attractor diversity
- persistence of topological or defect structure
- sensitivity without collapse

### Task performance

- temporal recall
- sequence continuation
- associative retrieval
- byte-level language modeling
- audio / waveform forcing and readout

### Comparative claim

Hold architecture fixed and vary only regime.

If the theory is right, the best useful-computation zone should cluster near `J_c = 2/pi`.

## Failure Conditions

The critical-learning thesis should be considered weakened if:

1. performance does not improve near `J_c`
2. diagnostics do not show a distinctive balance regime near `J_c`
3. large learned wrappers are required to make the system useful
4. the dynamics can be replaced by a trivial readout shell without loss

## Implementation Rule

When in doubt, prefer:

- richer lawful state
- smaller learned controller

over:

- thinner state
- thicker learned wrapper

The project succeeds only if the dynamics genuinely carry the computational burden.
