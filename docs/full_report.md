# PSPE — Complete Repository Report

**Perceive · Simulate · Plan · Explain: constrained intervention design for PDE-governed systems**

Compiled 2026-09-21 from the repository state on branch `phase-1-2-theory-and-joint`
(HEAD `df9a7a6` plus uncommitted edits to `docs/technical_report.md` and the TACC
scripts). Every number in this document was read from a file under `runs/`, from a
docstring, or recomputed from the per-seed JSON; where a number could not be traced
to a local artefact this is said explicitly. Section 12 lists issues found during
this audit that a paper draft must handle before any of the affected numbers are
quoted.

This document is written to be mined for a research paper: it states the problem,
every equation the code implements, every experiment with its protocol and result,
the theory and its measured status, the baselines and how they were validated,
and the positioning against prior work.

---

## Table of contents

1. Summary
2. Problem formulation
3. Testbeds, data, actuators, reward and constraints
4. Method: the four modules, with the mathematics as implemented
5. Theory: assumptions, propositions and their measured status
6. Baselines and their validation
7. Experimental protocol and instrumentation
8. Results
9. Positioning against prior work
10. Code map and reproduction
11. Claims that survive, claims that do not, claims not yet tested
12. Audit findings: issues to resolve before publication
13. References

---

## 1. Summary

PSPE is a four-module loop for deciding **where and how hard to intervene** in a
spatially extended physical system governed by a PDE, under hard constraints, with
a natural-language account of each decision:

| Module | Input → output | Implementation |
|---|---|---|
| Perceive | imagery → physical field estimate | frozen vision backbone (SigLIP, or an in-tree stub) + LoRA adapters + conv decoder; regression + InfoNCE against weak text |
| Simulate | (field, control) → next field | in-tree Fourier Neural Operator (FNO); DeepONet and GNOT as alternatives; data + physics-residual + multi-step rollout losses |
| Plan | field → actuator amplitudes | tanh-Gaussian policy; hybrid pathwise / likelihood-ratio gradient with variance-optimal mixing; PID-Lagrangian dual; optional second (equity) dual |
| Explain | (state, action) → brief | frozen LM (Qwen2.5-0.5B, or an in-tree stub) used as a sequence encoder with trainable brief embedding/head + prefix; frozen parser; faithfulness $F(b)=\exp(-\mathrm{KL})$; split-conformal certificate |

Three synthetic PDE testbeds (diffusion–advection–reaction `dar`, linearised
shallow water `swe`, FitzHugh–Nagumo fronts `rdf`) on a periodic $64\times64$
grid are solved by in-tree differentiable RK4 solvers; the same `rhs` is the
ground truth and the physics residual. Two real datasets are wired in and run:
PDEBench 2D shallow water (surrogate accuracy) and Next Day Wildfire Spread
(observed fire dynamics). Real backbones (SigLIP, Qwen2.5-0.5B) have been run for
Perceive and Explain.

Headline findings that survived multi-seed reruns (5 seeds unless stated, paired
$t$ on 4 df, $|t|>2.78 \Rightarrow p<0.05$):

* The planner beats all four reimplemented safe-RL baselines on return
  ($t=+3.84$ to $+17.95$) using $12.5\times$ fewer real transitions (3,072 vs
  38,400), or $9.2\times$ with the safety fix (4,160).
* As originally built the planner **violated the cost constraint on 7.3% of
  evaluations** while every model-free baseline violated on 0%. Root cause: the
  dual is driven by surrogate cost, violation is realised under true dynamics. A
  real-environment probe every 20 iterations plus a $2\sigma$ margin cut this to
  about 1% of evaluations across 15 seed-runs at no measurable return cost.
* FNO $\gg$ DeepONet $\gg$ GNOT on synthetic data (0.057 / 0.256 / 0.268 rel $L_2$)
  and on PDEBench shallow water (0.0098 / 0.0137 / 0.0543), where the FNO is within
  $\sim2\times$ of the published FNO number (different metric).
* Resolution invariance: an FNO trained at $64^2$ scores 0.0298 / 0.0298 / 0.0300
  at $64^2$ / $96^2$ / $128^2$.
* Joint Simulate+Plan training with a data anchor does not change the decision
  ($t=+0.66$) but lowers held-out surrogate error $9\times$; unanchored it
  destroys the surrogate (rel $L_2$ 0.005 → 0.975) and lowers return ($t=-5.84$).
* Assumption 1 ($L_G\le1$) holds on the trained FNO (0.985 ± 0.013; 4 of 5 seeds;
  5 of 5 with spectral normalisation). Proposition 1's infinite-horizon bound is
  vacuous ($4{,}000\times$ loose); the finite-horizon form holds and is $100\times$ loose.
* A generator supervised on the policy's reference briefs reaches $F(b)=0.468 \pm 0.16$
  on Qwen2.5-0.5B vs 0.182 for the "post-hoc" arm ($t=+21$); the REINFORCE
  faithfulness term adds nothing at its default weight and hurts above it.
* The split-conformal faithfulness certificate holds at 90% coverage on 3 of 3
  Qwen seeds (n_cal 200, n_test 400).
* On observed wildfire spread a U-Net reaches 0.277 ± 0.002 AUC-PR (97.5% of the
  published 0.284); a gated FNO path is declined by the model (gate 0.12) and the
  hybrid is worse ($t=+12.5$).

Findings that did not survive: physics-informed loss improving final accuracy;
adaptive $\alpha$ improving mean return (it prevents dual divergence instead);
the faithfulness loss term; LoRA + contrastive beating a frozen probe or a CNN on
field reconstruction (it wins retrieval instead); any magnitude for cross-family
transfer into the wave family; joint training beating the disaggregated pipeline
on return; the Eq. 8 mixing rule improving on the variance-only rule.

Section 12 lists audit findings — notably that the "post-hoc" explanation control
is an *untrained* generator, that Qwen briefs are truncated at 96 tokens against a
136-token reference, and that the headline 5-seed planner sweep did not pass the
seed into the planner config (so its evaluation initial conditions are identical
across seeds).

---

## 2. Problem formulation

### 2.1 Constrained partially observed MDP

The decision problem is a constrained POMDP (Altman, 1999):

$$
\max_{\pi}\; J_R(\pi)=\mathbb{E}_\pi\Big[\sum_{t=0}^{H-1}\gamma^t r(u_{t+1},a_t)\Big]
\quad\text{s.t.}\quad
J_{C_j}(\pi)=\mathbb{E}_\pi\Big[\sum_{t=0}^{H-1} g_j(u_{t+1},a_t)\Big]\le d_j,\;\; j=1,\dots,m,
$$

where $u_t\in\mathbb{R}^{C\times N\times N}$ is the physical field, $a_t\in[-1,1]^K$
the actuator amplitudes, and the field is not observed directly: the planner
receives $\hat u_t=\text{Perceive}(o_t)$ from imagery $o_t$ (or the true field, the
"perfect-perception" upper bound used in every planning experiment). In the code
the return is undiscounted at evaluation ($\gamma$ enters only the reward-to-go used
for the likelihood-ratio advantage), $H=12$, $K=9$.

Two constraints are implemented; the first is on in every experiment, the second
is off by default:

* $g_1$ (safety): exposure above a ceiling plus actuation over budget;
* $g_2$ (equity): the variance of residual harm across an $R\times R$ partition
  of the domain (Section 3.4).

Constraints are enforced by Lagrangian relaxation with one PID-controlled
multiplier per constraint:

$$
\mathcal{L}(\pi,\lambda)=J_R(\pi)-\sum_j\lambda_j\,(J_{C_j}(\pi)-d_j).
$$

### 2.2 Dynamics

The true dynamics are a discrete-time map $u_{t+1}=F(u_t,s_t)$ obtained by RK4
integration of a PDE with an additive intervention source $s_t=\Phi(a_t)$ on the
first channel (Section 3). The planner acts through a learned surrogate
$G_\theta\approx F$ that is differentiable in both arguments.

### 2.3 Sample accounting

Because the whole model-based argument rests on real-sample efficiency, every run
records separately: `samples_real_env` (transitions from the true solver:
surrogate training data + any real-cost probes), `samples_surrogate` (rollouts
inside $G_\theta$), and `rollout_steps`. Baselines report only real transitions.

---

## 3. Testbeds, data, actuators, reward and constraints

### 3.1 The three PDE families (`pspe/simulate/solvers.py`)

All on the periodic unit square, $N\times N$ grid ($N=64$ by default,
$\Delta x=1/N$), second-order central differences for $\partial_x,\partial_y$ and
the 5-point Laplacian, RK4 in time with `substeps` micro-steps per macro step
$\Delta t$. The macro step is what the surrogate learns. Control enters as an
additive source $s(x,t)$ on channel 0.

**`dar` — diffusion–advection–reaction** (1 channel, $\Delta t=0.02$, 4 substeps):

$$
u_t = D\,\nabla^2 u - \mathbf v\cdot\nabla u + r\,u(1-u) + s,\qquad
D=0.01,\ \mathbf v=(0.35,0.2),\ r=0.6 .
$$

Initial condition: $0.8\times$ a smooth random Fourier field (4 modes/axis,
amplitude decay $1/(1+k_x^2+k_y^2)$, min–max normalised to $[0,1]$).

**`swe` — linearised shallow water with drag** (3 channels $(h,u,v)$, $\Delta t=0.01$, 8 substeps):

$$
h_t=-H(u_x+v_y)-c_h h + s,\qquad u_t=-g\,h_x-c\,u,\qquad v_t=-g\,h_y-c\,v,
$$
$g=1,\ H=1,\ c=0.1,\ c_h=0.02$. Control forces $h$ only. IC: $h=0.3(\phi-0.5)$
with $\phi$ a 3-mode random field, $u=v=0$. Linearised so an explicit scheme is
stable at $64^2$; it keeps the hyperbolic/wave character that makes it a
different family from the two parabolic testbeds.

**`rdf` — FitzHugh–Nagumo front propagation** (2 channels $(u,v)$, $\Delta t=0.05$, 4 substeps):

$$
u_t = D_u\nabla^2u + u - u^3/3 - v + s,\qquad
v_t = D_v\nabla^2 v + \varepsilon(u + a - b v),
$$
$D_u=0.005,\ D_v=0.002,\ \varepsilon=0.25,\ a=0.7,\ b=0.8$. IC: $u=2\phi_3-1$,
$v=0.4\phi_2-0.2$.

**Stability at other resolutions.** Micro-steps scale as
$\lceil \text{base}\cdot (N/64)^2\rceil$ for $N>64$ (`stable_substeps`), because
the explicit diffusion number $\nu\,\delta t/\Delta x^2$ must stay bounded. This
was added after the resolution sweep returned NaN at $96^2$ and $128^2$ — the
*truth solver*, not the surrogate, diverged.

**Physics residual.** For a one-macro-step transition the residual is evaluated
at the midpoint so it is second-order in $\Delta t$:

$$
\mathcal{R}(u_t,u_{t+1},s_t)=\frac{u_{t+1}-u_t}{\Delta t}-f\!\Big(\tfrac{u_t+u_{t+1}}{2},\,s_t\Big),
$$

with $f$ the same `rhs` that generates the data. A forward-difference residual
would penalise a perfect surrogate at $O(\Delta t)$.

**Within-family parameter-shift protocols** (`pipeline.TRANSFER_PROTOCOLS`):
`dar`: $D=0.003,\ \mathbf v=(-0.5,0.45),\ r=1.0$; `swe`: $g=1.6,\ H=0.6,\ c=0.25$;
`rdf`: $D_u=0.010,\ D_v=0.001,\ \varepsilon=0.15,\ b=0.5$.

### 3.2 Dataset generation (`pspe/simulate/dataset.py`)

Trajectories are rolled out *with* randomly excited controls so the surrogate can
answer counterfactual actuation queries: amplitudes $a\sim0.5\,\mathcal N(0,I_K)$
are resampled every `control_hold`=4 steps and held. Defaults: 256 trajectories
× 24 steps, 80/20 train/val split by trajectory, windows of length `horizon`+1.
Datasets are cached per (testbed, grid) as `data/pde_testbeds/<testbed>_<grid>.npz`.

**Audit note.** The cached `dar_64.npz` used by every Vista run has **128**
trajectories (meta: `n_trajectories: 128`), not 256: `ensure_dataset` only
generates when the file is absent, and `--full` (which asks for 256) reused the
cached file. 128 × 24 = **3,072** transitions is the "real samples" figure in the
planner tables; `run_constraint_fix.py` counts states rather than transitions
(128 × 25 = **3,200**). Both refer to the same data.

### 3.3 Actuator basis (`pspe/envs/actuators.py`)

Actions are amplitudes on $K=9$ fixed isotropic Gaussians on a $3\times3$ lattice
inset from the edges (centres $(i+\tfrac12)/3$), width $\sigma=0.08$, periodic
distance, peak-normalised:

$$
s(x)=\Phi(a)(x)=a_{\max}\sum_{k=1}^{K}a_k\exp\!\Big(-\frac{d_{\text{per}}(x,c_k)^2}{2\sigma^2}\Big),\qquad a_{\max}=1 .
$$

$\Phi$ is linear and differentiable in $a$ — the pathwise gradient flows through it.

### 3.4 Reward and constraints (`pspe/envs/task.py`)

Per step, batched and differentiable, shared by planner, Gymnasium env and all
baselines:

$$
r(u,a)=-\Big[w_{\text{track}}\,\overline{(u_0-u^\star)^2}+w_{\text{effort}}\,\overline{a^2}\Big],
$$
$$
g_1(u,a)=w_{\text{exp}}\,\overline{\mathrm{relu}(u_0-u_{\max})}+w_{\text{bud}}\,\mathrm{relu}\big(\overline{|a|}-\text{budget}\big),
$$
$$
g_2(u,a)=w_{\text{eq}}\,\mathrm{Var}_{r\in\{1..R^2\}}\big(h_r\big),\qquad h_r=\text{mean over sub-region } r \text{ of }(u_0-u^\star)^2 ,
$$

where $\overline{\cdot}$ is the spatial mean and $u_0$ the tracked channel. A
trajectory is feasible when $\sum_t g_j\le d_j$. $g_2$ is zero iff every
sub-region carries equal residual harm; it is enforced by its own PID dual when
`equity.enabled=true` (off in all reported comparisons; $R=2$, $d_2=0.05$).

| testbed | tracked | $u^\star$ | $w_{\text{track}}$ | $u_{\max}$ | budget | $d_1$ (calibrated) |
|---|---|---|---|---|---|---|
| dar | $u$ | 0 | 1.0 | 0.55 | 0.35 | **0.936** |
| swe | $h$ | 0 | 4.0 | 0.04 | 0.03 | **0.192** |
| rdf | $u$ | −1 | 0.5 | 0.5 | 0.35 | **3.26** |

$w_{\text{effort}}=0.02$, $w_{\text{exp}}=w_{\text{bud}}=1$.

### 3.5 Constraint-limit calibration (`scripts/calibrate_constraints.py`)

A guessed limit silently turns a constrained comparison into an unconstrained tie
(the first Phase 2 table: every learner at cost ≈ 0.16 against $d=2.0$). The limit
is placed between two measured reference points:

$$
d_1 = c_0 + 0.35\,(c_{\text{greedy}}-c_0),
$$

$c_0$ the cost of doing nothing, $c_{\text{greedy}}$ the cost of a reward-greedy
policy trained with the dual disabled ($K_p=K_i=K_d=0$).

| testbed | $c_0$ | $c_{\text{greedy}}$ | old limit | calibrated |
|---|---|---|---|---|
| dar | 0.229 | 2.249 | 2.0 (0.9× greedy) | 0.936 |
| rdf | 0.594 | 8.213 | 2.0 (0.24× greedy) | 3.26 |
| swe | 0.108 | 0.347 | 1.0 (378× greedy, after threshold retune) | 0.192 |

`swe` also needed its thresholds retuned ($u_{\max}$ 0.12→0.04, budget 0.35→0.03)
because $|h|$ only reaches 0.113 at the 95th percentile and mean$|a|$ peaks at
0.051, so neither cost term ever fired. Recalibrate whenever reward weights, the
actuator basis or the horizon change.

### 3.6 Environments (`pspe/envs/pde_env.py`)

`BatchedFieldEnv`: batched, all-torch, keeps the autograd graph through
`surrogate(state, control)` so reward and cost are differentiable in the action.
`PDEControlEnv`: single-instance Gymnasium wrapper with `info["cost"]`
(Safety-Gymnasium/OmniSafe convention). Both share `TaskSpec` and the basis.

### 3.7 Real datasets

* **PDEBench 2D shallow water** (`2D_rdb_NA_NA.h5`, DaRUS id 133021, 6.6 GB;
  Takamoto et al. 2022): loaded into the `(states, controls)` schema with zero
  controls; evaluated against the recorded frames (`eval_vs_data=True`). 512
  trajectories, $64^2$, 20 epochs.
* **Next Day Wildfire Spread** (Huot et al. 2022; ~18k samples, 64 km patches at
  1 km, day-$t$ fire mask + 11 drivers → day-$t{+}1$ mask): pure-numpy TFRecord
  reader (no TensorFlow), `-1` unlabelled cells masked out of loss and metrics,
  robust (median/IQR) driver normalisation clipped at ±5. Fire prevalence 1.3%.
* **EuroSAT** (Sentinel-2 MSI → NDVI from RGB, NIR dropped): loader built and
  tested; a run exists only in the Colab notebook, no committed result.
* **Synthetic perception imagery** (`perceive/dataset.render_field`): field →
  three-band nonlinear response ($u^{0.7}, u, 1-u^{1.4}$) + separable Gaussian
  blur ($\sigma=1$) + shot noise (0.05) + vignetting; a lossy forward sensor model
  so inversion is a real inverse problem.

---

## 4. Method: the four modules, with the mathematics as implemented

### 4.1 Perceive (`pspe/perceive/`)

**Backbones.** `tiny`: an in-tree 4-layer patch transformer (patch 8, width 128)
— an offline stand-in, watermarked `backbone_is_stub: true`. `cnn`: a strided
conv encoder trained from scratch (the Section 7.2 control). Any HuggingFace
standard-ViT vision tower (used: `google/siglip-base-patch16-224`, hidden 768,
14×14 tokens; imagery resized to 224 and normalised to $[-1,1]$). Packed-patch
towers (Qwen2-VL, moondream2) are rejected with an actionable error.

**LoRA** (Hu et al. 2022), in-tree for the stub and via `peft` for HF towers:
$W x \to W x + \tfrac{\alpha}{r} B A x$, $A\in\mathbb R^{r\times d_{in}}$
Kaiming-initialised, $B=0$ at init so the adapted model equals the base; $r=8$,
$\alpha=16$; targets `q_proj,k_proj,v_proj,out_proj,qkv,proj,fc1,fc2`.
`assert_lora_only` raises if any non-LoRA backbone parameter is trainable.

**Decoder.** Tokens → linear → reshape to the token grid → bilinear upsample to
the field grid → two 3×3 convs → $C$ channels. **Projection head** on the mean
token for the contrastive term. **Text encoder**: frozen hashed bag-of-unigrams+bigrams
(blake2b → 512 bins) through a frozen random projection and $\tanh$, then a
trainable 2-layer head. Captions are synthesised from field statistics
("high concentration with peak in the north-east; mean 0.31, peak 0.82; mass
drifting east and steady").

**Loss** (paper Eq. 6):

$$
\mathcal L_{\text{perc}}=w_{\text{reg}}\,\frac{\|\hat u-u\|_2}{\|u\|_2}+w_{\text{con}}\,
\tfrac12\Big[\mathrm{CE}(\tau\,\hat Z_I\hat Z_T^{\top},\,\mathrm{I})+\mathrm{CE}(\tau\,\hat Z_T\hat Z_I^{\top},\,\mathrm{I})\Big],
$$

symmetric in-batch InfoNCE with $\ell_2$-normalised embeddings and learned
temperature $\tau=\exp(\text{logit\_scale})\le100$, $\log$-initialised at
$\ln(1/0.07)$; $w_{\text{reg}}=1$, $w_{\text{con}}=0.2$. Retrieval accuracy is
the in-batch argmax hit rate. Three arms share one decoder and dataset:

* **pspe**: frozen backbone + LoRA + contrastive (the method);
* **probe**: frozen backbone, no adapters, decoder only (isolates LoRA);
* **cnn**: from scratch, regression only.

The freeze-vs-fine-tune ablation (`train.freeze_encoder=false`) unfreezes the
backbone *with* adapters present, so it cannot isolate LoRA; the probe arm does.

### 4.2 Simulate (`pspe/simulate/`)

**FNO** (Li et al. 2021), in-tree. Input $[u_t,\,s_t,\,x,\,y]$ (state, control,
coordinate grid) → 1×1 lift to width $w=32$ → 4 blocks → 1×1 projection
($w\to2w\to C$) → **increment**: $G(u,s)=u+\text{net}(u,s)$ (`predict_delta`),
so the network learns the discrete time derivative the physics residual is
written in. Each block:

$$
v \leftarrow v + \rho\,\mathrm{GELU}\big(\mathrm{GN}(\mathcal K v + W v)\big),\qquad
(\mathcal K v)=\mathcal F^{-1}\big(R\cdot\mathbb 1_{|k|\le k_{\max}}\,\mathcal F v\big),
$$

with $R\in\mathbb C^{w\times w\times k_{\max}\times k_{\max}}$ for positive and
negative $k_x$ blocks (rfft2 keeps $k_y\ge0$), $k_{\max}=12$ modes per axis. The
spectral truncation is the "reduced order": memory and FLOPs scale with
$k_{\max}^2$ rather than $N^2$. The spectral path runs in float32 regardless of
autocast (ComplexHalf has no CUDA einsum kernel) and falls back to CPU on MPS.
1.19 M parameters at $C=1$.

**Lipschitz mode** (`lipschitz=True`, paper Assumption 1): spectral normalisation
(Miyato et al. 2018) on every 1×1 conv; per-mode SVD clipping of $R$ to
operator norm ≤ 1 after every optimiser step (under the orthonormal FFT the
spectral layer is block-diagonal across modes, so its operator norm is exactly
the largest per-mode spectral norm — the clip is tight, not a proxy); GroupNorm
removed (normalisation layers are not Lipschitz); residual scale $\rho=0.1$
(a residual block is $(1+L_{\text{branch}})$-Lipschitz, so $n$ blocks at branch
norm 1 would be $2^n$). The scale makes $L_G\le1$ *possible*; only the
measurement says whether it was *achieved*.

**Lipschitz estimation.** Power iteration on $J^\top J$ of the map
$u\mapsto G(u,s)$ at sampled states via `torch.func.jvp`/`vjp`, 20–30 iterations;
returns $\sigma_{\max}(J)$ (Rayleigh quotient). Verified to recover $k$ for a
linear map $ks$ to $10^{-4}$.

**Gradient checkpointing** (`use_checkpoint`): recompute each block in backward;
verified to give identical gradients; off by default at $64^2$.

**DeepONet** (Lu et al. 2021): conv branch (state+control → $p\cdot C$
coefficients, $p=64$), MLP trunk on coordinates + Fourier features (frequencies
1,2,4,8), output $\langle\beta_c,\tau(x)\rangle/\sqrt p$ + bias; 0.09 M params.
**GNOT** (Hao et al. 2023) at testbed scale: patchify (4×4) state+control+coords
into tokens, 4 pre-norm linear-attention blocks (elu+1 feature map, 4 heads,
width 96), unpatchify; 0.31 M params. All three share `forward(state, control)`,
so trainer, physics residual and planner are identical.

**Channel-padded surrogate** (`multifamily.py`): the core always sees
`MAX_CHANNELS`=3 state channels plus a validity mask channel (value
$C/3$) riding on the control stack, so one head trains on `dar` and rolls out on
`swe`/`rdf` with no weight change. Without the mask, "this channel is zero" and
"this channel does not exist" are indistinguishable.

**Loss** (paper Eq. 7), with autoregressive unroll of length $H_{\text{tr}}=4$:

$$
\mathcal L_{\text{sim}}=w_d\,\mathrm{relL}_2\big(G(u_0,s_0),u_1\big)
+\frac{w_r}{H_{\text{tr}}}\sum_{t=0}^{H_{\text{tr}}-1}\mathrm{relL}_2(\hat u_{t+1},u_{t+1})
+\frac{w_p}{H_{\text{tr}}}\sum_{t=0}^{H_{\text{tr}}-1}\big\|\mathcal R(\hat u_t,\hat u_{t+1},s_t)\big\|_2^2 ,
$$

$\hat u_0=u_0$, $\hat u_{t+1}=G(\hat u_t,s_t)$ with the graph kept through the
unroll, $\mathrm{relL}_2(p,q)=\mathbb E_b\|p-q\|_2/\|q\|_2$, weights
$w_d=1,\ w_r=1,\ w_p=0.1$; `use_physics=false` sets $w_p=0$. AdamW, lr $10^{-3}$,
wd $10^{-5}$, grad-clip 1.0, batch 16, 20 epochs. Evaluation: 16-step rollout
against the solver (or stored frames), reporting 1-step / final / mean rel $L_2$.

**Resolution generalisation** (`rollout.resolution_generalization`): the truth
solver is run *at each target grid*; the surrogate is scored against the physics
at that resolution, not a resampled $64^2$ field.

### 4.3 Plan (`pspe/plan/`)

**Policy.** `FieldEncoder` (3 strided convs, GELU, adaptive pool 2×2, linear →
128) → mean head; state-independent $\log\sigma$ (init −0.7, clamped to
$[-5,1]$). Action $a=\tanh z$, $z\sim\mathcal N(\mu(u),\sigma)$, reparameterised.
Log-density with the tanh change of variables:

$$
\log\pi(a\mid u)=\sum_i\Big[\log\mathcal N(z_i;\mu_i,\sigma_i)-2\big(\log2-z_i-\mathrm{softplus}(-2z_i)\big)\Big],
$$

the second term being $\log(1-\tanh^2 z_i)$ in a numerically stable form.
**Critic**: same encoder with reward-value and cost-value heads.

**Rollout.** Each iteration rolls $B=16$ episodes of $H=12$ steps inside the
surrogate with the graph kept, recording $r_t$, $g_t$, $\log\pi(a_t|u_t)$,
entropy. Lagrangian per-step reward $\ell_t=r_t-\lambda g_t\,(-\lambda_2 g_{2,t})$.

**Two per-sample losses from the same sampled actions** (the same $z$ feeds both
branches, so they estimate the same expectation):

* pathwise (reparameterisation; Heess et al. 2015 SVG-style):
  $\mathcal L^{\text{pw}}_b=-\sum_t\ell_{t,b}-c_H\sum_t\mathcal H_{t,b}$,
  differentiated through $G_\theta$ and $\Phi$;
* likelihood ratio (REINFORCE, Williams 1992) with critic baseline and
  batch-normalised advantage:
  $\mathcal L^{\text{lr}}_b=-\sum_t\log\pi(a_{t,b}|u_{t,b})\,\hat A_{t,b}-c_H\sum_t\mathcal H_{t,b}$,
  $\hat A_t=\big(R_t-V(u_t)\big)$ standardised, $R_t=\sum_{k\ge t}\gamma^{k-t}\ell_k$,
  $\gamma=0.98$, $c_H=10^{-3}$.

The critic regresses $V$ on $R_t$ and $V_C$ on discounted cost-to-go (MSE, Adam
$10^{-3}$).

**Hybrid estimator** (`hybrid_gradient.py`): $g_\alpha=\alpha g_{\text{pw}}+(1-\alpha)g_{\text{lr}}$,
flat over policy parameters, clipped to norm 10, Adam lr $3\times10^{-4}$.
Every `estimate_every`=5 steps the batch is split into 4 folds; each estimator's
gradient is computed per fold; $V_{\text{pw}},V_{\text{lr}},\mathrm{Cov}$ are the
traces of the fold-wise empirical (co)variances; then

$$
\alpha^\star=\mathrm{clip}_{[0,1]}\frac{V_{\text{lr}}-\mathrm{Cov}}{B^2+V_{\text{pw}}+V_{\text{lr}}-2\,\mathrm{Cov}},
\qquad \alpha\leftarrow0.9\,\alpha+0.1\,\alpha^\star .
$$

With $B^2=0$ this is the variance-only rule that shipped; with $\mathrm{Cov}=0$ it
is the paper's Eq. 8 $\alpha^\star=V_L/(B^2+V_p+V_L)$ (Section 5.2 gives the
derivation). `adaptive_alpha=false` pins $\alpha=0.5$.

**Measured pathwise bias** (`eq8_alpha=true`): the true solver is differentiable,
so at each probe iteration the pathwise gradient of the Lagrangian is taken from
identical start states and identical policy noise through the surrogate and
through the truth; $B^2=\|g^{\text{surr}}_{\text{pw}}-g^{\text{true}}_{\text{pw}}\|^2$,
EMA-smoothed (0.7/0.3). The paper leaves $B=C\varepsilon$ with an unknown constant;
here it is measured. Costs one extra differentiable truth rollout per probe,
counted in real samples.

**PID-Lagrangian dual** (Stooke et al. 2020; `lagrangian.py`), per constraint:

$$
\bar c_k=\eta\,\bar c_{k-1}+(1-\eta)\,c_k,\quad e_k=\bar c_k-d,\quad
I_k=\max(0,I_{k-1}+e_k),\quad D_k=\max(0,\bar c_k-\bar c_{k-1}),
$$
$$
\lambda_k=\mathrm{clip}_{[0,\lambda_{\max}]}\big(K_p e_k+K_i I_k+K_d D_k\big),
$$

$\eta=0.9$, $\lambda_{\max}=50$, one-sided derivative (anti-windup). Gains
$K_p=0.5,\ K_i=0.05,\ K_d=0.1$ — calibrated to the *reward* gradient: the
original $(0.05,\,5\times10^{-4},\,0.02)$ let $\lambda$ peak at 0.0475 while the
episode cost ran to 1.34 against 0.936, i.e. an unconstrained planner.

**Closing the surrogate/reality gap on the constraint** (the safety fix):

* *Probe* (`real_cost_every`=20, 8 episodes): roll the *stochastic* current
  policy in the true environment on a forked RNG stream; feed the measured real
  episode cost to the dual on probe iterations; between probes feed
  $c^{\text{surr}}+\hat b$ where $\hat b$ is an EMA (0.7/0.3) of
  $(c^{\text{real}}-c^{\text{surr}})$ and $\hat v$ an EMA of its squared deviation.
* *Margin* (`cost_margin_k`=2): plan against $d_{\text{eff}}=\max(0,\,d-k\sqrt{\hat v})$.

Probe transitions are added to `samples_real_env`. Without a probe there is no
measured error, so the margin arm alone is identical to the baseline.

**Evaluation** is on the numerical dynamics with the deterministic (mean) action,
8 episodes from a fixed IC set (`seed+10_000`), every 20 iterations and at the
end, on a forked RNG. Run-level constraint statistics (`constraint_summary`):
`violating_eval_fraction`, `cost_max_over_run`, `cost_mean_over_run` over all
evaluations (11 per 200-iteration run).

### 4.4 Explain (`pspe/explain/`)

**Brief grammar** (`brief.py`), closed vocabulary, parseable by construction:

```
step <t> . [actuator <i> at <x> <y> set to <±a> to {increase|reduce} the field .]*
predicted cost <c> is {within|above} the limit <d> . expected reward <R> .
confidence <word> .
```

Amplitudes are quantised to 0.05; every actuator above the deadband is named
(earlier top-3 truncation capped $F$ near 0.33). Confidence words map
inverse-consistently to representative stds via buckets
{confident 0.10, likely 0.22, moderate 0.37, uncertain 0.55, vague 0.80}.
Vocabulary ≈ 900 tokens (numbers on the quantisation grid, cost/reward at one
decimal, centres, words).

**Frozen parser** (`parser.py`): regex → $\hat\pi_b=\mathcal N(m_b,\,s_b\mathbf 1)$,
$m_{b,i}$ from "actuator i … set to a" (0 if unmentioned), $s_b$ from the
confidence word (default 0.4). No parameters — a learned parser could agree on a
private code with the generator.

**Conditioning**: $[\mu,\ \log\sigma,\ \bar u,\ \max u,\ c,\ d,\ r]\in\mathbb R^{2K+5}$
→ 2-layer MLP → 4 prefix embeddings prepended to token embeddings.

**Generator.** `tiny`: 4-layer word-level causal transformer, LoRA on attention
and MLP, embedding/head trainable, body frozen. HF path (`Qwen/Qwen2.5-0.5B-Instruct`,
4-bit NF4 with `prepare_model_for_kbit_training`, LoRA $r=8$ on `q_proj,v_proj`):
the LM is used as a **frozen sequence encoder** over the brief vocabulary — a
trainable brief embedding and a trainable output head over the ~900-token closed
vocabulary replace the LM's 150k-token embedding and head, the LM contextualises
in between. Trainable: 2.41 M of 496 M parameters. This is a design point the
paper must state: the LM's own language head is never used.

**Reference distribution.** Since the parser reads post-tanh amplitudes, the
policy distribution is expressed as $\pi=\mathcal N(\tanh\mu,\ \sigma)$ (post-tanh
mean, pre-tanh std — an approximation the code comments).

**Faithfulness** (paper Eq. 12, modified):

$$
F(b)=\exp\!\big(-D_{\mathrm{KL}}(\pi\,\|\,\hat\pi_b)\big),\qquad
D_{\mathrm{KL}}=\sum_{i=1}^{K}\Big[\log\frac{s_b}{\sigma_i}+\frac{\sigma_i^2+(\tanh\mu_i-m_{b,i})^2}{2s_b^2}-\frac12\Big].
$$

The proposal wrote $F=1-\mathrm{KL}$; the code uses $\exp(-\mathrm{KL})$
deliberately: bounded in $(0,1]$, monotone in KL, first-order equal at small KL,
and it stops one catastrophic sample from dominating a REINFORCE batch.
$F_{\text{ref}}$, the score of the rendered reference brief itself, is the
ceiling given quantisation and std bucketing (≈0.90).

**Objective** (paper Eq. 11):

$$
\mathcal L_{\text{exp}}=w_s\,\mathrm{NLL}(b^{\text{ref}}\mid\text{cond})
-\,w_f\,\mathbb E_{b\sim p_\phi(\cdot|\text{cond},T=0.3)}\Big[\big(F(b)-F(b^{\text{greedy}})\big)\log p_\phi(b)\Big],
$$

supervised NLL toward the reference brief (per-token mean) plus REINFORCE with a
**self-critical baseline** (Rennie et al. 2017): the greedy decode's own $F$.
Samples are drawn at temperature 0.3 without grad and scored with one
teacher-forced forward (`sample_and_score`), which is what makes a 0.5 B backbone
fit. $w_s=w_f=1$, AdamW $10^{-3}$, grad-clip 5, 200 iterations, batch 8 in the
baseline runner (16 in the Hydra config). A batch-EMA baseline at $T=0.8$ gave *exactly zero* gradient on both
backbones: no sample parsed, every score equalled the parser fallback. Parse rate
is logged as the starvation diagnostic.

**Post-hoc arm** (`posthoc=True`): `iterations = 0`. See Section 12.1.

**Split-conformal certificate** (`conformal.py`, paper Prop. 3). Nonconformity
$s=1-F\in[0,1)$. With $n$ calibration scores and one test score exchangeable,

$$
\hat s=s_{(\lceil (n+1)(1-\delta)\rceil)},\qquad
\mathbb P\big[F(b_{\text{new}})\ge 1-\hat s\big]\ge 1-\delta .
$$

The $+1$ is what makes the guarantee finite-sample exact. If
$\lceil(n+1)(1-\delta)\rceil>n$ the quantile is $+\infty$ and the floor is 0 (the
honest vacuous bound); minimum $n=\lceil(1-\delta)/\delta\rceil$ (9 at
$\delta=0.1$, 19 at 0.05). The module refuses to tune anything on the
calibration set and always reports empirical coverage on a disjoint test set,
with a one-sided binomial $p$-value (a one-std tolerance rejected a correct
certificate 16% of the time by construction); `holds` ⇔ $p\ge0.05$.

### 4.5 Joint Simulate + Plan training (`pspe/plan/joint.py`)

The paper's contribution #1 ("trained under a single objective"). `JointPlannerTrainer`
subclasses the planner; while the rollout graph is alive it takes
$\nabla_\theta\mathcal L^{\text{pw}}$ w.r.t. the *surrogate* parameters, and after
the policy step applies

$$
\theta\leftarrow\theta-\eta_\theta\Big[\nabla_\theta\mathcal L_{\text{data}}
+\beta\,\frac{\|\nabla_\theta\mathcal L_{\text{data}}\|}{\|\nabla_\theta\mathcal L^{\text{pw}}\|}\,\nabla_\theta\mathcal L^{\text{pw}}\Big],
$$

$\mathcal L_{\text{data}}$ a relative-MSE on a 32-transition minibatch from an
anchor split (90% of trajectories), $\beta=0.1$ ("the planner moves the surrogate
one tenth as hard as the data"), lr $10^{-4}$, grad-clip 1, spectral projection
if in Lipschitz mode. The norm scaling exists because the raw planning loss is
$O(1)$ and the data loss $O(10^{-4})$: at raw $\beta=0.1$ the anchor was swamped
within 8 iterations and held-out error rose $10\times$. `anchor=False` applies the
raw planning gradient at $\beta$ — the failure mode run on purpose. A *fixed*
held-out batch (10% of trajectories, split by trajectory) is scored before and
after; `surrogate_drift` = after − before is the capture diagnostic. Probe + margin
stay on so the dual is driven by real cost regardless of what the surrogate claims.

### 4.6 End-to-end loop and transfer protocols (`pspe/pipeline.py`)

`PSPEPipeline.rollout`: true field → rendered image → Perceive → estimate
(tracked channel replaced, other channels passed through) → policy mean action →
true env step → brief → $F(b)$; per-step records of perception rel $L_2$, reward,
cost, faithfulness. `runs/e2e/results.json` (smoke budget, stub backbones):
e2e return −2.29, cost 0.229/2.0, perception rel $L_2$ 0.170, $F$ 0.56.

`transfer_gap`: rel $L_2$ (8-step, zero controls) on target minus source;
cross-family needs a padded surrogate and uses the *target* family's ICs
(a `dar` IC is not a valid shallow-water state); parameter shift shares ICs.
`planning_transfer_gap`: train a policy against the transferred surrogate while
the true dynamics are the target family, and one against the target's own truth;
both evaluated on the target truth; gap = native − transferred return
(60 iterations, horizon 12).

---

## 5. Theory: assumptions, propositions and their measured status

The proposal PDF (`pspe_iclr.pdf`) is not in the repository; the statements below
are reconstructed from the code, its docstrings and `docs/`. Where the exact
paper statement may differ, the form actually *tested* is given.

### 5.1 Assumption 1 — the surrogate is 1-Lipschitz

$\|G_\theta(u,s)-G_\theta(u',s)\|\le L_G\|u-u'\|$ with $L_G\le1$, to be
enforced by spectral normalisation. **Status:** the default FNO (GroupNorm,
unscaled residuals) has $L_G=1.61$ at initialisation and **0.985 ± 0.013 after
training** (≤ 1 on 4 of 5 seeds; per-seed 0.991, 0.985, 1.002, 0.967, 0.980);
Lipschitz mode gives 0.964 ± 0.003 on all 5 seeds for +0.01 rel $L_2$
(0.079 vs 0.069). Measured as the local constant (largest Jacobian singular
value at sampled states), the honest computable proxy for the global sup.

### 5.2 Proposition 1 — return bias of planning through the surrogate

Let $\varepsilon=\sup\|G(u,s)-F(u,s)\|$ be the one-step calibration error and
$L\le1$ the surrogate's Lipschitz constant. With reward 1-Lipschitz in the state,
the state error after $t$ steps under the same action sequence obeys
$\|\hat u_t-u_t\|\le\varepsilon\sum_{k<t}L^k\le t\varepsilon$, so
$|r(\hat u_t)-r(u_t)|\le t\varepsilon$ and

$$
|J_G(\pi)-J_F(\pi)|\;\le\;\varepsilon\sum_{t=0}^{\infty}\gamma^t t=\frac{\gamma}{(1-\gamma)^2}\,L\,\varepsilon
\qquad\text{(infinite horizon)},
$$
$$
|J_G(\pi)-J_F(\pi)|\;\le\;\varepsilon\sum_{t=0}^{H-1}\gamma^t t\qquad\text{(horizon }H\text{, same proof truncated)}.
$$

If $L_G>1$ the recursion compounds geometrically and the bound's premise fails.
**Status** (`runs/lipschitz/`, 5 seeds, $\gamma=0.98$, $H=12$, $\varepsilon$ = max
one-step error over 16 episodes × 12 steps under a fixed stochastic policy):

| arm | rel $L_2$ | $L_G$ | $\varepsilon_{\max}$ | measured bias $\lvert J_G-J_F\rvert$ | bound, $\infty$ | bound, $H$ | holds ($H$) |
|---|---|---|---|---|---|---|---|
| default FNO | 0.069 ± 0.062 | 0.985 ± 0.013 | 0.172 ± 0.12 | 0.101 ± 0.092 | 422 ± 290 | 9.8 ± 6.6 | 5/5 |
| Lipschitz mode | 0.079 ± 0.026 | 0.964 ± 0.003 | 0.305 ± 0.081 | 0.106 ± 0.041 | 748 ± 200 | 17.3 ± 4.6 | 5/5 |

The infinite-horizon prefactor at $\gamma=0.98$ is 2,450, so the bound is
$\sim4{,}000\times$ loose (vacuous). The finite-horizon form holds on every seed and
is $\sim100\times$ loose. Report the finite-horizon form and say it is loose.

### 5.3 Theorem 1 / Eq. 8 — variance-optimal mixing

Let $g_{\text{lr}}$ be unbiased with variance $V_L$ and $g_{\text{pw}}$ have bias
$B$ and variance $V_p$ (with $B=C\varepsilon$ in the paper). For
$g_\alpha=\alpha g_{\text{pw}}+(1-\alpha)g_{\text{lr}}$,

$$
\mathrm{MSE}(\alpha)=\alpha^2(B^2+V_p)+(1-\alpha)^2V_L+2\alpha(1-\alpha)\,\mathrm{Cov},
$$
$$
\alpha^\star=\frac{V_L-\mathrm{Cov}}{B^2+V_p+V_L-2\,\mathrm{Cov}}\;\;\xrightarrow{\ \mathrm{Cov}=0\ }\;\;\frac{V_L}{B^2+V_p+V_L}\quad\text{(Eq. 8)},
$$

so $\alpha\to1$ as the score-function variance dominates and $\alpha\to0$ as
the pathwise bias grows. The implementation keeps the covariance term (the two
estimators share sampled actions and are correlated). **Status**
(`runs/alpha_rule/`, 5 seeds, probe + margin on in all arms): measured
$B^2=0.0013 \pm 0.0015$, negligible against $V_p+V_L$, so Eq. 8 and the
variance-only rule choose the same $\alpha$ (0.992); return $t=+0.22$. Eq. 8 was
the only arm with 0 violating evaluations on all 5 seeds, at 1,920 extra truth
transitions for the bias measurement — a tail effect of the extra rollouts, not
of the formula. At full budget $\alpha$ saturates (0.9917 ± 0.0006) versus 0.734
at smoke budget, consistent with $V_L/V_p\to\infty$ as the surrogate gradients
become reliable; the smoke-scale variance ratio was measured at ≈43,000.

### 5.4 Corollary 1 — PID-Lagrangian drives time-averaged violation to a neighbourhood of zero

**Status:** holds on the closed-form CMDP (Section 6.2) and *did not hold* on
`dar` at 200 iterations as originally built (7.3% of evaluations violating,
worst 1.109 vs 0.936; fixed-$\alpha$: 12.7%, worst 2.758, one seed diverging to
cost 8.1 with $\lambda=19.1$). The reason is structural, not a bug: the dual is
controlled on **surrogate** cost while violation is realised on the **true**
dynamics; where the surrogate under-predicts cost the policy is safe in-model and
unsafe in reality. With probe + margin: 13 of 15 seed-runs clean, ~1% of
evaluations violating (two single-evaluation excursions at 0.98 and 1.71). A
$7\times$ reduction, not a guarantee; a guarantee would need a margin from a
conformal quantile of the measured cost error rather than $2\sigma$.

### 5.5 Proposition 3 — conformal faithfulness certificate

Split-conformal coverage as in Section 4.4; needs only exchangeability of
(state, brief) pairs under a fixed policy, nothing about the LM or parser.
**Status** (`runs/conformal_real/`, Qwen2.5-0.5B, 3 seeds, $n_{\text{cal}}=200$,
$n_{\text{test}}=400$):

| seed | mean $F$ | $\delta=0.2$: floor / coverage / holds | $\delta=0.1$ | $\delta=0.05$ |
|---|---|---|---|---|
| 0 | 0.281 | 0.278 / 0.798 / yes | 0.277 / 0.878 / yes ($p=0.08$) | 0.276 / 0.928 / **no** ($p=0.03$) |
| 1 | 0.557 | 0.557 / 0.793 / yes | 0.556 / 0.893 / yes | 0.556 / 0.940 / yes |
| 2 | 0.567 | 0.562 / 0.773 / yes | 0.559 / 0.895 / yes | 0.558 / 0.935 / yes |

Holds at 80% and 90% on all seeds; one seed fails at 95%. The certified floor
tracks the generator (0.28 on the weak seed, 0.56 on the others): it is honest
about a weak generator rather than hiding it.

### 5.6 Closed-form CMDP used to validate the dual machinery

One step, scalar $a\in(-1,1)$, $r(a)=-(a-p)^2$, $c(a)=\mathrm{relu}(a)$,
$\mathbb E c\le d$. Unconstrained optimum $a^\star=p$; constrained
$a^\star=\min(p,d)$; $\lambda^\star=|dR/da|/|dC/da|=2(p-d)$ at the boundary.
With $p=0.8,\ d=0.4$: $a^\star=0.4,\ \lambda^\star=0.8$. $p$ is interior to the
tanh range on purpose (at $p=1$ the pre-tanh mean runs away and the gradient dies
for reasons unrelated to constraint handling).

---

## 6. Baselines and their validation

### 6.1 Safe-RL baselines (`baselines/safe_rl.py`)

OmniSafe does not install against Python 3.11+/gymnasium ≥ 0.29, so all four are
in-tree reimplementations sharing the env, reward, cost, policy architecture,
horizon, batch (16 × 12) and evaluation protocol with the planner. All are
model-free: they see detached transitions of the true solver. Shared skeleton:
on-policy collection → GAE ($\gamma=0.98,\ \lambda=0.95$) for reward and cost →
algorithm-specific update → critic regression (4 inner epochs). 200 iterations
= 38,400 real transitions.

* **PPO-Lagrangian** (Stooke et al. 2020): clipped surrogate ($\epsilon=0.2$, 4
  epochs, early stop at $1.5\times$ target KL 0.01) on the blended advantage
  $\hat A=(\tilde A_r-\lambda\tilde A_c)/(1+\lambda)$ where $\tilde A_r,\tilde A_c$
  are standardised **separately**; PID dual with gains $(0.5,0.05,0.1)$.
* **CPO** (Achiam et al. 2017): $g=\nabla J_R$, $b=\nabla J_C$ (cost advantage
  centred, **not** scaled), Fisher-vector products via the KL Hessian
  (damping 0.1, 10 CG iterations), constraint surplus in per-step units
  $c=(J_C-d)/H$, trust radius $\delta=0.01$. Dual solution
  $\lambda=\sqrt{(q-r^2/s)/(2\delta-c^2/s)}$, $\nu=\max\big((\lambda c-r)/s,0\big)$,
  step $(v-\nu w)/\lambda$; TRPO step when the cost gradient vanishes or the
  constraint is slack and the TRPO step keeps it so; recovery step
  $-\sqrt{2\delta/s}\,w$ when infeasible; clamps on the Cauchy–Schwarz numerator
  and degenerate denominator; backtracking line search (10 steps, 0.8) accepting
  only if KL ≤ $\delta$, objective improves (or recovery) and surrogate cost grows
  at most by the current slack.
* **Sauté RL** (Sootla et al. 2022): remaining budget $z_{t+1}=z_t-c_t$ appended
  as a constant observation channel (normalised by $d$); unsafe ($z<0$) states
  absorb with reward 0; unconstrained PPO on the augmented MDP. Rewards are
  shifted by a running floor before the absorbing rule (all PSPE rewards are ≤ 0).
* **Primal-dual NPG**: natural gradient (CG on the Fisher) on the standardised
  Lagrangian advantage with step $\sqrt{2\delta/(g^\top F^{-1}g)}$; projected dual
  ascent $\lambda\leftarrow\max(0,\lambda+0.02\,(J_C-d))$.

### 6.2 Validation against the closed-form CMDP (`tests/test_safe_rl_correctness.py`)

Two levels: **analytic** (CG vs explicit inverse; KL-HVP vs the analytic Gaussian
Fisher; NPG step vs its trust radius; PID terms vs hand arithmetic; CPO's
post-update KL vs `target_kl`) and **behavioural** (all four must reach
$a^\star=0.4$, $\lambda\to0.8$). Only primal-dual NPG was correct as first
written. Four bugs found, each of which would have corrupted the Phase 2 table
while producing plausible curves:

| algorithm | bug | symptom | fix |
|---|---|---|---|
| Sauté | zero reward in the absorbing state with rewards ≤ 0 | blowing the budget was the highest-reward outcome; converged to the unconstrained optimum | shift rewards by a running floor first |
| PPO-Lag | standardised the *combined* advantage | rescales away $\lambda$'s magnitude; dual chases a target it cannot move; policy oscillates to the bounds | standardise $A_r$, $A_c$ separately; blend as $(A_r-\lambda A_c)/(1+\lambda)$; raise dual gains |
| CPO | cost advantage scaled to unit variance | $b$ in different units from $c$; persistent 0.11 action bias from 300 to 900 iterations | centre only; $c$ in per-step units |
| CPO | no slack branch | $2\delta-c^2/s<0$ when comfortably feasible → $\lambda$ explodes, step collapses; CPO froze exactly when safe | reduce to TRPO when slack; clamp instead of `sqrt` of a negative |

After the fixes (closed form $a^\star=0.400$, $\lambda^\star=0.800$): CPO 0.402
(0.0001 error at 600 iterations), PPO-Lag 0.410 ($\lambda=1.007$), PD-NPG 0.439
($\lambda=0.878$), Sauté 0.491 (expected: state augmentation converges to a
conservative interior point, cost 0.309 vs limit 0.400). Before the CPO fixes it
sat at 0.519 with a 26% steady-state violation that would have read as "CPO is
weak" rather than "CPO is misimplemented".

`baselines/validate_safety_gym.py` (SafetyPointGoal1-v0 against published
trends) exists but **has not been run** (`safety-gymnasium`'s pygame wheel fails
on macOS/arm64). The toy CMDP is a correctness canary, not a scale test.

### 6.3 Operator baselines

DeepONet and GNOT (Section 4.2) under the identical trainer, losses and
evaluation. `tests/test_fno_canary.py` trains the production `SpectralConv2d` on
1D Burgers ($\nu=0.1$), where Li et al. report ~$10^{-3}$, and asserts a loose
bound separating "works" from "broken" (rel $L_2$ 0.008 observed).

### 6.4 Perception baselines

Frozen probe and from-scratch CNN (Section 4.1). The zero-shot VLM arm the
proposal lists (prompt a frozen instruction-tuned VLM for the field statistic) is
**not implemented**.

### 6.5 Explanation baselines

`no-faithful` (supervised only) and `post-hoc` (Section 4.4; see Section 12.1 for
what it actually is). A human Likert study harness exists (`eval/human_rating.py`:
correctness / completeness / actionability, raters blinded to reference and $F$)
but has no participants and no ratings.

### 6.6 Fire-spread baselines (`pspe/simulate/real/models.py`)

U-Net (width 32, depth 3, GroupNorm, 1.63 M params — the published-baseline
architecture class), FNO alone (+1×1 logit head, 1.19 M), and the hybrid
`HybridFNOUNet`: U-Net logits + $g\cdot\text{FNO}(x)$ with a learned scalar gate
$g$ **initialised at zero** (so it starts exactly at U-Net behaviour) + a learned
logit bias on the day-$t$ mask. An earlier equal-partner fusion scored worse than
its own U-Net half (0.220 vs 0.280) at $4\times$ compute; the gate makes "did the
operator help" a measurement. Training: BCE-with-logits, positive weight
$\sqrt{1/\text{prevalence}}\approx9$ (not $1/\text{prevalence}\approx77$, which
made the model predict fire everywhere), AdamW + cosine, unlabelled targets
masked, optional flips with the wind-direction channel sign-corrected. Metric:
AUC-PR (step-wise average precision, written out so the convention is explicit)
against the prevalence floor 0.011–0.013.

---

## 7. Experimental protocol and instrumentation

* **Seeds and statistics.** 5 seeds (3 for real-backbone runs), mean ± sample
  std, paired $t$ by seed on 4 df (2 df for 3 seeds); $|t|>2.78$ ⇒ $p<0.05$ on
  4 df. Arms share initial conditions, data and seed schedule. The reported
  spread is training stochasticity (init, minibatch order, policy sampling), **not**
  dataset resampling (datasets are cached per testbed/grid).
* **Budgets.** "Full": 20 surrogate epochs, 200 planner iterations, 200 baseline
  iterations, 10 perceive epochs, 200 explain iterations, $64^2$, horizon 12, 9
  actuators. "Quick": 3 / 30 / 30 / 3 / 30 at 64 trajectories. The
  `constraint_fix`, `alpha_rule`, `joint`, `lipschitz` and `faith_weights` runs
  were launched with `--epochs 20 --iterations 200` (paper-scale); the "quick
  budgets" string in their headers is the aggregator's default label.
* **Hardware.** NVIDIA GH200 nodes on TACC Vista, one seed per array task or all
  seeds packed on one node (~2 GB each of 96 GB). A full 5-seed Phase 2 sweep:
  ~9.5 min on 5 nodes vs ~9 h estimated on a laptop CPU. Explain on Qwen: ~20 min
  per arm per seed, peak 3.3–5.6 GB. Every `summary.json` records wall-clock and
  peak memory. CPU costs on an M-series laptop: data ~1 min/testbed, Phase 1 ~4
  min/epoch at $64^2$, test suite ~4 min.
* **Runner.** `eval/run_seeds.py` reruns `run_ablations.py` once per seed in a
  fresh process (seed set before any module touches the RNG), resumable,
  aggregated by `aggregate_seeds` into `results_seeds.{md,json}` with per-seed
  values retained. Per-experiment runners: `run_constraint_fix.py`,
  `run_alpha_rule.py`, `run_joint.py`, `run_lipschitz.py`, `run_conformal.py`,
  `run_explain_baselines.py`, `run_perception_baselines.py`, `run_transfer.py`,
  `run_resolution.py`, `run_pdebench.py`, `run_ndws.py`.
* **Tests.** 139 test functions across 17 files (7 parametrised; the docs quote
  156 collected), including regression tests for each measurement bug below, the
  closed-form CMDP checks, gradient-checkpoint identity, the Lipschitz estimator
  on a known map, conformal coverage on skewed distributions, eval-schedule
  independence, and the Eq. 8 bias plumbing.

### 7.1 Measurement bugs found and fixed (each invisible at single-seed scale)

1. **Constraint satisfaction read from one final evaluation.** Cost excursions
   decay within an evaluation interval; two identical runs were recorded as
   "violation 0.0" and "1.0" on snapshot timing. Now `violating_eval_fraction`
   and `cost_max_over_run` over all evaluations, for the planner and all
   baselines on the same schedule.
2. **Evaluation perturbed training.** `env.step` draws from the global RNG;
   adding periodic evaluation moved baseline returns from −2.51 to −2.30. Both
   evaluators fork the RNG and use a fixed IC set; `tests/test_baselines_arms.py`
   asserts eval-schedule independence; the probe uses its own stream for the
   same reason.
3. **Optional-dependency tests skipped silently.** `pytest.importorskip("h5py")`
   with `h5py` undeclared: the PDEBench loader path was green and failed on first
   GPU contact. Now declared.
4. **Zero-gradient faithfulness term.** At $T=0.8$ no REINFORCE sample parsed;
   all scores equalled the fallback; gradient exactly zero on stub and Qwen. Fixed
   with $T=0.3$ and the self-critical baseline (parse rate → 100%, $|\text{reinforce}|\approx0.02$–$0.04$).
5. **Truth solver NaN above $64^2$** (Section 3.1).
6. **Sample accounting** once counted the planner's 38,400 surrogate rollouts as
   env samples, hiding the model-based argument.
7. **NDWS pipeline**: unnormalised drivers spanning four orders of magnitude, a
   class weight of 77, and `predict_delta` across a 12-channel input stack —
   together most of the gap between the first run (0.035 AUC-PR; 0.073 in
   `runs/ndws/`) and 0.277.

---

## 8. Results

All tables: mean ± sample std over seeds; $t$ = paired $t$ by seed. Sources are
the `runs/*/results_seeds.json` files; per-seed values are quoted where they change
the reading.

### 8.1 Surrogate accuracy on PDEBench 2D shallow water (single seed)

512 real trajectories, $64^2$, 20 epochs, rollout scored against recorded frames.

| operator | rel $L_2$ 1-step | rel $L_2$ rollout | params | published FNO (nRMSE, Takamoto et al.) |
|---|---|---|---|---|
| **FNO** | **0.0018** | **0.0098** | 1.19 M | 0.0044 |
| DeepONet | 0.0044 | 0.0137 | 0.09 M | — |
| GNOT | 0.0056 | 0.0543 | 0.31 M | — |

Rollout error ~2.2× the published figure, one-step below it; relative $L_2$ vs
nRMSE are not identical metrics — an order-of-magnitude anchor. The ranking
FNO ≫ DeepONet ≫ GNOT reproduces the synthetic result. **Provenance:** the raw
`runs/pdebench/` directory is not in the local checkout; the numbers are the
committed record in `docs/technical_report.md` §7.1 / README.

### 8.2 Operator comparison and physics-loss ablation on `dar` (5 seeds, full budget)

| run | rel $L_2$ 1-step | rel $L_2$ rollout (16) | params | wall (s) |
|---|---|---|---|---|
| FNO, physics on | 0.0042 ± 0.0017 | **0.0567 ± 0.026** | 1.19 M | 138 |
| FNO, physics off | 0.0036 ± 0.0028 | 0.0496 ± 0.044 | 1.19 M | 123 |
| DeepONet | 0.0282 ± 0.0003 | 0.256 ± 0.004 | 0.09 M | 52 |
| GNOT | 0.0302 ± 0.0051 | 0.268 ± 0.040 | 0.31 M | 175 |

Physics on vs off: $t=+0.52$ (physics on nominally *worse*; per-seed on
0.064/0.015/0.056/0.087/0.063 vs off 0.040/0.017/0.047/0.125/0.019). At smoke
scale ($32^2$, 3 epochs) physics-on 0.117 vs off 0.273 and single-seed 0.063 vs
0.177: the residual term buys **convergence speed at small budgets**, not
asymptotic accuracy. The single-seed `runs/simulate/dar_fno` (CPU, 837 s) gives
0.154 rollout error, illustrating run-to-run spread.

### 8.3 Constrained planning vs safe-RL baselines (`runs/seeds_v3/`, `dar`, 5 seeds, $d=0.936$)

| method | return | episode cost (final) | violating evals | worst cost | real transitions | wall (s) |
|---|---|---|---|---|---|---|
| **PSPE hybrid, adaptive $\alpha$** | **−2.448 ± 0.029** | 0.412 ± 0.37 | **7.3%** | 1.109 ± 0.65 | 3,072 | 61 |
| PSPE hybrid, fixed $\alpha=0.5$ | −2.738 ± 0.365 | 2.752 ± 3.6 | 12.7% | 2.758 ± 3.6 | 3,072 | 41 |
| PPO-Lagrangian | −2.558 ± 0.039 | 0.266 ± 0.009 | 0% | 0.297 | 38,400 | 20 |
| CPO | −2.541 ± 0.035 | 0.263 ± 0.011 | 0% | 0.277 | 38,400 | 27 |
| Sauté RL | −2.537 ± 0.024 | 0.264 ± 0.008 | 0% | 0.275 | 38,400 | 20 |
| Primal-dual NPG | −2.544 ± 0.026 | 0.263 ± 0.008 | 0% | 0.275 | 38,400 | 26 |

Per-seed adaptive returns −2.454/−2.427/−2.409/−2.475/−2.473; violating fractions
0.09/0/0.18/0/0.09 (of 11 evaluations); worst costs 1.78/0.27/1.07/0.71/1.71.
Fixed $\alpha$ per-seed returns −2.525/−2.842/−3.335/−2.520/−2.469, worst costs
0.27/4.65/8.27/0.27/0.32 (seed 2 diverged: $\lambda=19.1$).

Paired $t$ (adaptive vs): PPO-Lag **+3.84**, CPO **+4.87**, Sauté **+17.95**,
PD-NPG **+4.52**; vs fixed $\alpha$ +1.66 (n.s.). Final $\alpha=0.9921\pm0.0002$.

Reading: the return advantage is real and the real-sample ratio is $12.5\times$;
the safety claim as built is false — part of the return is bought by running
past the limit. Adaptive $\alpha$ is a **variance/tail** effect (std 0.029 vs
0.365; 7.3% vs 12.7%; no divergence), not a mean effect. An earlier sweep
(`runs/seeds_full/`, before the run-level constraint metric) gave the same
picture: adaptive −2.234 ± 0.034, fixed −2.405 ± 0.25, baselines −2.505 to −2.529.
See Section 12.3 on what "± std" measures in this sweep.

Single-seed README table (200 iterations, earlier code): PSPE −2.251, cost 0.189,
wall 1,658 s vs baselines 577–1,508 s — the planner is the slowest in wall-clock;
it trades compute for real samples.

### 8.4 The safety fix (`runs/constraint_fix/`, `dar`, 5 seeds, 20 epochs / 200 iterations)

| arm | return | violating evals | worst cost | cost bias $\hat b$ | effective limit | real transitions (probe) |
|---|---|---|---|---|---|---|
| baseline | −2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 (max 1.73) | 0 | 0.936 | 3,200 (0) |
| probe only | −2.307 ± 0.16 | 1.8% | 0.56 ± 0.48 | 0.028 | 0.936 | 4,160 (960) |
| margin only | −2.302 ± 0.16 | 1.8% | 0.75 ± 0.62 | 0 | 0.936 | 3,200 (0) |
| **probe + margin** | **−2.301 ± 0.17** | **0% (5/5)** | **0.29 ± 0.15** | 0.023 | 0.758 ± 0.015 | 4,160 (960) |

Return is unchanged (per-seed −2.490/−2.088/−2.318/−2.176/−2.436 vs baseline
−2.465/−2.104/−2.345/−2.161/−2.436); 960 extra real transitions; the planner
still uses $9.2\times$ fewer real samples than the baselines. Margin alone is
bit-identical to baseline (no probe → $\hat v=0$). Across all 15 seed-runs of
the probe + margin configuration (this run, the `alpha_rule` variance arm and the
`joint` disaggregated arm), 13 are clean; `alpha_rule`/`joint` seed 3 had one
evaluation at 0.98 and the joint arm's seed 2 one at 1.71. Honest rate: ~1% of
evaluations vs 7.3% before. The pre-fix baseline here shows 1.8% (one seed) rather
than 7.3%; this runner trains its own surrogate and passes the seed to the
planner, so it evaluates on different IC sets per seed (Section 12.3) — the two
pre-fix figures bracket the true pre-fix rate.

### 8.5 Mixing rule: fixed / variance / Eq. 8 (`runs/alpha_rule/`, 5 seeds, probe + margin on)

| arm | return | violating evals | worst cost | final $\alpha$ | $B^2$ | real transitions |
|---|---|---|---|---|---|---|
| fixed 0.5 | −2.415 ± 0.30 | 3.6% | 1.13 ± 1.5 | 0.5 | — | 4,160 |
| variance rule | −2.302 ± 0.17 | 1.8% | 0.38 ± 0.33 | 0.9916 | — | 4,160 |
| Eq. 8 (measured $B^2$) | −2.299 ± 0.15 | **0%** | 0.34 ± 0.22 | 0.9918 | 0.0013 ± 0.0015 | 6,080 |

Eq. 8 vs variance $t=+0.22$; fixed vs variance $t=-1.51$ (per-seed fixed
−2.895/−2.127/−2.361/−2.223/−2.470: one bad seed again).

### 8.6 Joint vs disaggregated (`runs/joint/`, 5 seeds, $\beta=0.1$, probe + margin on)

| arm | return | violating evals | worst cost | held-out rel $L_2$ before → after | drift |
|---|---|---|---|---|---|
| disaggregated (frozen surrogate) | −2.302 ± 0.17 | 1.8% (1 seed) | 0.38 ± 0.33 | 0.0048 → 0.0048 | 0 |
| **joint, anchored** | −2.296 ± 0.18 | 1.8% (1 seed) | 0.56 ± 0.65 | 0.0048 → **0.0005 ± 0.0002** | −0.0043 |
| joint, unanchored | −2.369 ± 0.17 | 0% | 0.38 ± 0.23 | 0.0048 → **0.975 ± 0.034** | +0.970 |

Per-seed return joint/disaggregated: −2.492/−2.484, −2.090/−2.086,
−2.313/−2.313, −2.148/−2.191, −2.438/−2.438 ($t=+0.66$). Unanchored vs
disaggregated $t=-5.84$. The anchored surrogate is not captured: held-out one-step
error falls 9× on every seed (the planning gradient at one tenth of the data
gradient acts as extra regularised training on the states the policy visits).
The unanchored arm's 0% violation is not safety: a surrogate that predicts nothing
predicts no cost; the probe kept real cost in range. Claim that holds: joint
training is safe when anchored and destructive when not. Claim that does not:
joint beats disaggregated.

### 8.7 Ablations at stub scale (`runs/seeds_rest/`, 5 seeds, full budget)

| ablation | on | off | $t$ | verdict |
|---|---|---|---|---|
| physics-informed loss (rollout rel $L_2$) | 0.0567 ± 0.026 | 0.0496 ± 0.044 | +0.52 | no effect on final accuracy |
| frozen vs fine-tuned perception (val regression rel $L_2$) | 0.0681 ± 0.013 | 0.0908 ± 0.017 | −3.07 (−3.97 on total loss) | frozen wins |
| faithfulness term, stub LM ($F$) | 0.643 ± 0.15 | 0.598 ± 0.006 | +0.67 | no effect (one seed at 0.904; others 0.55–0.60) |

### 8.8 Perception controls

Stub backbone (`runs/perception_seeds/`, 5 seeds):

| arm | field rel $L_2$ | retrieval acc | trainable |
|---|---|---|---|
| pspe (frozen + LoRA + contrastive) | 0.0524 ± 0.008 | **0.510 ± 0.058** | 164 k |
| probe (frozen, decoder only) | 0.0476 ± 0.003 | 0.133 ± 0.007 | 115 k |
| cnn (from scratch, regression only) | **0.0422 ± 0.004** | 0.137 ± 0.018 | 563 k |

CNN vs pspe on rel $L_2$: $t=+2.17$; probe vs pspe $t=+1.74$; pspe vs probe on
retrieval $t=+14.9$.

SigLIP (`runs/perception_real/`, `google/siglip-base-patch16-224`, 3 seeds, `backbone_is_stub: false`):

| arm | field rel $L_2$ | retrieval acc | trainable |
|---|---|---|---|
| pspe | 0.0596 ± 0.027 | **0.754 ± 0.026** | 1.64 M |
| **probe** | **0.0345 ± 0.011** | 0.135 ± 0.009 | 1.64 M |
| cnn | 0.0564 ± 0.015 | 0.130 ± 0.016 | 0.56 M |

Probe vs CNN on rel $L_2$: $t=-4.87$; pspe vs probe: $t=+1.28$ (n.s., 2 df);
pspe vs probe on retrieval $t=+37$, vs CNN $t=+51$. Same shape as the stub:
the design buys **text–image alignment**, not reconstruction, and trades some
reconstruction for it; real SigLIP features are strong enough that adapters do
not help reconstruction.

### 8.9 Explanation

Stub LM against each seed's own trained planner (`runs/explain_seeds/`, 5 seeds):

| arm | $F(b)$ | $F_{\text{ref}}$ | KL | vs post-hoc |
|---|---|---|---|---|
| trained-in | 0.168 ± 0.13 | 0.903 | 2.10 | $t=+5.53$ |
| no-faithful | 0.229 ± 0.21 | 0.903 | 2.47 | $t=+2.56$ |
| post-hoc | 0.101 ± 0.11 | 0.903 | 3.60 | — |

Trained-in vs no-faithful $t=-1.40$ (n.s.). $F$ falls to 0.10–0.23 against a
*trained* policy from ~0.6 against an untrained one.

Qwen2.5-0.5B (`runs/explain_real_v2/`, `runs/faith_weights/`, 3 seeds, batch 8, 200 iterations):

| arm | $F(b)$ | KL | per-seed $F$ |
|---|---|---|---|
| trained-in, $w_f=1$ | **0.468 ± 0.16** | 0.81 | 0.2807 / 0.5570 / 0.5652 |
| no-faithful | 0.468 ± 0.16 | 0.81 | 0.2807 / 0.5570 / 0.5652 |
| trained-in, $w_f=10$ | 0.366 ± 0.29 | 1.59 | 0.0304 / 0.5570 / 0.5093 |
| trained-in, $w_f=100$ | 0.218 ± 0.19 | 9.48 | 0.0044 / 0.2679 / 0.3802 |
| trained-in, $w_f=1000$ | 0.399 ± 0.14 | 0.96 | 0.2807 / 0.5570 / 0.3596 |
| post-hoc | 0.182 ± 0.16 | 2.77 | 0.0035 / 0.2446 / 0.2967 |

Trained-in vs post-hoc $t=+21.3$. Trained-in and no-faithful are identical to
four decimals on every seed **with a working gradient** (parse rate 0.95–0.96,
mean $|\text{reinforce}|$ 0.014–0.038, advantage std 0.004–0.011): at $w_f=1$ the
term's contribution is ~two orders below the supervised term. Above the default
it is harmful and non-monotone (Section 12.2). Decision recorded in the repo: drop
the term from the method, keep trained-in supervision.

### 8.10 Cross-family transfer

Surrogate fidelity gap, channel-padded FNO, 5 seeds, full budget (`runs/transfer_seeds/`):

| pair | source rel $L_2$ | target rel $L_2$ | gap | measurable? |
|---|---|---|---|---|
| rdf → dar | 0.032 ± 0.016 | 0.390 ± 0.039 | 0.358 ± 0.033 | yes (~11σ) |
| swe → rdf | 0.171 ± 0.13 | 0.482 ± 0.037 | 0.311 ± 0.13 | yes |
| dar → rdf | 0.022 ± 0.011 | 1.84 ± 1.1 | 1.81 ± 1.1 | marginal |
| swe → dar | 0.171 ± 0.13 | 0.258 ± 0.071 | 0.087 ± 0.14 | no (crosses 0) |
| rdf → swe | 0.032 ± 0.016 | 10.6 ± 8.2 | 10.5 ± 8.2 | no (σ ≈ mean) |
| dar → swe | 0.022 ± 0.011 | 27.4 ± 26 | 27.4 ± 26 | no (σ ≈ mean; per-seed 10.3/13.0/71.9/10.3/31.5) |

Single-seed smoke matrix ($32^2$, 8 epochs) had dar→swe 38.2, rdf→swe 38.7,
dar↔rdf 1.7/0.4: one draw from a distribution spanning an order of magnitude.
Direction survives (parabolic ↔ parabolic finite; into the wave family collapses);
magnitudes into `swe` are not measurements. The `swe` surrogate is the weak link
(in-family 0.171 ± 0.13 vs 0.022 for `dar`). Within-family parameter shift on
`dar`: 0.030 → 0.253, gap 0.223 ± 0.012.

Planning transfer gap (`runs/transfer_planning/`, 5 seeds, 60 iterations):

| pair | fidelity gap | return transferred | return native | planning gap |
|---|---|---|---|---|
| dar → swe | 27.4 ± 26 | −0.180 ± 0.018 | −0.173 ± 0.010 | 0.007 ± 0.014 |
| dar → rdf | 1.81 ± 1.1 | −7.70 ± 0.78 | −7.11 ± 0.93 | **0.58 ± 0.22** |
| swe → dar | 0.087 ± 0.14 | −2.314 ± 0.17 | −2.280 ± 0.16 | 0.034 ± 0.046 |
| swe → rdf | 0.311 ± 0.13 | −7.44 ± 0.55 | −7.10 ± 0.58 | 0.33 ± 0.71 |
| rdf → dar | 0.358 ± 0.033 | −2.318 ± 0.17 | −2.292 ± 0.15 | 0.026 ± 0.027 |
| rdf → swe | 10.5 ± 8.2 | −0.175 ± 0.009 | −0.173 ± 0.010 | 0.002 ± 0.002 |

Forecast error and decision loss **decouple**: the largest fidelity gap
(dar→swe, 27) costs nothing in return; a fidelity gap of 1.8 (dar→rdf) costs
about 8% of return. On `swe` the return scale (−0.17) makes any planning gap
small in absolute terms; the calibrated `swe` task has little reward/cost tension.

### 8.11 Resolution generalisation (`runs/resolution_seeds/`, 5 seeds, FNO trained at $64^2$)

| eval grid | cells | rel $L_2$ |
|---|---|---|
| $64^2$ | 4,096 | 0.0298 ± 0.014 |
| $96^2$ | 9,216 | 0.0298 ± 0.014 |
| $128^2$ | 16,384 | 0.0300 ± 0.014 |

Flat to three decimals (discretisation invariance). Supersedes the single-seed
$32^2$-trained sweep (0.089 / 0.089 / 0.088 at 32/48/64). The first $64^2$ sweep
returned NaN above $64^2$ from the truth solver's stability limit.

### 8.12 Observed wildfire spread, NDWS (`runs/ndws_seeds/`, 3 seeds, validation split, ~6.0 M labelled cells)

| model | AUC-PR | vs published 0.284 | best epoch | params | gate | wall (s) |
|---|---|---|---|---|---|---|
| **U-Net** | **0.2769 ± 0.0023** (0.2746/0.2791/0.2770) | **97.5%** | 8.3 | 1.63 M | — | 921 |
| hybrid U-Net + gated FNO | 0.2347 ± 0.0047 | 82.7% | 2.3 | 2.82 M | **0.119 ± 0.013** | 2,688 |
| FNO alone (1 seed, `ndws_compare`) | 0.1995 | 70% | 3 | 1.19 M | — | 1,646 |
| all-zeros floor | 0.011–0.013 | — | — | — | — | — |

U-Net vs hybrid $t=+12.5$. The gate, free to take any value and initialised at
zero, settles at 0.12 on every seed: offered the Fourier operator, the model
declines it. The hybrid also overfits earlier (best epoch 2–3 vs 8–9). A spectral
model truncates the high frequencies that *are* a fire front; the same FNO
reaches 0.0098 on smooth shallow water. The remaining gap to 0.284 (Huot et al.)
and to the ~0.34 the repo cites as state of the art is architectural. A first
FNO-only run before the pipeline fixes scored 0.073 (`runs/ndws/`). Note this is
a forecasting result only: NDWS has no intervention channel.

### 8.13 Calibration and misc.

`runs/calibration/limits.json` records the `swe` calibration
($c_0=0.108$, $c_{\text{greedy}}=0.347$, proposed 0.192, old limit slack multiple
0.35). Smoke-scale ablation table (`runs/ablations/results.md`, uncalibrated
$d=2.0$, 30 iterations): all five learners at cost 0.16–0.20, violation 0 — the
degenerate comparison that motivated calibration.

---

## 9. Positioning against prior work

**Neural operators and PDE surrogates.** FNO (Li et al. 2021) learns
resolution-invariant maps between function spaces via truncated spectral
convolution; DeepONet (Lu et al. 2021) via branch/trunk factorisation; GNOT
(Hao et al. 2023) via heterogeneous linear attention. Physics-informed training
of operators follows PINO (Li et al. 2021b) and PINNs (Raissi et al. 2019); the
multi-step rollout term is in the spirit of the pushforward/unrolled training of
Brandstetter et al. (2022). PDEBench (Takamoto et al. 2022) provides the
published errors used as an anchor. What PSPE adds on this axis: the surrogate is
*acted through*, with its Lipschitz constant measured because a bound depends on
it, and the physics term's benefit is shown to be a convergence-speed effect at
this scale, not an accuracy effect — a negative result worth stating.

**Weather/climate ML.** FourCastNet (Pathak et al. 2022), GraphCast (Lam et al.
2023), ClimaX (Nguyen et al. 2023), WeatherBench (Rasp et al. 2020) are
forecasting systems. PSPE targets the step after forecasting: constrained
intervention under a budget. The NDWS result (Huot et al. 2022) is the
repository's only dynamics number against *observed* truth and it is a
forecasting number.

**Model-based RL and hybrid gradients.** Differentiating returns through a learned
model is the stochastic value gradient line (Heess et al. 2015); mixing
reparameterisation and likelihood-ratio gradients with (inverse-)variance
weighting is exactly the mechanism of PIPPS / total propagation (Parmas et al.
2018), which should be cited as the closest precedent for the adaptive $\alpha$;
PSPE's addition is the *measured* pathwise bias term entering the denominator
(Eq. 8) and the finding that on this problem $B^2$ is too small to matter.
Dyna-style and short-rollout methods (MBPO, Janner et al. 2019; PETS, Chua et al.
2018; Dreamer, Hafner et al. 2020) address model exploitation by limiting how far
the model is trusted; PSPE's probe + margin is the constraint-side analogue:
trust the surrogate for reward gradients, but drive the dual from measured real
cost with a margin derived from measured model error.

**Decision-aware model learning.** Joint Simulate+Plan training relates to
value-equivalent / control-oriented model learning (Farahmand et al. 2017; Grimm
et al. 2020; Nikishin et al. 2022). PSPE's result is a cautionary one: without a
data anchor the planning gradient captures the surrogate (rel $L_2$ 0.005 → 0.975);
with a norm-scaled anchor it is harmless and even improves held-out error, but
does not improve the decision.

**Constrained RL.** CPO (Achiam et al. 2017), PID-Lagrangian (Stooke et al.
2020), Sauté RL (Sootla et al. 2022), primal-dual natural gradient (after Kakade
2002 / TRPO, Schulman et al. 2015), CMDPs (Altman 1999), Safety Gym (Ray et al.
2019). PSPE's contribution is not a new constrained-RL algorithm but the
observation that in model-based constrained planning the *cost* channel must be
grounded in reality even when the reward channel is not, and that the standard
dual controlled on model cost fails silently (7.3% → ~1% with probe + margin).
The reimplemented baselines and their closed-form validation (four bugs found)
are themselves a reusable artefact.

**Explanation.** Post-hoc rationalisation of a fixed policy is the dominant
paradigm (the repo names TalkToAgent-style briefs as the control). Faithfulness
as agreement between the explanation and the model's actual reasoning follows
Jacovi & Goldberg (2020); PSPE operationalises it as a KL between the policy and
a frozen parser's reading of the brief, and trains against it with self-critical
sequence training (Rennie et al. 2017). The certificate is split conformal
prediction (Vovk et al. 2005; Shafer & Vovk 2008; Angelopoulos & Bates 2021)
applied to a faithfulness score, related in spirit to conformal guarantees for
LM outputs (Quach et al. 2023). Frozen-backbone + LoRA (Hu et al. 2022) and
SigLIP (Zhai et al. 2023) / Qwen2.5 (Qwen team 2024) are the open-weight
components; the contrastive term is InfoNCE (van den Oord et al. 2018) as in
CLIP (Radford et al. 2021). The honest positioning: the faithfulness *objective*
does not add to supervised brief training on this task; the *certificate* is the
part that holds on a real LM.

**Fire spread.** U-Net (Ronneberger et al. 2015) is the architecture class of the
NDWS paper's baselines; the gated-FNO result is a measured scope limit for
spectral operators on sharp, sparse fronts.

---

## 10. Code map and reproduction

```
pspe/
  simulate/   solvers.py (3 PDEs, RK4, residual), fno.py (FNO, Lipschitz mode, power iteration),
              operators.py (DeepONet, GNOT, factory), multifamily.py (channel padding),
              losses.py, rollout.py (fidelity, resolution), trainer.py, dataset.py,
              pdebench.py, real/{ndws.py, tfrecord.py, models.py}
  plan/       policy.py (tanh-Gaussian, critic), hybrid_gradient.py (adaptive alpha, B^2),
              lagrangian.py (PID dual), trainer.py (probe, margin, Eq. 8, equity dual), joint.py
  perceive/   encoder.py (tiny/cnn/HF backbones, decoder), lora.py, losses.py (InfoNCE),
              text.py (captions, hashed text encoder), dataset.py (renderer), eurosat.py, trainer.py
  explain/    brief.py (grammar, vocabulary), parser.py (frozen), faithfulness.py (F, SCST),
              model.py (tiny / HF-as-encoder), conformal.py, tokenizer.py, trainer.py
  envs/       task.py (reward, g_1, g_2, calibrated limits), actuators.py, pde_env.py
  pipeline.py (e2e loop, transfer_gap, planning_transfer_gap)
  utils/      common.py (constraint_summary, timer, memory), logging.py
baselines/    safe_rl.py (4 algorithms), toy_cmdp.py, run_*_baselines.py, validate_safety_gym.py
eval/         run_ablations.py, run_seeds.py, metrics.py, run_{constraint_fix,alpha_rule,joint,
              lipschitz,conformal,explain_baselines,perception_baselines,transfer,resolution,
              pdebench,ndws}.py, human_rating.py
scripts/      train_{simulate,plan,perceive,explain,e2e}.py (Hydra), generate_data.py,
              calibrate_constraints.py, download_{pdebench,ndws}.py, make_*_figure(s).py,
              tacc/ (SLURM: vista_*.slurm, vista_packed.slurm, driver.sh, fetch/watch scripts)
configs/      Hydra yaml per phase
tests/        17 files, 139 tests (canary and slow markers)
docs/         technical_report.md, proposal_deltas.md, paper_gap_and_plan.md, slides_*.md,
              figures/ (architecture, per-module, applications, digital twin; svg/pdf/png)
runs/         per-experiment results (gitignored raw; results_seeds.{md,json} are the record)
notebooks/    pspe_colab.ipynb (EuroSAT, SigLIP, Qwen on a T4)
```

Key commands: `make data`, `make seeds` (Phase 2 over 5 seeds), `make
perception-baselines`, `make explain-baselines`, `make transfer`, `make
resolution`, `make constraint-fix`, `make pdebench`, `make ndws`, `make
calibrate`, `make test` / `make test-all` / `make canary`. Vista:
`sbatch --export=ALL,EXPERIMENT={cfix|expl|expl2|lip|alpha|joint|res}[,TESTBED=swe|rdf] scripts/tacc/vista_packed.slurm`.
Hydra overrides work on every training script (e.g. `planner.adaptive_alpha=false`,
`model.backbone=google/siglip-base-patch16-224`, `train.padded=true`).

Licences: repository MIT; PyTorch/NumPy/Gymnasium/Hydra BSD/Apache/MIT;
transformers/peft Apache-2.0; Qwen2.5-0.5B/1.5B Apache-2.0 (larger sizes use the
Tongyi Qianwen licence). Check the SigLIP, NDWS and PDEBench licences/terms on
their own pages before redistribution — the repo README does not record them.
No paid APIs anywhere.

---

## 11. Claims that survive, claims that do not, claims not yet tested

**Survive multi-seed scrutiny**

1. Return advantage over all four safe-RL baselines ($t=+3.84$ to $+17.95$).
2. $12.5\times$ real-sample efficiency ($9.2\times$ with probe + margin).
3. Probe + margin cuts constraint violation ~7× (7.3% → ~1% of evaluations; 13/15
   clean) at no return cost.
4. Adaptive $\alpha$ prevents dual divergence (a tail/stability claim).
5. FNO ≫ DeepONet ≫ GNOT on synthetic and PDEBench data; FNO within ~2× of the
   published PDEBench figure.
6. Resolution invariance $64^2\to128^2$.
7. Direction of cross-family transfer; forecast error and decision loss decouple.
8. Joint training safe when anchored (surrogate improves 9×, decision unchanged),
   destructive when not.
9. Assumption 1 holds after training; finite-horizon Prop. 1 holds (loosely).
10. Supervised trained-in briefs beat the untrained-generator control on stub and
    Qwen ($t=+5.5$, $+21$); the conformal certificate holds at 90% on 3/3 Qwen seeds.
11. Frozen SigLIP probe reconstructs best; LoRA + contrastive buys alignment
    (retrieval 0.75 vs 0.13).
12. U-Net at 97.5% of published on observed fire spread; the FNO's scope limit on
    sharp fronts, measured by a gate.

**Do not survive**

1. Constraint satisfaction as originally built.
2. Physics-informed loss improving final accuracy.
3. Adaptive $\alpha$ improving mean return.
4. The faithfulness loss term (null at default weight, harmful above, on stub and Qwen).
5. LoRA + contrastive beating a probe or CNN on reconstruction.
6. Any magnitude for transfer into the wave family.
7. Joint > disaggregated on return.
8. Eq. 8 improving on the variance-only rule.
9. Prop. 1's infinite-horizon bound as a usable number.
10. Constraint satisfaction as a guarantee.
11. The FNO as a universal surrogate.

**Not yet measured / not run**

* Constraint fix and joint training on `swe` and `rdf` (Vista jobs 1013191–1013194
  submitted 2026-09-21; fetcher writes `runs/VISTA_PHASE3_RESULTS.md`).
* PDEBench over multiple seeds; the 2D reaction–diffusion PDEBench file.
* Safety-Gymnasium scale validation of the baselines.
* Zero-shot VLM perception baseline; perception on real imagery (EuroSAT loader
  built, no committed result); agreement with human descriptions.
* Human Likert study on real briefs.
* Planning on any real dataset (NDWS has no intervention channel); a flood
  testbed on real data (Sen1Floods11 + DEM mentioned as next).
* Equity constraint $g_2$ in any reported comparison.
* $128^2$ headline runs for planning (only the resolution sweep touches $128^2$).
* Conformal-quantile margin for the constraint (proposed replacement for $2\sigma$).

---

## 12. Audit findings: issues to resolve before publication

These were found while tracing numbers to code during this audit. None are
mentioned in `docs/technical_report.md`.

### 12.1 The "post-hoc" explanation control is an untrained generator

`ExplainTrainConfig(posthoc=True)` sets `iterations = 0`
(`pspe/explain/trainer.py`). The arm therefore evaluates a *randomly initialised*
prefix MLP, brief embedding and output head over the frozen LM — it has never
been trained on any policy or any brief. Its briefs do not parse, so its $F$ is
the parser-fallback score $\exp(-\mathrm{KL}(\pi\,\|\,\mathcal N(0,0.4\mathbf 1)))$,
which is why the per-seed post-hoc $F$ (0.0035 / 0.245 / 0.297) tracks how close
each seed's policy is to zero action. "Trained-in beats post-hoc, $t=+21$" is
therefore "a supervised generator beats an untrained generator". A defensible
post-hoc control would train the same generator on briefs from a *different*
policy (or on generic briefs) and then explain this policy without further
training; or prompt the instruction-tuned LM directly. Until one of these runs,
the trained-in-vs-post-hoc claim should be stated as trained-vs-untrained.

### 12.2 Qwen briefs are truncated at 96 tokens; the reference needs ~136

`ExplainConfig.max_len` defaults to 96 and the Vista runners construct
`ExplainConfig(backbone=...)` without overriding it (the Colab notebook sets 160
for exactly this reason). In `runs/conformal_real/seed_*/train/briefs.jsonl` every
generated brief is exactly 96 tokens, names 7 of 9 actuators and never reaches
the cost / reward / confidence footer, so the parsed std falls back to 0.4 and two
actuators default to 0. The reference briefs average 136 tokens. Consequences:

* the Qwen $F\approx0.57$ plateau (seeds 1, 2) is a **truncation ceiling**, not a
  learning limit; $F_{\text{ref}}=0.90$ is computed on the untruncated reference
  and overstates what the generator could reach;
* the weight sweep and the conformal floors inherit the ceiling;
* the generated briefs are near-identical across states and steps (same seven
  actuators at −0.15/−0.20), suggesting the trained `dar` policy is close to
  state-independent — worth checking before claiming the briefs are informative.

Rerun the Qwen arms with `max_len ≥ 160` before quoting any Explain number.

Also relevant to §8.9: with $w_f=1000$ two of three seeds return $F$ identical to
$w_f=1$ to four decimals despite a non-zero REINFORCE gradient (mean unweighted
magnitude 0.036–0.046), while $w_f=10$ and $100$ collapse the supervised fit
(final NLL 0.9 vs 0.05). Evaluation is on the *greedy* decode, a discrete object:
the term perturbs the distribution without moving the argmax unless it is large
enough to flip tokens, and when it does it flips them for the worse. The sweep is
non-monotone in $w_f$; 3 seeds are not enough to characterise it.

### 12.3 The headline 5-seed planner sweep did not pass the seed to the planner or baselines

`eval/run_ablations.py` builds `PlannerConfig(iterations=..., adaptive_alpha=...)`
and `SafeRLConfig(iterations=...)` without `seed=`, so in `runs/seeds_v3/` (and
`seeds_full`, `seeds_rest`) every "seed" uses `PlannerConfig.seed = 0`: the
training-rollout generator and the evaluation IC set (`seed + 10_000`) are
**identical across seeds**; only the global RNG (`seed_everything`) — network
initialisation, surrogate training, minibatch order — varies. Consequences:

* the reported ± std in §8.3 (0.029 for the planner) understates seed variance
  because evaluation ICs are not resampled; runners that do pass the seed
  (`run_constraint_fix.py`, `run_joint.py`, `run_alpha_rule.py`) show 0.16–0.17;
* the paired $t$-tests against baselines remain valid (all arms share the same
  eval set within a seed), but the two families of runs are **not comparable on
  absolute return** (−2.448 vs −2.302 is an eval-set difference, not a budget
  difference — `docs/slides_experiments.md` slide 4 attributes it to a "quick
  budget", which is wrong; `runs/PHASE1_ANALYSIS.md` is right that the budgets
  are identical);
* the 7.3% violation figure comes from one eval IC set replicated 5 times; the
  seed-passing baseline arm in `constraint_fix` shows 1.8% (one seed). Both are
  pre-fix numbers on the same code path; the honest pre-fix statement is "between
  ~2% and ~7% of evaluations depending on the evaluation set, worst cost 1.1–1.7
  against 0.936".

Fix: pass `seed=seed` into both configs in `run_ablations.py` and rerun
`make seeds` before the paper's main table.

### 12.4 Dataset size and sample counts

The cached `dar_64.npz` has 128 trajectories (Section 3.2); `--full` requests 256
but does not regenerate. State this in the paper (128 × 24 = 3,072 transitions,
102 train / 26 val trajectories), and use one convention (transitions) for the
real-sample column — `constraint_fix` reports 3,200 (states) for the same data.

### 12.5 Provenance gaps

* PDEBench results (§8.1) are single-seed and their raw run directory is not in
  the local `runs/`; only the committed tables exist.
* `runs/seeds_v2/` and `runs/seeds/` are superseded sweeps; do not mix.
* The Sauté baseline's 0.49 on the toy CMDP is a known conservative bias of the
  method, not a bug; say so if the table is reproduced.
* The "156 tests" figure is the collected count with parametrisation; 139 test
  functions exist.

### 12.6 Smaller points a reviewer will ask

* The reference distribution for $F$ uses the post-tanh mean with the pre-tanh
  std; state the approximation.
* The HF Explain path uses the LM as a frozen encoder with a *closed 900-token
  vocabulary* and trainable embedding/head; it is not free-text generation by
  Qwen. Say so explicitly, and do not describe the briefs as "natural language"
  produced by the LM.
* `swe` returns are ~−0.17 with almost no reward/cost tension; planning-transfer
  gaps into `swe` are small for that reason as much as any other.
* Adaptive $\alpha$ saturates at 0.992 at full budget: on `dar` the planner is
  effectively pathwise-only; the hybrid's value shows at small budgets (0.73) and
  in preventing the fixed-$\alpha$ divergence.
* The equity constraint has never been exercised in a reported run.
* The safety fix adds 960 real transitions (probe) — include them in every
  sample-efficiency ratio ($9.2\times$, not $12.5\times$, once the fix is on).
* Corollary 1's statement should be weakened to the probe + margin rate (~1%) and
  the "0% on 5/5" figure retired (it was one draw of 15).

---

## 13. References

Achiam, Held, Tamar, Abbeel. Constrained Policy Optimization. ICML 2017.
Altman. Constrained Markov Decision Processes. Chapman & Hall, 1999.
Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction. 2021.
Brandstetter, Worrall, Welling. Message Passing Neural PDE Solvers. ICLR 2022.
Chua, Calandra, McAllister, Levine. Deep RL in a Handful of Trials (PETS). NeurIPS 2018.
Farahmand, Barreto, Nikovski. Value-Aware Loss Function for Model-based RL. AISTATS 2017.
Grimm, Barreto, Singh, Silver. The Value Equivalence Principle for Model-Based RL. NeurIPS 2020.
Hafner, Lillicrap, Ba, Norouzi. Dream to Control (Dreamer). ICLR 2020.
Hao et al. GNOT: A General Neural Operator Transformer for Operator Learning. ICML 2023.
Heess et al. Learning Continuous Control Policies by Stochastic Value Gradients. NeurIPS 2015.
Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
Huot et al. Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data. IEEE TGRS 2022.
Jacovi, Goldberg. Towards Faithfully Interpretable NLP Systems. ACL 2020.
Janner, Fu, Zhang, Levine. When to Trust Your Model: Model-Based Policy Optimization (MBPO). NeurIPS 2019.
Kakade. A Natural Policy Gradient. NeurIPS 2002.
Lam et al. Learning Skillful Medium-Range Global Weather Forecasting (GraphCast). Science 2023.
Li et al. Fourier Neural Operator for Parametric PDEs. ICLR 2021.
Li et al. Physics-Informed Neural Operator for Learning PDEs (PINO). 2021.
Lu, Jin, Pang, Zhang, Karniadakis. Learning Nonlinear Operators via DeepONet. Nat. Mach. Intell. 2021.
Miyato, Kataoka, Koyama, Yoshida. Spectral Normalization for GANs. ICLR 2018.
Nguyen et al. ClimaX: A Foundation Model for Weather and Climate. ICML 2023.
Nikishin et al. Control-Oriented Model-Based RL with Implicit Differentiation. AAAI 2022.
van den Oord, Li, Vinyals. Representation Learning with Contrastive Predictive Coding. 2018.
Parmas, Rasmussen, Peters, Doya. PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos. ICML 2018.
Pathak et al. FourCastNet. 2022.
Quach et al. Conformal Language Modeling. 2023.
Qwen Team. Qwen2.5 Technical Report. 2024.
Radford et al. Learning Transferable Visual Models from Natural Language Supervision (CLIP). ICML 2021.
Raissi, Perdikaris, Karniadakis. Physics-Informed Neural Networks. J. Comput. Phys. 2019.
Rasp et al. WeatherBench. JAMES 2020.
Ray, Achiam, Amodei. Benchmarking Safe Exploration in Deep RL (Safety Gym). 2019.
Rennie et al. Self-Critical Sequence Training for Image Captioning. CVPR 2017.
Ronneberger, Fischer, Brox. U-Net. MICCAI 2015.
Schulman et al. Trust Region Policy Optimization. ICML 2015; Proximal Policy Optimization. 2017.
Shafer, Vovk. A Tutorial on Conformal Prediction. JMLR 2008.
Sootla et al. Sauté RL: Almost Surely Safe RL Using State Augmentation. ICML 2022.
Stooke, Achiam, Abbeel. Responsive Safety in RL by PID Lagrangian Methods. ICML 2020.
Takamoto et al. PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS 2022 D&B.
Vovk, Gammerman, Shafer. Algorithmic Learning in a Random World. Springer 2005.
Williams. Simple Statistical Gradient-Following Algorithms for Connectionist RL. Mach. Learn. 1992.
Zhai, Mustafa, Kolesnikov, Beyer. Sigmoid Loss for Language Image Pre-Training (SigLIP). ICCV 2023.
