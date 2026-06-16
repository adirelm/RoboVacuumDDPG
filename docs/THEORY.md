# THEORY.md — DDPG Mathematical Foundations (RoboVacuumDDPG)

> Bar-Ilan *Vibe Coding & RL* workshop — **Assignment 5** (Lecture 09, DDPG),
> group code `adrl-001`, repo **RoboVacuumDDPG**.
> Cross-references each equation to its source paper, the `config/config.yaml`
> key that seals its hyperparameter, and the `src/` module that implements it.
> DDPG is implemented **from scratch in PyTorch** — no Gymnasium, no Gazebo,
> no SB3 (brief §"דרישת חובה"). Every equation below points at our own code.

---

## 0  Problem setting — continuous control on a vacuum MDP

The vacuum task is a continuous-action Markov Decision Process
$(\mathcal{S}, \mathcal{A}, P, r, \gamma)$. The agent observes a
20-dim normalized state and emits a **2-dim continuous action**
$a = [\text{throttle}, \text{steer}] \in [-1,1]^2$ (spec §3). Because the
action space is continuous, value-table and discrete-logit methods
(Q-Learning, DQN, and the discrete-action PPO of Assignment 4) cannot be
applied directly: there is no finite $\arg\max_a Q(s,a)$ to enumerate, and a
categorical softmax cannot represent a real-valued motor command. DDPG
(Lillicrap et al. 2016) resolves this by learning a **deterministic** actor
$\mu_\theta(s)$ that *outputs* the maximizing action, paired with a critic
$Q_\varphi(s,a)$ that scores it — an actor-critic instantiation of the
Deterministic Policy Gradient theorem (Silver et al. 2014).

**State** (spec §3): $s = [\,d_1/d_{\max}, \dots, d_{n}/d_{\max},\; v,\; \omega,\;
\cos\beta,\; \sin\beta\,]$ — $n$ lidar ray distances (default $n=16$,
`env.n_rays`) normalized by $d_{\max}=5.0$ m (`env.ray_max`), the current
linear/angular speed $(v,\omega)$, and the **two-component** $(\cos\beta,
\sin\beta)$ unit-vector bearing $\beta$ toward the nearest uncleaned cell. With
$n=16$ this is $16 + 2 + 2 = \mathbf{20}$ dimensions (`state_dim = n_rays + 4`).

**Action → kinematics** (spec §3, unicycle model): $v = \text{throttle}\cdot
V_{\max}$, $\omega = \text{steer}\cdot \Omega_{\max}$ with $V_{\max}=0.5$
(`env.v_max`), $\Omega_{\max}=1.5$ (`env.omega_max`), integrated over
$\Delta t = 0.1$ s (`env.dt`):

$$x \mathrel{+}= v\cos\theta\,\Delta t,\qquad
  y \mathrel{+}= v\sin\theta\,\Delta t,\qquad
  \theta \mathrel{+}= \omega\,\Delta t.$$

**Reward** (spec §3, `config.reward`):

$$r = k_{\text{cov}}\cdot(\text{new cells cleaned})
     \; - \; k_{\text{col}}\cdot \mathbb{1}[\text{collision}]
     \; - \; k_{\text{step}},$$

with $k_{\text{cov}}=1.0$ (`reward.k_coverage`), $k_{\text{col}}=10.0$
(`reward.k_collision`), $k_{\text{step}}=0.01$ (`reward.k_step`).
Implementation: `src/env/reward.py`.

The learning objective is the discounted return
$J(\mu) = \mathbb{E}\big[\sum_{t\ge 0}\gamma^t r_t\big]$ with $\gamma=0.99$
(`ddpg.gamma`).

---

## 1  Deterministic policy & critic (Lillicrap et al. 2016, arXiv:1509.02971)

### 1.1  Deterministic actor $\mu_\theta$

Unlike a stochastic policy $\pi_\theta(a\mid s)$, DDPG learns a **deterministic**
map from state to a single action:

$$a = \mu_\theta(s),\qquad \mu_\theta:\mathcal{S}\to[-1,1]^2.$$

The final layer is **Tanh**-bounded so that the action lands exactly in
$[-1,1]^2$ (spec §3, §5.1), matching the unicycle's normalized
$[\text{throttle}, \text{steer}]$ domain. The hidden trunk is an MLP of widths
`ddpg.hidden_sizes = [256, 256]`.

Implementation: `src/model/actor.py` — the `Actor` MLP whose `forward()` ends
in `torch.tanh(...)` to enforce the bound (architecture test asserts
$\mu_\theta(s)\in[-1,1]$).

### 1.2  Critic $Q_\varphi$

The critic is an action-value function that takes the state **and** the action
(spec §5.1) and estimates the discounted return of following $\mu_\theta$
thereafter:

$$Q_\varphi(s,a) \approx
  \mathbb{E}\Big[\textstyle\sum_{k\ge 0}\gamma^{k} r_{t+k}
  \,\Big|\, s_t=s,\,a_t=a,\,a_{t+k}=\mu_\theta(s_{t+k})\Big].$$

The action is concatenated with the state ($s\oplus a$) and fed through an MLP
of widths `ddpg.hidden_sizes = [256, 256]` to a single scalar $Q$.

Implementation: `src/model/critic.py` — the `Critic` MLP consuming
`torch.cat([state, action], dim=1)` (architecture test asserts the critic
output is shape `(batch, 1)`).

---

## 2  Bellman TD target with target networks

### 2.1  Target networks $\mu_{\theta'}$, $Q_{\varphi'}$

DDPG keeps slow-moving copies of both networks — the **target actor**
$\mu_{\theta'}$ and **target critic** $Q_{\varphi'}$ — to stabilize the
bootstrap (spec §5.2; Lillicrap et al. 2016 §3). Using the live critic to form
its own regression target couples the prediction and the label, which drives
divergence; the target nets break that coupling.

### 2.2  TD target (one-step Bellman backup, deterministic bootstrap)

For a transition $(s, a, r, s', d)$ sampled from replay, the regression target
is:

$$y = r + \gamma\,(1 - d)\,Q_{\varphi'}\!\big(s',\,\mu_{\theta'}(s')\big),$$

where $d\in\{0,1\}$ is the done flag. The $(1-d)$ terminal mask zeroes the
bootstrap on episode boundaries (`VacuumEnv` ends at `env.max_steps = 1000` or
an optional coverage target, spec §3) so $Q_{\varphi'}(s',\cdot)$ is not added
across a `reset()`. The bootstrap action is the **deterministic** target-actor
output $\mu_{\theta'}(s')$ — there is no expectation over an action
distribution, which is what makes DDPG's target cheaper than a stochastic
actor-critic's.

> **Known simplification (time-limit vs. true termination).** `VacuumEnv`
> reports a single `done` for *both* the goal (coverage target) and the
> `max_steps` time-limit, and the mask zeroes the bootstrap for both. Because
> the coverage target is rarely reached in this run, most episodes end by
> time-limit, so the value of bootstrapping at a non-terminal cutoff state is
> dropped on each episode's last transition — a slight low-bias on that one
> transition per 1000. The textbook-correct treatment separates *termination*
> from *truncation* and only masks true terminals; we keep the single flag for
> simplicity and note the deviation rather than hide it. Impact is bounded
> (1/1000 of transitions) and does not change the qualitative results.

**Sealed hyperparameter:** $\gamma = 0.99$ (`config.ddpg.gamma`; asserted at
`DDPGAgent.__init__`).

Implementation: `src/ddpg/agent.py` `update()` — `y = r + gamma * (1 - done) *
target_critic(next_state, target_actor(next_state))`, computed under
`torch.no_grad()`.

---

## 3  Critic loss — minimize TD error (MSE)

The critic is fit by regressing $Q_\varphi(s,a)$ onto the target $y$ over a
mini-batch $\mathcal{B}$ of `ddpg.batch_size = 128` transitions drawn uniformly
from the replay buffer (`ddpg.buffer_size = 1_000_000`):

$$L(\varphi) = \frac{1}{|\mathcal{B}|}\sum_{(s,a,r,s',d)\in\mathcal{B}}
  \Big(Q_\varphi(s,a) - y\Big)^2,
  \qquad
  y = r + \gamma\,(1-d)\,Q_{\varphi'}\!\big(s',\mu_{\theta'}(s')\big).$$

The gradient $\nabla_\varphi L$ is taken with the target $y$ treated as a
constant (it is built under `no_grad`). To curb the value-explosion failure
mode of off-policy critics, gradients are clipped to norm
`ddpg.grad_clip = 1.0` before the optimizer step. The critic uses the faster
learning rate $\text{lr}_{\text{critic}} = 10^{-3}$ (`ddpg.lr_critic`) — standard
DDPG practice, since the actor should chase a critic that has already settled.

**Sealed hyperparameters:** $\text{lr}_{\text{critic}}=10^{-3}$
(`config.ddpg.lr_critic`), batch $128$ (`ddpg.batch_size`), buffer $10^6$
(`ddpg.buffer_size`), grad-clip $1.0$ (`ddpg.grad_clip`).

This curve is the deliverable `results/figures/critic_loss.png` (spec §7) —
the per-episode-mean critic loss vs episode (convergence time), rendered by
`render_critic_loss.py` from the step-level `critic_losses` history.

Implementation: critic MSE in `src/ddpg/agent.py` `update()`; uniform sampling
in `src/ddpg/replay_buffer.py`.

---

## 4  Deterministic policy gradient (Silver et al. 2014; Lillicrap et al. 2016)

The actor is improved by ascending the expected critic value of its own
actions. The **Deterministic Policy Gradient theorem** (Silver et al. 2014)
gives the gradient of $J$ w.r.t. the actor parameters via the chain rule
through the critic:

$$\nabla_\theta J
  = \mathbb{E}_{s\sim\mathcal{B}}\Big[\,
      \nabla_a Q_\varphi(s,a)\big|_{a=\mu_\theta(s)}
      \;\cdot\;
      \nabla_\theta \mu_\theta(s)
    \,\Big].$$

Intuitively: $\nabla_a Q_\varphi$ says "which way in action space raises value";
$\nabla_\theta\mu_\theta$ pushes the actor parameters so its output moves that
way. In PyTorch this is realized by **minimizing** the surrogate actor loss

$$L(\theta) = -\,\frac{1}{|\mathcal{B}|}\sum_{s\in\mathcal{B}}
  Q_\varphi\!\big(s,\mu_\theta(s)\big),$$

whose autograd gradient is exactly the (negated) DPG estimate above — the
critic is held fixed during the actor step. The actor uses the slower learning
rate $\text{lr}_{\text{actor}} = 10^{-4}$ (`ddpg.lr_actor`), one order of
magnitude below the critic.

**Sealed hyperparameter:** $\text{lr}_{\text{actor}}=10^{-4}$
(`config.ddpg.lr_actor`).

Implementation: `src/ddpg/agent.py` `update()` — `actor_loss =
-critic(state, actor(state)).mean()`, backpropagated into `actor.py` only.

---

## 5  Polyak soft target update

After each gradient step, both target networks are nudged toward their live
counterparts by **Polyak averaging** (spec §5.2; Lillicrap et al. 2016 §3),
the brief-mandated "soft target update" whose code lines must be cited:

$$\theta' \leftarrow \tau\,\theta + (1-\tau)\,\theta',
  \qquad
  \varphi' \leftarrow \tau\,\varphi + (1-\tau)\,\varphi'.$$

A small $\tau$ makes the targets track the online nets slowly, so the
regression target $y$ in §2.2 drifts gently rather than jumping each step —
this is the second stabilizer (alongside the target nets themselves) that
prevents the off-policy critic from collapsing (answered in `docs/ANALYSIS.md`
Q3). Hard updates would correspond to $\tau=1$.

**Sealed hyperparameter:** $\tau = 0.005$ (`config.ddpg.tau`; the brief example
value), asserted at `DDPGAgent.__init__`. A dedicated unit test checks the
Polyak math element-wise (spec §8).

Implementation: `src/ddpg/agent.py:92-100`, `DDPGAgent.soft_update(self)` — for
each `(online, target)` in `[(actor, actor_target), (critic, critic_target)]`,
under `torch.no_grad()`:
`pt.mul_(1.0 - self.tau).add_(self.tau * po)` (agent.py:100 — in-place Polyak
update of each target parameter `pt` toward its online counterpart `po`). Called
once per learning step from `update()` (agent.py:67).

---

## 6  Gaussian exploration noise

A deterministic actor explores nothing on its own, so during **data collection**
(not during the target/critic computations) the action is perturbed by additive
**Gaussian** noise — the brief explicitly mandates Gaussian, not
Ornstein-Uhlenbeck (spec §5.4, ADR-003):

$$a = \operatorname{clip}\big(\mu_\theta(s) + \mathcal{N}(0,\sigma^2),\,-1,\,1\big).$$

The standard deviation is annealed linearly from $\sigma_{\text{start}}=0.2$
(`noise.sigma_start`) to $\sigma_{\text{end}}=0.05$ (`noise.sigma_end`) over
`noise.sigma_decay_steps = 50000` steps, so the policy explores broadly early
and exploits its learned $\mu_\theta$ later. Before learning begins, the first
`ddpg.warmup_steps = 1000` actions are sampled uniformly at random to seed the
replay buffer (spec §5.3, `config.ddpg.warmup_steps`). Removing this noise too
early collapses the coverage map to a narrow repeated path — the failure mode
quantified in `docs/ANALYSIS.md` Q2.

**Sealed hyperparameters:** $\sigma_{\text{start}}=0.2$ (`noise.sigma_start`),
$\sigma_{\text{end}}=0.05$ (`noise.sigma_end`), decay $50000$
(`noise.sigma_decay_steps`), warmup $1000$ (`ddpg.warmup_steps`), noise type
`gaussian` (`noise.type`).

Implementation: `src/ddpg/noise.py:31` — `GaussianNoise.sample()` (seeded for
reproducibility, spec §8); added to the actor action inside `DDPGAgent.act()`
(`src/ddpg/agent.py:50`) during collection only (`explore=True`).

---

## 7  Full DDPG update step (assembled)

One call to `DDPGAgent.update()` chains §3 → §4 → §5:

1. Sample mini-batch $\mathcal{B}$ from `replay_buffer.py` (uniform, size 128).
2. **Critic step** (§3): build $y$ (§2.2) under `no_grad`, minimize MSE
   $L(\varphi)$, clip grad-norm to 1.0, step the lr$=10^{-3}$ optimizer.
3. **Actor step** (§4): minimize $-\,\mathbb{E}[Q_\varphi(s,\mu_\theta(s))]$
   (the DPG surrogate), step the lr$=10^{-4}$ optimizer.
4. **Polyak soft update** (§5): nudge $\theta'$ and $\varphi'$ by $\tau=0.005$.

The collection ↔ store ↔ update ↔ log outer loop lives in
`src/services/trainer.py`; the SDK (`src/sdk/sdk.py` `RoboVacuumSDK`) is the
single entry point (`build_env`, `train`, `evaluate`, `rollout`,
`coverage_report`, `trajectory`, `map_walls`, `coverage_grid`, `live_session`). The
episode-return trace is the deliverable
`results/figures/learning_curve.png` (spec §7, mean±CI over
`training.seeds = [42, 7, 123, 314, 271]`).

---

## 8  Equation–Module Cross-Reference

| Equation | Paper / source | `src/` module | Config key |
|----------|----------------|---------------|-----------|
| $a=\mu_\theta(s)$ — Tanh-bounded deterministic actor | Lillicrap 2016; Silver 2014 | `model/actor.py` (`Actor.forward`) | `ddpg.hidden_sizes` |
| $Q_\varphi(s,a)$ — critic on $s\oplus a$ | Lillicrap 2016 | `model/critic.py` (`Critic.forward`) | `ddpg.hidden_sizes` |
| $y=r+\gamma(1-d)Q_{\varphi'}(s',\mu_{\theta'}(s'))$ — TD target | Lillicrap 2016 §3 | `ddpg/agent.py` (`update`) | `ddpg.gamma=0.99` |
| $L(\varphi)=\mathbb{E}[(Q_\varphi-y)^2]$ — critic MSE | Lillicrap 2016 | `ddpg/agent.py` (`update`) | `ddpg.lr_critic=1e-3`, `ddpg.grad_clip=1.0` |
| $\nabla_\theta J=\mathbb{E}[\nabla_a Q_\varphi\,\nabla_\theta\mu_\theta]$ — DPG | Silver 2014; Lillicrap 2016 | `ddpg/agent.py` (`update`) | `ddpg.lr_actor=1e-4` |
| $\theta'\leftarrow\tau\theta+(1-\tau)\theta'$ — Polyak soft update | Lillicrap 2016 §3 | `ddpg/agent.py` (`soft_update`) | `ddpg.tau=0.005` |
| $a=\operatorname{clip}(\mu_\theta(s)+\mathcal{N}(0,\sigma^2))$ — Gaussian noise | Lillicrap 2016; ADR-003 | `ddpg/noise.py` (`GaussianNoise`) + `ddpg/agent.py` (`act`) | `noise.sigma_start=0.2`, `noise.sigma_end=0.05`, `noise.sigma_decay_steps=50000` |
| $r=k_{\text{cov}}\Delta - k_{\text{col}}\mathbb{1} - k_{\text{step}}$ — reward | spec §3 | `env/reward.py` | `reward.{k_coverage,k_collision,k_step}` |
| unicycle pose integration | spec §3; HouseExpo sim | `env/kinematics.py` | `env.{dt,v_max,omega_max}` |
| uniform replay sampling | Lillicrap 2016 | `ddpg/replay_buffer.py` | `ddpg.{batch_size,buffer_size}` |

---

## 9  References

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y.,
  Silver, D., & Wierstra, D. *Continuous Control with Deep Reinforcement
  Learning* (DDPG). arXiv:1509.02971 (2015); published at ICLR 2016. (In-text
  citations use the ICLR-2016 year.)
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M.
  (2014). *Deterministic Policy Gradient Algorithms* (DPG). *ICML 2014*.
- Li, T., Ho, D., Li, C., Zhu, D., Wang, C., & Meng, M. Q.-H. (2019).
  *HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms
  on Mobile Robots*. arXiv:1903.09845.
