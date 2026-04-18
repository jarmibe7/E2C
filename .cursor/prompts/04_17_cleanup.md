### 1. Overview

**Project name:**
data4wm (data for world models)

**Purpose:**
This repository is associated with a paper that focues on \textit{how to collect data for world models}. All interactions are completely unstructured and unsupervised; it might be closest to categorize this as "Runtime Unsupervised Robot Play". It implements a world model _WITHOUT_ any reward function, and focuses on designing several "exploration" objectives for collecting data.

**Associated paper / theory:**
I am pasting in some relevant sections of the abstract and methodology from the paper:
Abstract:
Although reinforcement learning has provided a powerful framework for learning, robots operating from scratch in unstructured environments cannot rely on task-specific rewards or privileged state information to discover physical affordances. Instead, an agent must acquire knowledge of how their actions can affect the world through unsupervised interactions. We propose a belief-space model predictive control framework that maximizes expected information gain over a learned pixel-based latent dynamics model, explicitly selecting actions that induce the greatest change in the agent’s internal beliefs. We further show that action-conditioned belief-space information gain can be naturally composed into two distinct objectives: one over latent dynamics that induces persistent state coverage, and another over the latent pixel observations that induces sustained interactions in pixel regions of interest. Without access to rewards or environment states, our approach consistently produces contact-rich object interaction, outperforming exploration baselines in interaction-centric metrics across 5 simulated robotic environments. Finally, we demonstrate the same principle on a physical robot in a 2D pushing task with previously unseen objects with just one hour of physical interaction, highlighting how belief-driven data acquisition can expose affordance structure without task-specific supervision.

Key points:
Recent progress in learning latent dynamics models from pixels \cite{PlaNet, DreamerV3, DayDreamer} predicts the evolution of explicit beliefs over latent under candidate actions.
This creates the opportunity to move beyond model-based reactive curiosity signals and instead reason directly about how actions reshape an agent’s internal model of the world. 
\textit{The goal of this work is to optimize robot behaviors for environment interactions without relying on any model of environmental structure.}

We measure information gain by maximizing the Kullback-Leibler (KL) divergence across action-conditioned beliefs in the latent space. We optimize agent actions through a closed-loop, model predictive control framework that maximizes this objective.
We further decompose affordance discovery into two distinct objectives: 
1) learning relevant pixels in the observation space, and 
2) learning relevant actions in the control space.  
In this setting, affordance discovery is an end-to-end process, where an agent can simultaneously learn and collect actions.

Drawing from the latent distributions in recurrent state space models formulation \cite{PlaNet, DreamerV3}, we design a recurrent predictive latent model which resolves dynamics into a deterministic hidden state $h_t$ and a stochastic latent state $z_t$. Given an image observation $o_t$ and control inputs $u_t$, the model defines: 
\begin{equation}
    \begin{aligned}
    \text{Observation Encoder:} \quad
    & z_t \sim q_\phi(z_t \mid o_t) \\[2pt]
    \text{Deterministic Dynamics:} \quad
    & h_{t+1} = g_\theta(h_{t}, z_{t}, u_{t}) \\[2pt]
    \text{Stochastic Dynamics:} \quad
    & \hat{z}_{t+1} \sim p_\theta(\hat{z}_{t+1} \mid h_{t+1}) \\[2pt]
     \text{Predictive Belief:} \quad
    & \hat{z}_{t+1} \sim f_\theta(z_t, u_t) \\[1pt] 
         & \quad \,\,\,\,\,\, \equiv  p_\theta(\hat{z}_{t+1} \mid g_\theta(h_{t}, z_{t}, u_{t})) \\[2pt]
    \text{Observation Decoder:} \quad
    & \hat{o}_{t+1} \sim p_\theta(o_{t+1} \mid \hat{z}_{t+1}) \\
    \end{aligned}
\label{eq:rssm-structure}
\end{equation}
% Note the decoder can take a latent vector from either the encoder or transition model. 
We use $f_\theta(z_t, u_t)$ to denote the predictive distribution over $z_{t+1}$ induced by the deterministic dynamics, which aggregates interaction history, and stochastic dynamics, which captures predictive uncertainty. (The world model implicitly keeps track of the recurrent state $h_t$.) Together, they form the agent’s internal belief over future evolutions. Note that the the latent $z_t$ from the observation encoder and the latent $\hat{z}_{t+1}$ from the predictive belief encoder share the same decoder.

The model is trained with a reconstruction loss term to ensure accurate observation predictions and a self-consistency regularization across imagined rollouts to ensure temporal predictive consistency:

\vspace{-10pt}
\begin{equation}
    \begin{aligned}
    \mathcal{L}(\theta, \phi; o_{0:T}, u_{0:T}) = \quad 
    \Bigg[
    \alpha 
    \underbrace{
    \sum_{t=0}^{T}
    \,
    \mathbb{E}_{q_\phi(z_t \mid o_t)}
    \left[
    \log p_\theta(o_t \mid z_t)
    \right]
    }_{\text{reconstruction loss}}
    \Bigg] \\[1pt]
    -
    \Bigg[
    \beta 
    \underbrace{
    \sum_{t=0}^{T-1}\,
    D_{\mathrm{KL}}
    \left(
    q_\phi(\hat{z}_{t+1} \mid \hat{o}_{t+1}) \;\|\;
    f_\theta(\hat{z}_{t} \mid u_{t})
    \right)
    }_{\text{latent belief consistency}}
    \Bigg]
    \end{aligned}
\label{eq:loss-terms}
\end{equation}

The first term in the loss function in Eq. \ref{eq:loss-terms} penalizes disagreement between reconstructed and ground truth image observations.
The second term penalizes disagreement between the encoded predicted image at the next timestep and the dynamics-predicted next latent state through a KL-divergence term during training.

In this work, we instantiate belief divergence in two complementary ways: between predictive latent beliefs to encourage exploration of latent dynamics, and between predictive and observation-updated beliefs to encourage environment interactions. 

Actions are selected using a receding-horizon model predictive control (MPC) framework operating directly on the learned predictive belief model; specifically, we use cross entropy method (CEM).

The closed loop planner proceeds as follows:
\begin{enumerate}
    \item Encode the current image $o_t$ to initialize or update the current latent belief.
    \item Roll forward the learned stochastic predictive belief $f_\theta (z_t, u_t)$ over horizon $H$ for each of $N$ candidate action sequences $u_{t:t+H}$.
    \item Evaluate the intrinsic belief divergence objective $J(o_t, u_{t:t+H})$ over imagined belief trajectories.
    \item Select the highest-value action sequence using the Cross Entropy Method (CEM) and execute its first action.
    \item Observe the next image at $o_{t+1}$, and repeat this planning procedure at each timestep over the episode horizon $T$.
\end{enumerate}

As shown on the left in the attached figure (network_architecture.png), the model consists of 4 parts: \begin{enumerate}
% [label=(\arabic*)]
    \item a static, per-frame image encoder that maps observations $o$ to stochastic latent states $z$;
    \item a deterministic recurrent dynamics model that forward propagates temporal states $h_{t+1}$ conditioned on actions $u_t$;
    \item a stochastic latent transition model that represents the internal predictive belief of future latent states $\hat{z}_{t+1}$;
    and
    \item a static, per-frame image decoder that reconstructs observations $\hat{o}$ from latent states $z$.
\end{enumerate}

Pixel objective, defined as:
\begin{equation}
    J_P(o_t, u_{t}) =  D_{\mathrm{KL}}
    \left[\left(
    q_\phi(\hat{z}_{t+1} \mid \hat{o}_{t+1}) \;\|\;
     p_\theta(\hat{z}_{t+1} \mid g_\theta(h_{t}, z_{t}, u_{t})) \right) \right ]
\end{equation}

Dynamics objective measures the difference between the latent state from the predictive belief $\hat{z}_{t+1}$ and the current encoded latent state $\hat{z}_t$:
\begin{equation}
    J_D(o_t, u_{t}) =  D_{\mathrm{KL}}
    \left[
     p_\theta(\hat{z}_{t+1} \mid g_\theta(h_{t}, z_{t}, u_{t})) \;\|\;
     q_\phi(\hat{z}_{t} \mid \hat{o}_{t}) \right ]
\end{equation}

---

### 2. Goals for This Review

* [ ] Remove dead / unused code (except that which is used for hardware experiments)
* [ ] Simplify overly complex logic (if needed)
* [ ] Improve readability and structure. MAINTAIN SIMILAR LOGIC
* [ ] If applicable, ensure modularity.
* [ ] Improve naming (variables, functions, classes)
* [ ] Add / improve docstrings and comments
* [ ] Optimize performance (only if safe)
* [ ] Ensure consistency (style, formatting)
* [ ] Identify potential bugs or edge cases
**Priority order (important):**
1. make sure the core logic of the code is unchanged. I do not want to go back and debug things.
2. Make sure the code and algorithm is correct.
3. Improve performance (e.g., by broadcasting) if possible.
4. Improve readability (docstrings) and modularity.

---

### 3. Constraints & Non-Negotiables

* Do NOT change core algorithmic behavior unless explicitly asked
* Preserve experimental logic, even if it looks unusual. ask questions if you are remotely uncertain.
* Keep compatibility with:
  * Python version:
  * Libraries / frameworks:
* Avoid introducing new dependencies unless necessary
* Maintain reproducibility of results

---

### 4. Context About the Codebase

* KL divergence as the objective function is intentionally implemented in two different ways, one for the pixel objective and one for the dynamics objective.
* There is a hardware version of the closed loop trainer.
* DO NOT DELETE ANY DATA (videos, json, etc)

---

### 5. Repository Structure

```
/E2C (project root)
  /config - this contains a bunch of different configurations, mostly related to different datasets. Naming conventions are important for the current logic.
  /data - datasets for the RL environments we did benchmarks on. we do not need particle in gravity code.
  /figures - mostly obsolete code 
  /notebooks - mostly old code related to previous iterations of this project.
  /src - main folder with code
    /data_gen - responsible for creating datasets for warmstarting model. also has scripts to evaluate results.
    /model - code for convolutional encoder, decoder, rssm, etc
    important scripts: eval.py, main.py, train.py, trainer.py, etc
  /tests - mostly obsolete, but double check
  README.md - OLD. DO NOT READ THIS
  ...
```

### 6. Code Style Preferences

Feel free to give:
* Architectural critique
* Suggestions for scaling or generalization
* Identification of hidden technical debt
* Suggestions for improving research clarity / reproducibility

* Keep current coding style and convention as much as possible
* Try to Follow PEP8
* Type hints: [optional, but nice to have]
* Comment level: (minimal)

---

### 7. Output Format You Want

* [ ] Inline suggestions with explanations, along with cleaned up suggestions. This shoould be similar to Git-style diff patches
* [ ] Bullet list of issues + fixes. Put this in a markdown file for me to review.
ALL CHANGES SHOULD
* Include explanations. put this in the same markdown file as the bulleted list



