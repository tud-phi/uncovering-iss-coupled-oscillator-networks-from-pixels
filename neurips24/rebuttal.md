# NeurIPS 2024 Submission #20062 Rebuttal

## Response to Reviewer Ys99 (R1)

We thank the Reviewer very much for the kind words, for their interest in our research activities, and for the
very insightful comments that they have provided. In the following, we will respond to the weaknesses and questions raised by the Reviewer.

### Is the stability of CON trivial?

> <cite>R1</cite>: Theorem 1 seems to be trivial as it elaborates on the (nonlinear) coupling of passive systems (damped mass-spring systems), which is always passive since it does not “produce” any energy and thus, is inherently stable. (Same for Theorem 2). In this sense, the theoretical contribution seems limited. Is there anything I missed?

We thank the Reviewer for their question and for allowing us to further elaborate on this topic. We differ with the Reviewer on the following points:

#### (A): Passivity -> Stability?
Passivity, even in the scalar case, does not imply (global) stability. We illustrate this with two simple examples, that deviate slightly from the assumptions in Theorem 1 and 2.

1. We consider a passive CON as stated in Eq. (2) in the scalar case with $K=1$ and $D=0.4$ (i.e., a typical harmonic oscillator). However, conflicting with the assumptions in Theorem 1, we select $W=-5$. Now, depending on the choice of $b$, this scalar system either has 1, 2, or 3 equilibria (see Fig. R1 in global response). All global stability guarantees are lost if the system has more than one attractor. 
2. We study a scalar CON with $K=-1$, positive damping $D=0.4$, and $W = 3$, $b=0$ (i.e., a harmonic oscillator with negative stiffness) with the EOM
$\ddot{x} + K x + D \dot{x} = \tau$. After defining the output $o = \dot{x}$, we prove passivity according to Definition 6.3 (Khalil, 2001) using the storage function $V(x) = K x^2 + \dot{x}^2$:
$\dot{V}(x, \tau) = \tau \dot{x} - D \dot{x}^2 \leq \tau o. $
Fig. R2 demonstrates how this passive system is globally unstable and only exhibits an unstable equilibrium at the origin.

Therefore, we conclude that passivity is not sufficient to prove global asymptotic stability.

#### (B): Stable harmonic oscillators -> Stable Networks?

Even if the individual systems (i.e., the harmonic oscillators) are stable, care needs to be taken when coupling them not to create an unstable network.
We illustrate this with the following example: consider a CON of dimensionality two with $K = [[1.0, -1.4],[-1.4, 1.0]]$, $D = \mathrm{diag}(0.4, 0.4)$ $W = \mathrm{diag}(3, 3)$, 
It can be easily shown that the oscillators individually with the EoM
$\ddot{x}_i + k x_i + d \dot{x}_i + \tanh(w x_i)= 0,$
where $k=1$, $d=0.4$, and $w=3$ are globally asymptotically stable. However, the linear stiffness matrix is negative definite,
and therefore, the system is not globally asymptotically stable. This is illustrated in Fig. R3 of the global response.


### PID vs. PD

> <cite>R1</cite>: The statement in Line 266 “PID controller has several well-known drawbacks, such […] steady-state errors (in case the integral gain is chosen to be zero)” is misleading as in this case we would say it’s a PD controller.

We thank the Reviewer for their comment and for pointing out the mistake. We agree that this sentence is indeed badly written and confusing.
In the final version of the paper, we will remove the subsentence "_steady-state errors (in case the integral gain is chosen to be zero)_".

### Performance and Limitations of CON

> <cite>R1</cite>: The main contribution seems to be the new network structure that allows to exploit the potential energy for energy shaping methods. The VAE and controller are existing methods. Thus, a more detailed elaboration on the performance and limitation of CONs would be beneficial as it is evaluated for soft robotics data sets only.

### Inference time of CON

> <cite>R1</cite>: Can you add the inference time for the different methods in Table 1? If that’s not possible, could you give some general comments on the inference time of CONs?

We thank the reviewer for their question about the inference time of the various methods. 
We note that the number of training steps per second of all methods included in Table 1 was already reported in the original submission in Table 4 of Appendix D.

For this rebuttal, we performed additional evaluations of the inference time (i.e., without computation of loss function and gradient descent) of the various models.
We considered the same setting as in Table 4 of Appendix D: latent dimension $n_z = 8$, input images (from the PCC-NS-2 dataset) of size $32 \times 32$ px, the rollout of the dynamics for $101$ time steps (equivalent to 2.02 seconds), encoding the input image at each time step, and decoding the latent state at each time step.
In an actual deployment scenario, we would likely perform a single prediction at a time. Therefore, we set the batch size to 1.
We executed the rollout 5000 times for each method on an Nvidia RTX 3090, measured the inference time, and averaged the results.
The results can be found in the table below. We plan to add this column to Table 4 in Appendix D of the final paper.

| Method        | Inference time [ms] |
|---------------|---------------------|
| RNN           | 02.6                |
| GRU           | 03.2                |
| coRNN         | 02.7                |
| NODE          | 50.2                |
| MECH-NODE     | 50.3                |
| CON-S (our)   | 50.2                |
| CON-M (our)   | 60.1                |
| CFA-CON (our) | 13.6                |

### Oscillations of the controller with FF term. 

> <cite>R1</cite>: Figure 3 visualizes that the controller with a feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question and for raising the topic. 
When tuning the gains PID-like controllers, a trade-off naturally exists between transient behavior (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time but allowed for stable behavior.
The oscillations are caused by a combination of (a) the underdamped nature of the system and (b) the magnitude of the proportional term.
Importantly, we kept the gains of the feedback controller the same for both the _P-satI-D_ and _P-satI-D + FF_ cases.
A higher proportional term is beneficial for the response time (and the performance) of the _P-satI-D_,
while it leads to overshooting and oscillations in the _P-satI-D + FF_ case.
We stress that this is not an inherent problem of the feedback controller but can be mitigated by tuning the feedback gains differently.
For this rebuttal, we have simulated a _P-satI-D + FF_ controller with $K_\mathrm{p} = 0$, $K_\mathrm{i} = 2$, and $K_\mathrm{d} = 0.1$, which means that we set the proportional term to zero and increased the damping factor.
The results, included as Fig. R4 in the global response PDF, show that the oscillations and overshooting are both significantly reduced.
We also note that the issues raised by the reviewer would not be present for a system with higher damping.


## Response to Reviewer W9L3 (R2)

We thank the Reviewer for the careful reading and the encouraging comments. 
In the following, we will respond to the questions raised by the reviewer, which also relate to the weaknesses mentioned by the reviewer.

### Application of CON to non-soft robots

> <cite>R2</cite>: Have you tried the method on other (non-soft) robots? Do you have any intuition on how your method would perform on environments with many contact forces?

### Application of CON to non-physical systems

> <cite>R2</cite>: Have you considered non-physical systems?

We thank the Reviewer for their interest in this topic. While we do think that the CON / CFA-CON models could be potentially useful
in other applications where strong stability guarantees are required, we strive to focus in this paper on learning
the dynamics of physical/mechanical systems as we can here leverage (a) the shared stability characteristics between
the original and the latent space systems, and (b) exploit the mechanical structure of the CON model for control.
To make this focus clear, we will, in the final paper, modify the sentence starting on line 69 to (with changes marked in **bold**):

_We resolve all the above-mentioned challenges by proposing Coupled Oscillator Networks (CONs), a new formulation of a coupled oscillator network that is inherently Input-to-State Stability (ISS) stable, **for learning the dynamics of physical systems,** and subsequently exploiting its structure for model-based control in latent space._

