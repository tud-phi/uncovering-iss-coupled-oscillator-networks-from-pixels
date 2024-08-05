# Response to Reviewer Ys99 (R1)

We thank the Reviewer very much for the kind words, for their interest in our research activities, and for the very insightful comments that they have provided. In the following, we will respond to the weaknesses and questions raised by the Reviewer.

## Is the stability of CON trivial?

> <cite>R1</cite>: Theorem 1 seems to be trivial as it elaborates on the (nonlinear) coupling of passive systems (damped mass-spring systems), which is always passive since it does not “produce” any energy and thus, is inherently stable. (Same for Theorem 2). In this sense, the theoretical contribution seems limited. Is there anything I missed?

We thank the Reviewer for their question and for allowing us to further elaborate on this topic. In short, we differ with the Reviewer's technical arguments, and, consequently, with their conclusions on the triviality of Theorem 1. 
In essence, we recognize that the Reviewer's arguments are well funded in the context of port-Hamiltonian theory. However, substantial extra properties/assumptions are needed to yield to the Reviewer's conclusion. Thus, the PH way of proving Theorem 1 would be an interesting and proper one, but it would definitely not result in the trivial proof the Reviewer is hinting at.
More precisely, we point out two main points of technical disagreement, which we list below.

### (A): Passivity does not imply Stability
Passivity, even in the scalar case, does not imply (global) stability. Additional assumptions on the (global) convexity of the energy landscape are needed, which would be hard to impose. As a trivial counter example, think of a damped mass placed on the top of an infinite hill. 
To further illustrate this point, we propose two simple passive-but-not-globally-stable CONs, that deviate slightly from the assumptions in Theorem 1 and 2.

1. Take a passive CON as stated in Eq. (2) in the scalar case with $K=1$ and $D=0.4$. Conflicting with the assumptions in Theorem 1, we select $W=-5$. Now, depending on the choice of $b$, this scalar system either has 1, 2, or 3 equilibria (see Fig. R1 in global response). All global stability guarantees are lost if the system has more than one attractor. 
2. We study a scalar CON with $K=-1$, positive damping $D=0.4$, and $W = 0$, $b=0$ resulting in the EOM $\ddot{x} + K x + D \dot{x} = \tau$. We take the passive output $o = \dot{x}$, and we prove passivity according to Def. 6.3 (Khalil, 2001) using the storage function $V(x) = K x^2 + \dot{x}^2$: $\dot{V}(x, \tau) = \tau \dot{x} - D \dot{x}^2 \leq \tau o. $
However, this system is unstable, as it can be easily assessed by looking at the linearization at the equilibrium. 2. Fig. R2 reports the globally repulsive vector field, which exhibits a single unstable equilibrium at the origin.

Therefore, we conclude that passivity is not sufficient to prove global asymptotic stability.

### (B): Stable harmonic oscillators do not imply Stable Networks

Even if the individual systems (i.e., the harmonic oscillators) are stable, care needs to be taken when coupling them not to create an unstable network.

We illustrate this with the following example: consider a CON of dimensionality two with $K = [[1.0, -1.4],[-1.4, 1.0]]$, $D = \mathrm{diag}(0.4, 0.4)$ $W = \mathrm{diag}(3, 3)$, 
It can be easily shown that the oscillators individually with the EoM $\ddot{x}_i + k x_i + d \dot{x}_i + \tanh(w x_i)= 0,$ where $k=1$, $d=0.4$, and $w=3$ are globally asymptotically stable. 

However, the linear stiffness matrix is negative definite, and therefore, the system is not globally asymptotically stable. This is illustrated in Fig. R3 of the global response.


## PID vs. PD

> <cite>R1</cite>: The statement in Line 266 “PID controller has several well-known drawbacks, such […] steady-state errors (in case the integral gain is chosen to be zero)” is misleading as in this case we would say it’s a PD controller.

We thank the Reviewer for their comment and for pointing out the mistake. We agree that this sentence is indeed badly written and confusing. In the final version of the paper, we will remove the subsentence "_steady-state errors (in case the integral gain is chosen to be zero)_".

## Performance and Limitations of CON

> <cite>R1</cite>: The main contribution seems to be the new network structure that allows to exploit the potential energy for energy shaping methods[MOU3]. The VAE and controller are existing methods. Thus, a more detailed elaboration on the performance and limitation of CONs would be beneficial as it is evaluated for soft robotics data sets only.

We thank the Reviewer for their comment. We want to stress that the proposed network allows to impose a beneficial inductive bias for conserving global stability and ISS. Therefore, we consider the two proofs (i.e., GAS and ISS) to be important contributions of the paper.
Furthermore, we also regard the closed-form approximation of the CON dynamics as an important tool for deploying oscillator networks in practice.

## Inference time of CON

> <cite>R1</cite>: Can you add the inference time for the different methods in Table 1? If that’s not possible, could you give some general comments on the inference time of CONs?

We thank the reviewer for their question about the inference time of the various methods. We note that the number of training steps per second of all methods included in Table 1 was already reported in the original submission in Table 4 of Appendix D.

For this rebuttal, we performed additional evaluations of the inference time (i.e., without computation of loss function and gradient descent) of the various models and report the results in Table R2 of the global response PDF.

## Oscillations of the controller with FF term. 

> <cite>R1</cite>: Figure 3 visualizes that the controller with a feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question and for raising the topic. When tuning the gains PID-like controllers, a trade-off naturally exists between transient behavior (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time but allowed for stable behavior. The oscillations are caused by a combination of (a) the underdamped nature of the system and (b) the magnitude of the proportional term.
Importantly, to have a fair comparison, we kept the gains of the feedback controller the same for both the _P-satI-D_ and _P-satI-D + FF_ cases. A higher proportional term is beneficial for the response time (and the performance) of the _P-satI-D_,
while it leads to overshooting and oscillations in the _P-satI-D + FF_ case. We stress that this is not an inherent problem of the feedback controller but can be mitigated by tuning the feedback gains differently.
For this rebuttal, we tuned a controller with reduced proportional and increased damping term and the results, included as Fig. R4 in the global response PDF, show that the oscillations and overshooting are both significantly reduced.