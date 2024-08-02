# NeurIPS 2024 Submission #20062 Rebuttal

## Response to Reviewer Ys99 (R1)

We thank very much the Reviewer for the kind words, for the interest in our research activities, and for the
very insightful comments that they have provided. In the following, we will respond to the weakness and questions raised by the Reviewer.

### Is stability of CON trivial?

> <cite>R1</cite>: Theorem 1 seems to be trivial as it elaborates on the (nonlinear) coupling of passive systems (damped mass-spring systems) which is always passive since it does not “produce” any energy and thus, is inherently stable. (Same for Theorem 2). In this sense, the theoretical contribution seems limited. Is there anything I missed?

We thank the Reviewer for their question. In our opinion, these theoretical (global) stability results are not trivial for the following reasons:

1. Already small modifications to the assumptions stated in Theorem 1 and 2 would lead to multistability and, with that, lead to the loss of all global stability guarantees. As an illustrative example, we consider CON as stated in Eq. (2) in the scalar case with $K=1$ and $D=0.4$ (i.e., a typical harmonic oscillator). However, conflicting with the assumptions in Theorem 1, we select $W=-5$. Now, ddepending on the choice of $b$, this scalar system either has 1, 2 or 3 equilibria (see PDF in global response). If the system has more than 2 equilibria, the global stability guarantees are lost. This motivates why we have to carefully choose the nonlinear coupling between the (passive) harmonic oscillators.

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
We considered the same setting as in Table 4 of Appendix D: latent dimension $n_z = 8$, input images (from the PCC-NS-2 dataset) of size $32 \times 32$ px, rollout of the dynamics for 101 time steps (equivalent to 2.02 seconds), encoding the input image at each time step, and decoding the latent state at each time step.
In an actual deployment scenario, we would likely perform a single prediction at a time. Therefore, we set the batch size to 1.
We executed the rollout 5000 times for each method on an Nvidia RTX 3090, measured the inference time and averaged the results.
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

It is interesting note that the main speedup by CFA-CON is achieved during training, where we see almost 2x as many steps per second. During rollout, CFA-CON is roughly 50% slower than CON-M.
We attribute this difference to the small batch size, execution on the CPU, and no need for backpropagation through the ODE integrator during inference vs. training.

### Oscillations of the controller with FF term. 

> <cite>R1</cite>: Figure 3 visualizes that the controller with feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question and raising the topic. 
When tuning the gains PID-like controllers, there naturally always exists a trade-off between transient behaviour (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time, but still allow for stable behaviour.
The oscillations are caused by a combination of (a) the underdamped nature of the system and (b) the magnitude of the proportional term.
Importantly, we kept the gains of the feedback controller the same for both the _P-satI-D_ and _P-satI-D + FF_ case.
Actually, a higher proportional term is beneficial for the response time (and with that the performance) of the _P-satI-D_,
while it leads to overshooting and oscillations in the _P-satI-D + FF_ case.
We stress that this is not an inherent problem of the feedback controller, but can be mitigated through a different tuning of the feedback gains.
For this rebuttal, we have simulated a _P-satI-D + FF_ controller with $K_\mathrm{p} = 0$, $K_\mathrm{i} = 2$, and $K_\mathrm{d} = 0.1$, which means that we set the proportional term to zero and increased the damping factor.
The results, that are included in the PDF of the global response, show that the oscillations and the overshooting are both significantly reduced.
We also note that the issues raised by the Reveiwer would not be present for a system with higher damping.


## Response to Reviewer W9L3 (R2)

We thank the Reviewer for the careful reading and the encouraging comments. 
In the following, we will respond to the questions raised by the reviewer, which also relate to the weaknesses mentioned by the reviewer.

### Application of CON to non-soft robots

> <cite>R2</cite>: Have you tried the method on other (non-soft) robots? Do you have any intuition on how your method would perform on environments with many contact forces?

### Application of CON to non-physical systems

> <cite>R2</cite>: Have you considered non-physical systems?

