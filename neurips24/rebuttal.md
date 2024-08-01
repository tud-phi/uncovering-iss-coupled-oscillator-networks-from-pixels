# NeurIPS 2024 Submission #20062 Rebuttal

## Response to Reviewer Ys99 (R1)

We thank very much the Reviewer for the kind words, for the interest in our research activities, and for the
very insightful comments that they have provided. In the following, we will respond to the weakness and questions raised by the Reviewer.

### Is stability of CON trivial?

> <cite>R1</cite>: Theorem 1 seems to be trivial as it elaborates on the (nonlinear) coupling of passive systems (damped mass-spring systems) which is always passive since it does not “produce” any energy and thus, is inherently stable. (Same for Theorem 2). In this sense, the theoretical contribution seems limited. Is there anything I missed?

### PID vs. PD

> <cite>R1</cite>: The statement in Line 266 “PID controller has several well-known drawbacks, such […] steady-state errors (in case the integral gain is chosen to be zero)” is misleading as in this case we would say it’s a PD controller.

### Performance and Limitations of CON

> <cite>R1</cite>: The main contribution seems to be the new network structure that allows to exploit the potential energy for energy shaping methods. The VAE and controller are existing methods. Thus, a more detailed elaboration on the performance and limitation of CONs would be beneficial as it is evaluated for soft robotics data sets only.

### Inference time of CON

> <cite>R1</cite>: Can you add the inference time for the different methods in Table 1? If that’s not possible, could you give some general comments on the inference time of CONs?

### Oscillations of the controller with FF term. 

> <cite>R1</cite>: Figure 3 visualizes that the controller with feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question. When tuning the gains PID-like controllers, there naturally always exists a trade-off between transient behaviour (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time, but still allow for stable behaviour. 
Importantly, we kept the gains for the _P-satI-D_ and 

## Response to Reviewer W9L3 (R2)

We thank the Reviewer for the careful reading and the encouraging comments. 
In the following, we will respond to the questions raised by the reviewer, which also relate to the weaknesses mentioned by the reviewer.

### Application of CON to non-soft robots

> <cite>R2</cite>: Have you tried the method on other (non-soft) robots? Do you have any intuition on how your method would perform on environments with many contact forces?

### Application of CON to non-physical systems

> <cite>R2</cite>: Have you considered non-physical systems?

