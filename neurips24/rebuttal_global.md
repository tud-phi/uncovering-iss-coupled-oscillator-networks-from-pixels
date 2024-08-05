# Global Rebuttal

## Inference time of CON (R1)

> <cite>R1</cite>: Can you add the inference time for the different methods in Table 1? If that’s not possible, could you give some general comments on the inference time of CONs?

We thank the reviewer for their question about the inference time of the various methods. We note that the number of training steps per second of all methods included in Table 1 was already reported in the original submission in Table 4 of Appendix D.

For this rebuttal, we performed additional evaluations of the inference time (i.e., without computation of loss function and gradient descent) of the various models and report the results in Table R2 of the global response PDF.

## Oscillations of the controller with FF term. (R1)

> <cite>R1</cite>: Figure 3 visualizes that the controller with a feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question and for raising the topic. When tuning the gains PID-like controllers, a trade-off naturally exists between transient behavior (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time but allowed for stable behavior. The oscillations are caused by a combination of (a) the underdamped nature of the system and (b) the magnitude of the proportional term.
Importantly, to have a fair comparison, we kept the gains of the feedback controller the same for both the _P-satI-D_ and _P-satI-D + FF_ cases. A higher proportional term is beneficial for the response time (and the performance) of the _P-satI-D_,
while it leads to overshooting and oscillations in the _P-satI-D + FF_ case. We stress that this is not an inherent problem of the feedback controller but can be mitigated by tuning the feedback gains differently.
For this rebuttal, we tuned a controller with reduced proportional and increased damping term and the results, included as Fig. R4 in the global response PDF, show that the oscillations and overshooting are both significantly reduced.