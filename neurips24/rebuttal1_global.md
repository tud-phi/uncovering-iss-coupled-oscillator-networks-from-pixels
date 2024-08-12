# Global Rebuttal

## Performance of CON on non-soft-robotic datasets (R1 & R2)

> R1: Thus, a more detailed elaboration on the performance and limitations of CONs would be beneficial as it is evaluated for soft robotics data sets only.

> R2: Have you tried the method on other (non-soft) robots?

We thank both reviewers for their interest in the performance of CON on non-soft-robotic datasets.
For this rebuttal, we compared the performance of CON against the baseline method on three additional mechanical, non-soft-robotic datasets: a mass-spring with friction (_M-SP+F_) (i.e., a damped harmonic oscillator), a single pendulum with friction (_S-P+F_), and a double pendulum with friction (_D-P+F_). These datasets are based on an interesting publication by Botev et al. (2021) [25], which appeared in the _NeurIPS 2021 Track on Datasets and Benchmarks_ and benchmarks various models for learning latent space dynamics.

The results, which we will refer to as Table R1 of the global response PDF, show that the NODE model slightly outperforms the CON network on the _M-SP+F_ and _S-P+F_. However, as the datasets do not consider system inputs, we can remove the input mapping from all models (e.g., RNN, GRU, coRNN, CON, and CFA-CON). With that adjustment, the CON network has the fewest parameters among all models and particularly two orders of magnitude less than the NODE model. Therefore, we find it very impressive that the CON network is roughly on par with the NODE model.
For the _D-P+F_ dataset, we can conclude that the CFA-CON model offers the best performance across all methods.
Finally, most of the time, the CON & CFA-CON networks outperform the other baseline methods that have more trainable parameters.

## Inference time of CON (R1)

> R1: Can you add the inference time for the different methods in Table 1? If that’s not possible, could you give some general comments on the inference time of CONs?

We thank the reviewer for their question about the inference time of the various methods. We note that the number of training steps per second of all methods included in Table 1 was already reported in the original submission in Table 4 of Appendix D.

For this rebuttal, we performed additional evaluations of the inference time (i.e., without computation of loss function and gradient descent) of the various models and report the results in Table R2 of the global response PDF.

## Oscillations of the controller with FF term. (R1)

> R1: Figure 3 visualizes that the controller with a feed-forward part leads to heavy oscillations in the systems. Is that due to a poorly tuned controller or do you see the reason in the CON model?

We thank the Reviewer for their question and for raising the topic. When tuning the gains PID-like controllers, a trade-off naturally exists between transient behavior (e.g., oscillations and overshooting) and response time.
In this case, we chose gains that minimized the response time but allowed for stable behavior. The oscillations are caused by a combination of (a) the underdamped nature of the system and (b) the magnitude of the proportional term.
Importantly, to have a fair comparison, we kept the gains of the feedback controller the same for both the _P-satI-D_ and _P-satI-D + FF_ cases. A higher proportional term is beneficial for the response time (and the performance) of the _P-satI-D_,
while it leads to overshooting and oscillations in the _P-satI-D + FF_ case. We stress that this is not an inherent problem of the feedback controller but can be mitigated by tuning the feedback gains differently.
For this rebuttal, we tuned a controller with reduced proportional and increased damping term and the results, included as Fig. R4 in the global response PDF, show that the oscillations and overshooting are both significantly reduced.
