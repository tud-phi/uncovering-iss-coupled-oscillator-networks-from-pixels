# Rebuttal 2 - Reviewer 1

> Finally, it's the same story for deriving stability from the single oscillators to the network. Of course you can make the stiffness matrix negative definite to achieve a non-stable network but again, if all parameters are positive (definite), meaning defined so that you would expect the system to be stable, I think that the system is stable. I'd change me opinion if you can provide a counter example where the parameters are positive (definite) but would lead to an unstable system / network.

We are now ready to go back to our technical discussion, with the discussed counterexamples. These, hopefully, also better illustrate our point #3. Essentially, our argument here is that even positive-definite stiffness & damping matrices are not sufficient to claim global asymptotic stability. This should answer the Reviewer's direct technical question.

In the following, we will give two examples of modifications to the CON network (in the original coordinates) would have, according to the Reviewer's argument, appeared to be stable, but a global stability proof would be hard.

## Example 1: A system without a valid potential energy

We consider the slightly modified network dynamics $\ddot{x} + K x + D \dot{x} + W^{-1} \tanh(Wx + b) = 0$, where $K, D, W \succ 0$ are positive definite matrices.
For $\tau_\mathrm{pot} = -K x + D \dot{x} + -W^{-1} \tanh(Wx + b)$ to be a valid potential force, it would need to satisfy the property $\frac{\partial \tau_\mathrm{pot}}{\partial x} = \left ( \frac{\partial \tau_\mathrm{pot}}{\partial x} \right )^\mathrm{T}$. Therefore, we derive $\frac{\partial \tau_\mathrm{pot}}{\partial x} = $

However, this is not the case for the given dynamics, as $$

## Example 2: A system with multiple equilibria

We consider the slightly modified network dynamics $\ddot{x} + K x + D \dot{x} + W^{-1} \tanh(Wx + b) = 0$, where $K, D, W \succ 0$ are positive definite matrices. The equilibrias of this system are given by the characteristic equation $K \bar{x} + W^{-1} \tanh(W \bar{x} + b) = 0$.
For the system to be globally asymptotically stable, it would need to have a single equilibrium point. However, if we simulate the system with $K = [[5.0, -2.2], [-2.2, 1.0] \succ 0$ (positive-definite as symmetric and positive eigenvalues $0.027$ and $5.973$), $D = \mathrm{diag}(0.2, 0.2) \succ 0$, $W = [[1.0, -2.2], [-2.2, 5.0]] \succ 0$ (positive-definite as symmetric and positive eigenvalues $0.027$ and $5.973$), we notice that the system is actually bistable with two attractors at $\bar{x}_1 = [-212.6, -475.2]$ and $\bar{x}_2 = [212.6, 475.2]$. This is illustrated in the time series plot and phase portrait attached in https://anonymous.4open.science/r/neurips24-20062-rebuttal-7770. Therefore, the system is **not** globally asymptotically stable.