# Rebuttal 2 - Reviewer 1

> Finally, it's the same story for deriving stability from the single oscillators to the network. Of course you can make the stiffness matrix negative definite to achieve a non-stable network but again, if all parameters are positive (definite), meaning defined so that you would expect the system to be stable, I think that the system is stable. I'd change me opinion if you can provide a counter example where the parameters are positive (definite) but would lead to an unstable system / network.

We are now ready to go back to our technical discussion, with the discussed counterexamples. These, hopefully, also better illustrate our point #3. Essentially, our argument here is that even positive-definite stiffness & damping matrices are not sufficient to claim global asymptotic stability. This should answer the Reviewer's direct technical question.

In the following, we will give two examples of modifications to the CON network (in the original coordinates) that would have, according to the Reviewer's argument, appeared to be stable, but for which global stability proof would be hard to prove.

## Example 1: A system with multiple equilibria and without a valid potential energy

We consider the slightly modified network dynamics $\ddot{x} + K x + D \dot{x} + W^{-1} \tanh(Wx + b) = 0$, where $K, D, W \succ 0$ are positive definite matrices.

The equilibrias of this system are given by the characteristic equation $K \bar{x} + W^{-1} \tanh(W \bar{x} + b) = 0$.
For the system to be globally asymptotically stable, it would need to have a single equilibrium point. However, if we simulate the system with $K = [[5.0, -2.2], [-2.2, 1.0] \succ 0$ (positive-definite as symmetric and positive eigenvalues $0.027$ and $5.973$), $D = \mathrm{diag}(0.2, 0.2) \succ 0$, $W = [[1.0, -2.2], [-2.2, 5.0]] \succ 0$ (positive-definite as symmetric and positive eigenvalues $0.027$ and $5.973$), we notice that the system is actually bistable with two attractors at $\bar{x}_1 = [-212.6, -475.2]$ and $\bar{x}_2 = [212.6, 475.2]$. This is illustrated in the time series plot and phase portrait attached in https://anonymous.4open.science/r/neurips24-20062-rebuttal-7770. Therefore, the system is **not** globally asymptotically stable.

For $\tau_\mathrm{pot} = -K x -W^{-1} \tanh(Wx + b)$ to be a valid potential force, it would need to satisfy the property $\frac{\partial \tau_\mathrm{pot}}{\partial x} = \left ( \frac{\partial \tau_\mathrm{pot}}{\partial x} \right )^\mathrm{T}$. Therefore, we derive $\frac{\partial \tau_\mathrm{pot}}{\partial x} = W^{-1} \mathrm{diag}(\mathrm{sech}^2(Wx + b)) W$. Its transpose is given by $\left ( \frac{\partial \tau_\mathrm{pot}}{\partial x} \right )^\mathrm{T} = W^\mathrm{T} \mathrm{diag}(\mathrm{sech}^2(Wx + b)) W^{-\mathrm{T}}$. Therefore, $\tau_\mathrm{pot}$ only stems from a potential iff $W^{-\mathrm{T}} = W \rightarrow W^\mathrm{T} W = \mathbb{I}$ (i.e., $W$ is orthogonal). This is not the case for a general positive-definite $W$.

This example motivates that even if the matrices are positive-definite, the system can still have multiple equilibria (i.e., lose global asymptotic stability) and not have a valid potential energy function.

## Example 2: A system where the $\mathcal{W}$-coordinate transformation does not help

The proof underlying Theorem 1 is only (relatively) simple as the coordinate transformation $x_\mathrm{w} = W x$ helps us to identify the potential energy function in the $\mathcal{W}$ coordinates, where the hyperbolic nonlinearity operates elementwise. However, this is not always possible. For example, consider the system $\ddot{x} + K x + D \dot{x} + W^{\mathrm{T}} \tanh(Wx + b) = 0$, where $K, D, W \succ 0$ are positive definite matrices. 

Remark: In contrast to Example 1, this system actually has a single equilibrium point. 

Unfortunately, the unactuated dynamics in $\mathcal{W}$ coordinates are now given by $M_\mathrm{w} \ddot{x}_\mathrm{w} + K_\mathrm{w} W + D \dot{x}_\mathrm{w} + W^\mathrm{T} \tanh(x_\mathrm{w} + b)$, for which we cannot easily derive a potential energy function (i.e., integrate), as the hyperbolic term is not (easily) separable.

This example motivates why we chose the proposed CON network architecture such that we can (easily) derive the kinetic and potential energy terms, subsequently prove GAS & ISS, and perform model-based control.
