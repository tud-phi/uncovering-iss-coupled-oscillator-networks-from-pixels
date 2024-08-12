# Rebuttal 2 Reviewer 1

We appreciate the Reviewer's effort and time responding to our rebuttal and the interesting comments.

> Finally, it's the same story for deriving stability from the single oscillators to the network. Of course you can make the stiffness matrix negative definite to achieve a non-stable network but again, if all parameters are positive (definite), meaning defined so that you would expect the system to be stable, I think that the system is stable. I'd change me opinion if you can provide a counter example where the parameters are positive (definite) but would lead to an unstable system / network.

We agree with that Reviewer that global asymptotic stability usually requires very strong assumptions that lead to a globally convex potential energy landscape. The most common way to establish a convex potential energy landscape is to the design the system such that it has positive (definite) stiffness.
Still, we are of the opinion that the stability proof is not trivial and provides value to both the paper and the larger research community. Specifically, we state the following reasons:

1. Minor modifications to the network formulation while keeping the matrices positive definite would lead to asymmetric potential forces and, consequently, would not allow us to conduct a stability analysis.
2. To the best of our knowledge, we are the first to provide in this community explicit input-to-state stability guarantees including convergence rates into the region of attraction (Theorem 2). This contribution makes it possible for practitioners to design/sample/constrain the system parameters such that certain convergence rates are achieved (and guaranteed).
