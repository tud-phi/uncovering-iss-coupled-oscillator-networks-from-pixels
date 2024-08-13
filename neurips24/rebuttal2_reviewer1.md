# Rebuttal 2 - Reviewer 1

> Finally, it's the same story for deriving stability from the single oscillators to the network. Of course you can make the stiffness matrix negative definite to achieve a non-stable network but again, if all parameters are positive (definite), meaning defined so that you would expect the system to be stable, I think that the system is stable. I'd change me opinion if you can provide a counter example where the parameters are positive (definite) but would lead to an unstable system / network.

We appreciate the direct nature of the Reviewer's comment, which rises indeed a compelling point with which however we still disagree. In our next response, we will try again to convince the Reviewer of our argument, although proving that something is _not trivial_ is quite a challenging task, as triviality is very subjective.

So, before delving into it, we would like to take the chance of zooming out. Indeed, at the moment, we see the risk that we are debating on a relatively narrow point while it may be that we could essentially agree on several key aspects. Or at least, we believe we can work to find a common ground on these while we continue in parallel our discussion.

1. We believe that Theorem 2 is the main _stability proof_ contribution of the paper. To the best of our knowledge, we are the first in this community/problem setting to provide explicit input-to-state stability guarantees, including convergence rates into the region of attraction (Theorem 2). We believe that the proof of Theorem 2 is far from trivial.

Does the Reviewer disagree on this point?

2. In this sense, we agree with the Reviewer that the proof of Theorem 1 is comparably simpler (although not trivial!) and less impactful than the one of Theorem 2.

Would renaming "_Theorem 1_" as a "_Lemma_" or even as a "_Proposition_" better reflect the opinion of the Reviewer on the importance of this contribution?

3. Finally, we realize now that we did not make a great job, with the manuscript and with our previous answers, in conveying the message that: the relatively simplicity of the proof of Theorem 1 is no accident.
The proof of Theorem 1 could be derived via _standard_ arguments from the control of mechanical systems, _because_ we designed the CON network in a particular way. So the Theorem 1 statement with its relative simplicity implicitly stresses this point.

If the Reviewer agrees on this point, we are happy to include in the revision any remarks or changes that they see fit to better bring this point home.

Specifically, the proof in its current form is possible because (a) a coordinate transformation exists that allows us to identify a Lyapunov candidate, and (b) the system has a potential energy. We see these two design choices a contribution by themselves as they allow the proof to be simpler as it could have been otherwise.

We want to stress that already small modifications to the network formulation and the underlying assumptions in Theorem 1 would have made the proof much more difficult (or even impossible).
