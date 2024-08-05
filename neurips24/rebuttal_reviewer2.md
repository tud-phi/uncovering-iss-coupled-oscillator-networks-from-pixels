# Response to Reviewer W9L3 (R2)

We thank the Reviewer for the careful reading and the encouraging comments. 
In the following, we will respond to the questions raised by the reviewer, which also relate to the weaknesses mentioned by the reviewer.

## Application of CON to non-soft robots

> <cite>R2</cite>: Have you tried the method on other (non-soft) robots? Do you have any intuition on how your method would perform on environments with many contact forces?

We thank the Reviewer for inquiring about results on non-soft robots. We provide additional results on various mechanical/robotic systems in the global rebuttal.

Next, we reply the Reviewer's question about how the method would perform on environments with many contact forces.
First, we want to stress that we have not conducted any experiments with contact-rich systems yet and reserve this interesting challenge for future work. 
We hypothesize that the highly discontinuous dynamics of contact-rich systems could present a challenge for the method in its present form.
Still, we envision that the method could be augmented to handle such systems. For example, we could add stick-slip friction or similar mechanisms to the CON dynamics to increase their expressiveness while maintaining the physical structure.
Furthermore, contact-rich systems often have multiple equilibria, which would be a violation of the stability conditions introduced in this work. As touched on in Section 6 of the paper, these strong global stability guarantees would probably need to be relaxed.

## Application of CON to non-physical systems

> <cite>R2</cite>: Have you considered non-physical systems?

We thank the Reviewer for their interest in this topic. While we do think that the CON / CFA-CON models could be potentially useful in other applications where strong stability guarantees are required, we strive to focus in this paper on learning the dynamics of physical/mechanical systems as we can here leverage (a) the shared stability characteristics between the original and the latent space systems, and (b) exploit the mechanical structure of the CON model for control.
To make this focus clear, we will, in the final paper, modify the sentence starting on line 69 to (with changes marked in **bold**):

_We resolve all the above-mentioned challenges by proposing Coupled Oscillator Networks (CONs), a new formulation of a coupled oscillator network that is inherently Input-to-State Stability (ISS) stable, **for learning the dynamics of physical systems,** and subsequently exploiting its structure for model-based control in latent space._

This work was specifically focused on how to impose structural biases that enforce second order Lagrangian structure and stability properties. However, we believe that the proposed model can be used more broadly, beyond mechanical systems, whenever strong stability guarantees are needed. This will be the focus of future research. 