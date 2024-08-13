# Rebuttal 2 - Reviewer 2

> I agree with the authors' view that their method might underperform for systems with highly discontinuous dynamics. Since this paper is framed as a method for prediction and control of systems, with soft robots being only one application area (rather than a method for the control of soft robots), I would appreciate a lucid analysis of classes of systems/application areas in which the method might perform strongly/weakly (including a clear reasoning why; and potentially a remedy left to future work). Please consider including this, both from a systems theory and an application standpoint into the paper.

We appreciate the Reviewer's feedback and suggestion. We fully agree with the Reviewer that specific categories or examples of systems for which the proposed method might be suitable or unsuitable should be discussed in the paper. Because of time constraints, we will prepare a full version of this paragraph for the final version of the paper. However, we would like to provide below already some preliminary thoughts and examples.

## Systems for which we would expect the proposed method to work

- **Mechanical systems with continuous dynamics, dissipation and a single, attractive equilibrium point:** The proposed method is particularly well-suited for mechanical systems with continuous dynamics, dissipation, and a single, attractive equilibrium point. In this case, the real system and the latent dynamics share both the energetic structure and stability guarantees. Examples of such systems include many soft robots, deformable objects with dominant elastic behavior, other mechanical structures with elasticity, etc.

## Systems for which we could envision the proposed method to work under modifications

- **Mechanical systems without dissipation:** The proposed method would currently not work well for mechanical systems without dissipation, as (a) the original system will likely not have a gobally asymptotically stable equilibrium point, and more importantly, (b) we currently force the damping learned in latent space to be positive definite. A possible remedy could be to relax the positive definiteness of the damping matrix in the latent space, allowing for zero damping. This would allow the method to work for systems without dissipation, such as conservative systems. Examples of such systems include a mass-spring system without damping, the n-body problem, etc.

- **(Mechanical) systems with discontinuous dynamics:** The proposed method might underperform for systems with highly discontinuous dynamics, such as systems with impacts, friction, or other discontinuities. In these cases, the latent dynamics might not capture the real system's behavior accurately, and the control performance of feedforward + feedback will very likely be worse than pure feedback. A possible remedy could be to augment the latent dynamics with additional terms that capture the discontinuities such as contact and friction models (e.g., stick-slip friction).

- **(Mechanical) systems with multiple equilibrium points:** The original system having multiple equilibria conflicts with the stability assumptions underlying the proposed CON latent dynamics. In this case, as for example seen on the pendulum+friction and double pendulum + friction results, the method might work locally, but will not be able to capture the global behavior of the system. A possible remedy could be to relax the global stability assumptions of the CON network. For example, the latent dynamics could be learned in the original coordinates of CON while allowing $W$ also to be negative definite. This would allow the system to have multiple equilibria & attractors. Examples of such systems include a robotic arm under gravity, pendula under gravity, etc.

- **(Mechanical) systems with periodic behavior:** The proposed method will likely not work well for systems with periodic behavior, as the they do not have a single, attractive equilibrium point. A possible remedy could be to augment the latent dynamics with additional terms that capture the periodic behavior, such as substituting the harmonic oscillators with Van der Pol oscillators to establish a limit cycle or a supercritical Hopf bifurcation. Examples of such systems include a mass-spring system with a periodic external force, a pendulum with a periodic external force, some chemical reactions, etc.

## Systems for which we would not expect the proposed method to work

- **Nonholonomic systems:** The proposed method likely would not work well for nonholonomic systems, as both structure (e.g., physical constraints) and stability characteristics would not be shared between the real system and the latent dynamics. Examples of such systems include vehicles, a ball rolling on a surface, and many mobile robots.

- **Partially observable and non-markovian systems:** As the CON dynamics are evaluated based on the latent position and velocity encoded by the observation of the current time step and the observation-space velocity, we implicitly assume that the system is (a) fully observable, and (b) fullfills the Markov property. This assumption might not hold for partially observable systems, such as systems with hidden states or systems with delayed observations. Examples of such cases include settings where the system is partially occluded, or not sufficient (camera) perspectives are available. Furthermore, time-dependent material properties, such as viscoelasticity or hysteresis, that are present and significant in some soft robots and deformable objects, are not captured by the method in its current formulation.