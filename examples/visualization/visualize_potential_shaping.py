import jax
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

# activate plotting with latex typesetting
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

def potential_energy_fn(q: Array, q_eq: Optional[Array] = None):
    """
    Quadratic potential energy function for a n-dimensional system.
    Arguments:
        q: configuration as a n-dimensional vector
        q_eq: equilibrium configuration as a n-dimensional vector
    Returns:
        potential energy of the system
    """
    if q_eq is None:
        q_eq = jnp.zeros_like(q)

    return jnp.sum(jnp.square(q - q_eq))

if __name__ == "__main__":
    q1_grid, q2_grid = jnp.meshgrid(jnp.linspace(-1, 1, 100), jnp.linspace(-1, 1, 100))
    q1_pts = q1_grid.flatten()
    q2_pts = q2_grid.flatten()
    q_pts = jnp.stack([q1_pts, q2_pts], axis=-1)


    # Create the figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    """
    # Plot the surface
    q_eq = jnp.array([-0.5, -0.5])
    # q_eq = jnp.zeros((2,))
    U_pts = jax.vmap(potential_energy_fn)(q_pts)
    U_grid = U_pts.reshape(q1_grid.shape)
    surface = ax.plot_surface(q1_grid, q2_grid, U_grid, cmap='viridis')
    """

    # q_eqs = jnp.array([0.5, 0.5])
    q_eqs = [jnp.array([0.5, 0.5]), jnp.array([-0.5, -0.5])]
    for q_eq in q_eqs:
        U_pts = jax.vmap(lambda q: potential_energy_fn(q, q_eq))(q_pts)
        U_grid = U_pts.reshape(q1_grid.shape)
        ax.plot_surface(q1_grid, q2_grid, U_grid, cmap='viridis', alpha=0.5)

    # Label axes
    # ax.set_xlabel(r"$q_1$")
    # ax.set_ylabel(r"$q_2$")
    # ax.set_zlabel(r"Potential Energy $\mathcal{U}$")

    # Add a colorbar to indicate potential energy magnitude
    # fig.colorbar(surface, shrink=0.5, aspect=5, label=r"$\mathcal{U}(q_1, q_2)$")

    # switch off the grid
    ax.grid(False)

    # remove ticks from axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # switch off the axes
    ax.axis('off')

    plt.tight_layout()
    plt.show()
