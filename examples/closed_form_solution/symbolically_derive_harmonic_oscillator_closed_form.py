from sympy import symbols, Eq, Function

from sympy.solvers.ode.systems import dsolve_system

x_fn, v_fn = symbols("f1 f2", cls=Function)

t = symbols("t")
x0, v0 = symbols("x0 v0")
m, gamma, epsilon = symbols("m gamma epsilon", real=True, positive=True)
# m, gamma, epsilon = 1.0, 0.1, 0.05

# ODE
eqs = [
    Eq(x_fn(t).diff(t), v_fn(t)),
    Eq(v_fn(t).diff(t), m ** (-1) * (-gamma * x_fn(t) - epsilon * v_fn(t)))
]

# initial conditions
ics = {x_fn(0): x0, v_fn(0): v0}

sol = dsolve_system(eqs, ics=ics)
print(sol)
