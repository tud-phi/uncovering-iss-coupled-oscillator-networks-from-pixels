from pathlib import Path
import sympy as sp


def check_integrability_assumption(A: sp.Matrix) -> bool:
    """
    Check the integrability assumption for bringing the system into collocated form.
    Equation (4) of Pustina, Pietro, et al. "On the Collocated Form with Input Decoupling of Lagrangian Systems."
    arXiv preprint arXiv:2306.07258 (2023).
    Args:
        A: constant actuation matrix
    Returns:

    """
    # define the state variables
    q_ls = sp.symbols(f"q_ls1:{A.shape[0] + 1}")

    for i in range(A.shape[1]):
        # iterate over columns
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                lhs = A[j, i].diff(q_ls[k])
                rhs = A[k, i].diff(q_ls[j])
                if lhs != rhs:
                    print("Not integrable")
                    print("i = ", i)
                    print("j = ", j)
                    print("k = ", k)
                    print("lhs = ", lhs)
                    print("rhs = ", rhs)
                    return False

    print(
        "The integrability assumption is satisfied and we can bring the sytem into collocated form."
    )
    return True


def symbolically_derive_collocated_form(A: sp.Matrix):
    # define symbolic functions
    t = sp.Symbol("t", real=True, nonnegative=True)
    q_fn_ls = [sp.Function(f"xi_fn{i + 1}", real=True)(t) for i in range(A.shape[0])]
    q_d_fn_ls = [q_fn_ls[i].diff(t) for i in range(A.shape[0])]
    q_fn = sp.Matrix(q_fn_ls)
    q_d_fn = sp.Matrix(q_d_fn_ls)

    # derive collocated form
    dy_fn = A.T @ q_d_fn
    print("dy_fn =\n", dy_fn)

    # replace the time variable for integration
    tau = sp.Symbol("tau", real=True, nonnegative=True)
    dy_fn = dy_fn.subs(t, tau)
    # perform integration
    y_fn = sp.integrate(dy_fn, tau)
    print("y_fn =\n", y_fn)

    """
    # replace the symbolic functions with the variables
    y = y_fn.copy()
    for i in range(len(xi_fn)):
        y = y.subs(xi_fn[i], sym_exps["state_syms"]["xi"][i])
        y = y.subs(xi_d_fn[i], sym_exps["state_syms"]["xi_d"][i])
    y = sp.simplify(y)
    print("y =\n", y)
    """


if __name__ == "__main__":
    A = sp.Matrix([[1, 2], [3, 4]])

    assert (
        check_integrability_assumption(A) is True
    ), "Integrability assumption not satisfied"

    symbolically_derive_collocated_form(A)
