import dolfin as dl
import numpy as np
from hippylib.algorithms.linalg import (
    get_diagonal, Solver2Operator, estimate_diagonal_inv2,
    MatMatMult, amg_method, Operator2Solver,
)
from hippylib.algorithms.lowRankOperator import LowRankOperator
from hippylib.algorithms.multivector import MultiVector
from hippylib.algorithms.randomizedEigensolver import doublePass, doublePassG
from hippylib.utils.random import parRandom


# ---------------------------------------------------------------------------
#  Helpers for prior trace estimation
# ---------------------------------------------------------------------------

class _RinvM:
    """Operator that models the action of R^{-1} M (used in trace estimation)."""

    def __init__(self, Rsolver, M):
        self.Rsolver = Rsolver
        self.M = M

    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)

    def mult(self, x, y):
        self.Rsolver.solve(y, self.M * x)

class prior_distro:
    """
    L^2 Tikhonov prior: R = gamma * M, where M is the FEM mass matrix.

    Provides sampling via the square-root mass matrix approach and
    pointwise variance / trace computation via hippylib utilities.
    """

    def __init__(self, Vh, R, Rsolver, gamma, mean=None,
                 rel_tol=1e-12, max_iter=1000):
        self.R = R
        self.Rsolver = Rsolver
        self.Vh = Vh
        self.gamma = dl.Constant(gamma)

        # FEM mass matrix and solver
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)
        varfM = dl.inner(trial, test) * dl.dx

        self.M = dl.assemble(varfM)
        self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False

        # Prior operator A = gamma * M and its solver
        self.A = dl.assemble(self.gamma * varfM)
        self.Asolver = dl.PETScKrylovSolver("cg", amg_method())
        self.Asolver.set_operator(self.A)
        self.Asolver.parameters["maximum_iterations"] = max_iter
        self.Asolver.parameters["relative_tolerance"] = rel_tol
        self.Asolver.parameters["error_on_nonconvergence"] = True
        self.Asolver.parameters["nonzero_initial_guess"] = False

        # Square-root mass matrix (for prior sampling)
        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2 * Vh._ufl_element.degree()
        metadata = {"quadrature_degree": qdegree}

        representation_old = dl.parameters["form_compiler"]["representation"]
        dl.parameters["form_compiler"]["representation"] = "quadrature"

        num_sub = Vh.num_sub_spaces()
        if num_sub <= 1:
            element = dl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(),
                                       qdegree, quad_scheme="default")
        else:
            element = dl.VectorElement("Quadrature", Vh.mesh().ufl_cell(),
                                       qdegree, dim=num_sub, quad_scheme="default")
        Qh = dl.FunctionSpace(Vh.mesh(), element)

        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(dl.inner(ph, qh) * dl.dx(metadata=metadata))

        if num_sub <= 1:
            one_constant = dl.Constant(1.)
        else:
            one_constant = dl.Constant(tuple([1.] * num_sub))
        ones = dl.interpolate(one_constant, Qh).vector()
        dMqh = Mqh * ones
        Mqh.zero()
        dMqh.set_local(ones.get_local() / np.sqrt(dMqh.get_local()))
        Mqh.set_diagonal(dMqh)

        MixedM = dl.assemble(dl.inner(ph, test) * dl.dx(metadata=metadata))
        self.sqrtM = MatMatMult(MixedM, Mqh)

        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        dl.parameters["form_compiler"]["representation"] = representation_old

        # Mean
        if mean is not None:
            self.mean = mean
        else:
            self.mean = dl.Vector(self.R.mpi_comm())
            self.init_vector(self.mean, 0)

    def sample(self, noise, s, add_mean=True):
        """Draw a sample from the prior given white noise."""
        rhs = self.sqrtM * noise
        self.Asolver.solve(s, rhs)
        if add_mean:
            s.axpy(1., self.mean)

    def init_vector(self, x, dim):
        """Initialize a vector compatible with the prior operator."""
        if dim == "noise":
            self.sqrtM.init_vector(x, 1)
        else:
            self.A.init_vector(x, dim)

    def pointwise_variance(self, method, k=1000000, r=200):
        """Compute/estimate the prior pointwise variance.

        method = "Exact" | "Estimator" | "Randomized".
        """
        pw_var = dl.Vector(self.R.mpi_comm())
        self.init_vector(pw_var, 0)
        if method == "Exact":
            get_diagonal(Solver2Operator(self.Rsolver,
                                        init_vector=self.init_vector), pw_var)
        elif method == "Estimator":
            estimate_diagonal_inv2(self.Rsolver, k, pw_var)
        elif method == "Randomized":
            Omega = MultiVector(pw_var, r)
            parRandom.normal(1., Omega)
            d, U = doublePass(Solver2Operator(self.Rsolver),
                              Omega, r, s=1, check=False)
            for i in np.arange(U.nvec()):
                pw_var.axpy(d[i], U[i] * U[i])
        else:
            raise NameError("Unknown method: %s" % method)
        return pw_var

    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100, r=200):
        """Compute/estimate tr(R^{-1} M)."""
        from hippylib.algorithms.traceEstimator import TraceEstimator

        op = _RinvM(self.Rsolver, self.M)

        if method == "Exact":
            marginal_variance = dl.Vector(self.R.mpi_comm())
            self.init_vector(marginal_variance, 0)
            get_diagonal(op, marginal_variance)
            return marginal_variance.sum()
        elif method == "Estimator":
            tr_estimator = TraceEstimator(op, False, tol)
            tr_exp, _ = tr_estimator(min_iter, max_iter)
            return tr_exp
        elif method == "Randomized":
            dummy = dl.Vector(self.R.mpi_comm())
            self.init_vector(dummy, 0)
            Omega = MultiVector(dummy, r)
            parRandom.normal(1., Omega)
            d, _ = doublePassG(Solver2Operator(self.Rsolver),
                               Solver2Operator(self.Msolver),
                               Operator2Solver(self.M),
                               Omega, r, s=1, check=False)
            return d.sum()
        else:
            raise NameError("Unknown method: %s" % method)
        

# ---------------------------------------------------------------------------
#  Low-rank posterior classes
# ---------------------------------------------------------------------------

class LowRankHessian:
    """Low-rank approximation of the posterior Hessian and its inverse.

    Given the dominant generalised eigenpairs (d, U) of H_misfit,
    the full Hessian is approximated as  R + R U D U^T R  and
    its inverse as  R^{-1} - U (I + D^{-1})^{-1} U^T.
    """

    def __init__(self, prior, d, U):
        self.prior = prior
        self.LowRankH = LowRankOperator(d, U)
        dsolve = d / (np.ones(d.shape, dtype=d.dtype) + d)
        self.LowRankHinv = LowRankOperator(dsolve, U)
        self.help = dl.Vector(U[0].mpi_comm())
        self.init_vector(self.help, 0)
        self.help1 = dl.Vector(U[0].mpi_comm())
        self.init_vector(self.help1, 0)

    def init_vector(self, x, dim):
        self.prior.init_vector(x, dim)

    def inner(self, x, y):
        Hx = dl.Vector(self.help.mpi_comm())
        self.init_vector(Hx, 0)
        self.mult(x, Hx)
        return Hx.inner(y)

    def mult(self, x, y):
        self.prior.R.mult(x, y)
        self.LowRankH.mult(y, self.help)
        self.prior.R.mult(self.help, self.help1)
        y.axpy(1, self.help1)

    def solve(self, sol, rhs):
        self.prior.Rsolver.solve(sol, rhs)
        self.LowRankHinv.mult(rhs, self.help)
        sol.axpy(-1, self.help)


class LowRankPosteriorSampler:
    """Sample from the low-rank posterior approximation.

    y = (I - U S U^T) x,   S = I - (I + D)^{-1/2},   x ~ N(0, R^{-1}).
    """

    def __init__(self, prior, d, U):
        self.prior = prior
        ones = np.ones(d.shape, dtype=d.dtype)
        self.d = ones - np.power(ones + d, -0.5)
        self.lrsqrt = LowRankOperator(self.d, U)
        self.help = dl.Vector(U[0].mpi_comm())
        self.init_vector(self.help, 0)

    def init_vector(self, x, dim):
        self.prior.init_vector(x, dim)

    def sample(self, noise, s):
        self.prior.R.mult(noise, self.help)
        self.lrsqrt.mult(self.help, s)
        s.axpy(-1., noise)
        s *= -1.


class GaussianLRPosterior:
    """Low-rank Gaussian approximation of the posterior.

    Wraps LowRankHessian and LowRankPosteriorSampler to provide
    Hessian apply/solve, sampling, trace, pointwise variance, and KL divergence.
    """

    def __init__(self, prior, d, U, mean=None):
        self.prior = prior
        self.d = d
        self.U = U
        self.Hlr = LowRankHessian(prior, d, U)
        self.sampler = LowRankPosteriorSampler(self.prior, self.d, self.U)
        self.mean = mean

    def cost(self, m):
        if self.mean is None:
            return 0.5 * self.Hlr.inner(m, m)
        dm = m - self.mean
        return 0.5 * self.Hlr.inner(dm, dm)

    def init_vector(self, x, dim):
        """Initialize a vector compatible with the posterior Hessian."""
        self.prior.init_vector(x, dim)

    def sample(self, *args, **kwargs):
        """Draw a posterior sample.

        Two calling conventions:
            sample(s_prior, s_post, add_mean=True)
            sample(noise, s_prior, s_post, add_mean=True)
        """
        add_mean = kwargs.get("add_mean", True)

        if len(args) == 2:
            self.sampler.sample(args[0], args[1])
            if add_mean and self.mean is not None:
                args[1].axpy(1., self.mean)
        elif len(args) == 3:
            self.prior.sample(args[0], args[1], add_mean=False)
            self.sampler.sample(args[1], args[2])
            if add_mean:
                args[1].axpy(1., self.prior.mean)
                if self.mean is not None:
                    args[2].axpy(1., self.mean)
        else:
            raise ValueError("Expected 2 or 3 positional arguments, got %d" % len(args))

    def trace(self, **kwargs):
        """Return (post_trace, prior_trace, correction_trace)."""
        pr_trace = self.prior.trace(**kwargs)
        corr_trace = self.Hlr.LowRankHinv.trace(self.prior.M)
        post_trace = pr_trace - corr_trace
        return post_trace, pr_trace, corr_trace

    def pointwise_variance(self, **kwargs):
        """Return (post_pw_var, prior_pw_var, correction_pw_var)."""
        pr_pw = self.prior.pointwise_variance(**kwargs)
        corr_pw = dl.Vector(self.prior.R.mpi_comm())
        self.init_vector(corr_pw, 0)
        self.Hlr.LowRankHinv.get_diagonal(corr_pw)
        post_pw = pr_pw - corr_pw
        return post_pw, pr_pw, corr_pw

    def klDistanceFromPrior(self, sub_comp=False):
        """KL(posterior || prior).  If sub_comp, also return components."""
        dplus1 = self.d + np.ones_like(self.d)
        c_logdet = 0.5 * np.sum(np.log(dplus1))
        c_trace = -0.5 * np.sum(self.d / dplus1)
        c_shift = self.prior.cost(self.mean)
        kld = c_logdet + c_trace + c_shift
        if sub_comp:
            return kld, c_logdet, c_trace, c_shift
        return kld