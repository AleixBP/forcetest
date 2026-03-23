from dolfin import *
import math

def a_f(Lx, Ly, degree):
    class F(UserExpression):
        def __init__(self, Lx, Ly, **kwargs):
            self.center = [Lx/2., Ly/2.]
            self.lambda_=1.9762
            self.mu=1.
            self.lmm=self.lambda_+2.*self.mu
            self.lm=self.lambda_+self.mu
            super().__init__(**kwargs)

        def value_shape(self):
            return (2,)

        def eval(self, value, x):
            value[0]=50.*x[0]*(x[0]-1.)*(x[0]-0.5)*25.*x[1]*(x[1]-1.)*(x[1]-0.5)
            value[1]=50.*x[1]*(x[1]-1.)*(x[1]-0.5)*25.*x[0]*(x[0]-1.)*(x[0]-0.5)

    f=F(Lx=Lx, Ly=Ly, degree=degree)
    return f


def a_f_gaussian_dipole(Lx, Ly, degree, A=500.0, width=50.0):
    """
    Simulates a biological cell pulling inward from two focal adhesion sites.
    Creates two Gaussian bumps of force pointing towards the center.
    """
    class Dipole(UserExpression):
        def __init__(self, Lx, Ly, A, width, **kwargs):
            self.cx1 = Lx * 0.3
            self.cy1 = Ly * 0.5
            self.cx2 = Lx * 0.7
            self.cy2 = Ly * 0.5
            self.Ly = Ly
            self.A = A
            self.width = width
            super().__init__(**kwargs)

        def value_shape(self):
            return (2,)

        def eval(self, value, x):
            # Distance squared to left adhesion
            r1_sq = (x[0] - self.cx1)**2 + (x[1] - self.cy1)**2
            # Distance squared to right adhesion
            r2_sq = (x[0] - self.cx2)**2 + (x[1] - self.cy2)**2
            
            # Left bump pulls right, Right bump pulls left
            bump1_x = self.A * math.exp(-self.width * r1_sq)
            bump2_x = -self.A * math.exp(-self.width * r2_sq)
            
            # Pull slightly towards the center Y axis
            bump1_y = self.A * (self.Ly/2.0 - x[1]) * math.exp(-self.width * r1_sq)
            bump2_y = self.A * (self.Ly/2.0 - x[1]) * math.exp(-self.width * r2_sq)

            value[0] = bump1_x + bump2_x
            value[1] = bump1_y + bump2_y

    f = Dipole(Lx=Lx, Ly=Ly, A=A, width=width, degree=degree)
    return f


def a_f_gaussian_dipole_cplus(Lx, Ly, degree, A=500.0, width=50.0):
    """
    Simulates a biological cell pulling inward from two focal adhesion sites.
    Creates two Gaussian bumps of force pointing towards the center.
    """
    # C++ strings are completely immune to FEniCS Python __init__ RecursionErrors.
    cpp_code_x = "A * exp(-width * (pow(x[0] - cx1, 2) + pow(x[1] - cy1, 2))) - A * exp(-width * (pow(x[0] - cx2, 2) + pow(x[1] - cy2, 2)))"
    cpp_code_y = "A * (Ly/2.0 - x[1]) * exp(-width * (pow(x[0] - cx1, 2) + pow(x[1] - cy1, 2))) + A * (Ly/2.0 - x[1]) * exp(-width * (pow(x[0] - cx2, 2) + pow(x[1] - cy2, 2)))"

    f = Expression((cpp_code_x, cpp_code_y),
                   degree=degree,
                   A=A,
                   width=width,
                   cx1=Lx * 0.3, cy1=Ly * 0.5,
                   cx2=Lx * 0.7, cy2=Ly * 0.5,
                   Ly=Ly)
    return f

def a_f_gaussian_tripole(Lx, Ly, degree, A=500.0, width=50.0):
    """
    Simulates a biological cell pulling inward from three focal adhesion sites.
    Creates a dipole plus a third smaller adhesion site at the top pointing down.
    A: Amplitude of the base force
    width: Controls the spread of the Gaussian (higher = narrower bump)
    """
    class Tripole(UserExpression):
        def __init__(self, Lx, Ly, A, width, **kwargs):
            # Center of the left adhesion (x=0.3, y=0.4) - slightly lower
            self.cx1 = Lx * 0.3
            self.cy1 = Ly * 0.4
            # Center of the right adhesion (x=0.7, y=0.4) - slightly lower
            self.cx2 = Lx * 0.7
            self.cy2 = Ly * 0.4
            
            # Third pole at the top
            self.cx3 = Lx * 0.5
            self.cy3 = Ly * 0.8
            
            self.Ly = Ly
            self.A = A
            self.width = width
            super().__init__(**kwargs)

        def value_shape(self):
            return (2,)

        def eval(self, value, x):
            # Distance squared to adhesions
            r1_sq = (x[0] - self.cx1)**2 + (x[1] - self.cy1)**2
            r2_sq = (x[0] - self.cx2)**2 + (x[1] - self.cy2)**2
            r3_sq = (x[0] - self.cx3)**2 + (x[1] - self.cy3)**2
            
            # Base dipole: pulling inwards
            bump1_x = self.A * math.exp(-self.width * r1_sq)
            bump2_x = -self.A * math.exp(-self.width * r2_sq)
            
            bump1_y = self.A * (self.Ly/2.0 - x[1]) * math.exp(-self.width * r1_sq)
            bump2_y = self.A * (self.Ly/2.0 - x[1]) * math.exp(-self.width * r2_sq)
            
            # Top bump pulls straight downwards. Half amplitude, double tightness/area
            A3 = self.A * 0.5
            width3 = self.width * 2.0
            bump3_x = 0.0
            bump3_y = -A3 * math.exp(-width3 * r3_sq)

            value[0] = bump1_x + bump2_x + bump3_x
            value[1] = bump1_y + bump2_y + bump3_y

    # Pass the local variables securely into the wrapper to prevent variable scope closure failures in UserExpression
    f = Tripole(Lx=Lx, Ly=Ly, A=A, width=width, degree=degree)
    return f