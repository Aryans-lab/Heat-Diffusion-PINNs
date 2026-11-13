"""idealpinn: what does an inverse physics-informed neural network estimate?

Two halves that are deliberately independent of each other:

* :mod:`idealpinn.exact` - closed-form (NumPy) theory for the heat equation.
  It computes, without any neural network, the estimator that the PINN
  objective defines in the infinite-capacity / perfect-optimisation limit.
* :mod:`idealpinn.nn` - an actual PINN stack (PyTorch) that is trained on the
  same data sets so that the two can be compared number by number.
"""

__version__ = "1.0.0"
