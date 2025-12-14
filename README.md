# Heat-Diffusion-PINNs
Physics-informed neural network solution of the heat equation in 1D, 2D, and 3D using Gaussian initial conditions and prescribed boundary conditions.

# Heat Diffusion using Physics-Informed Neural Networks (PINNs)

## Motivation (Physics-Oriented)

Partial differential equations (PDEs) such as the heat equation govern a wide range of physical processes, from thermal conduction to diffusion phenomena.  
Traditional numerical solvers require structured grids and can become computationally expensive in higher dimensions or complex domains.

Physics-Informed Neural Networks (PINNs) provide an alternative framework by embedding physical laws directly into the loss function, enabling mesh-free solutions and seamless extension to higher dimensions.

This project explores the use of PINNs to solve the **heat diffusion equation** in **1D, 2D, and 3D**.

---

## Governing Equation

The heat diffusion equation solved in this work is:

\[
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
\]

where:
- \( u(\mathbf{x}, t) \) is the temperature field  
- \( \alpha \) is the thermal diffusivity  
- \( \nabla^2 \) is the Laplacian operator  

---

## Methodology

### Physics-Informed Neural Network (PINN)

A fully connected neural network is trained to approximate the solution \( u(\mathbf{x}, t) \) by minimizing a composite loss function that enforces physical constraints.

#### Loss Function Components

The total loss is defined as:

\[
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}}
\]

where:

- **PDE residual loss**
  \[
  \mathcal{L}_{\text{PDE}} = \left\| \frac{\partial u}{\partial t} - \alpha \nabla^2 u \right\|^2
  \]

- **Initial condition loss**
  \[
  u(\mathbf{x}, 0) = \exp\left(-\frac{|\mathbf{x}-\mathbf{x}_0|^2}{\sigma^2}\right)
  \]

- **Boundary condition loss**
  - Dirichlet or Neumann boundary conditions depending on the experiment

Automatic differentiation is used to compute spatial and temporal derivatives.

---

## Experimental Setup

- **Dimensions solved:**  
  - 1D heat diffusion  
  - 2D heat diffusion  
  - 3D heat diffusion  

- **Sampling strategy:**  
  - Random collocation points sampled uniformly in space–time domain  

- **Initial condition:**  
  - Gaussian temperature bump  

- **Boundary conditions:**  
  - Fixed (Dirichlet) or zero-gradient (Neumann), depending on case  

---

## Results

### Qualitative Behavior

The PINN successfully reproduces the expected physical behavior:

- Smooth diffusion of the Gaussian temperature profile over time  
- Progressive flattening of temperature gradients  
- Dimensional consistency across 1D, 2D, and 3D cases  

The learned solution remains stable and physically meaningful without explicit discretization or grid-based solvers.

---

### Dimensional Scaling

- **1D:** Accurate temporal diffusion with symmetric profile spreading  
- **2D:** Radially symmetric diffusion from the Gaussian peak  
- **3D:** Correct volumetric spreading and decay of peak temperature  

The same neural architecture and training framework generalize naturally across dimensions.

---

### Training Dynamics

- The PDE residual loss decreases steadily during training  
- Boundary and initial condition constraints are satisfied to high accuracy  
- No numerical instability or divergence observed  

This indicates that the network successfully balances data-free physics constraints with optimization stability.

---

## Key Observations

- PINNs can solve diffusion-type PDEs without labeled data  
- Random collocation sampling avoids mesh generation  
- Higher-dimensional problems are handled without reformulating numerical schemes  

---

## Limitations

- Training time increases with dimensionality  
- Convergence can be sensitive to loss weighting  
- Sharp gradients may require adaptive sampling strategies  

---

## Future Work

- Extension to variable diffusivity \( \alpha(\mathbf{x}) \)  
- Comparison with finite-difference / finite-element solvers  
- Adaptive collocation point refinement  
- Extension to nonlinear PDEs  

---

## Author

**Aryan Bandyopadhyay**  
Integrated MSc Physics  
School of Physical Sciences  
NISER Bhubaneswar
