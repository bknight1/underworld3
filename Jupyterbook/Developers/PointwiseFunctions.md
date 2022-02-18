# PETSc pointwise functions and UW3 functions

PETSc provides an abstraction for the numerical representation of a physical problem that generally allows an application programmer to focus on describing the physics of the problem. The numerical implementation can be 

See `{cite}`knepleyAchievingHighPerformance2013




PETSc provides a mechanism to automatically generate a finite element weak form from the point-wise (strong form) of the governing equations. This takes the form of a template equation and its Jacobians. [Expand]

In `uw3`, we provide a fully symbolic approach to constructing the strong form of some common systems of equations, and use automatical differentiation to construct the required Jacobian terms which otherwise are prone to human error.


## Example 1 - The Poisson Equation


## Example 2 - The Scalar Advection-diffusion Equation


## Example 3 - The Stokes Equation


(Potential example 4, Navier Stokes)



```python

```



