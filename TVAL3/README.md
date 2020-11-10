# TVAL3
The TVAL3 implementation within the framework utilizes the TVAL3 algorithm source code from [Rice University](https://www.caam.rice.edu/~optimization/L1/TVAL3/).
Unused files such as demos have been removed from the source code.

The list of parameters for TVAL3 include the following:
* **TVL2** - Whether to use the TVL2 version of the algorithm, where TVL2 utilizes a penalty parameter ||Au-b||_2^2 instead of an equality constraint Au = b
* **TVnorm** - The L-norm to use. Either 1 (anisotropic) or 2 (isotropic).
* **nonneg** - Whether to use non-negative models, such that by the termination of the algorithm u >= 0.
* **mu** - The primary penalty parameter.
* **beta** - The secondary penalty parameter.
* **tol** - The outer stopping tolerance.
* **tol_inn** - The inner stopping tolerance.
* **maxit** - The maximum total iterations.
* **maxcnt** - The maximum outer iterations.
* **isreal** - If the signal is composed solely of real values.
* **disp** - Whether information should be printed each iteration.
* **init** - The initial guess for the algorithm. Defaults to At(b).
* **scale_A** - Whether A should be scaled.
* **scale_b** - Whether b should be scaled.
* **consist_mu** - Whether mu should be scaled along with A and b.
* **mu0** - The initial value of mu. Defaults to mu. Set much lower than mu to trigger the 'continuation scheme,' which increases the number of iterations and should increase convergence.
* **beta0** - The initial value of beta. Defaults to beta. Set much lower than beta to trigger the 'continuation scheme,' which increases the number of iterations and should increase convergence.
* **rate_ctn** - The continuation rate of the penalty parameters.
* **c** - The nonmonotone line search tolerance modifier.
* **gamma** - The nonmonotone line search alpha continuation parameter.
* **gam** - Controls the degree of nonmonotonicity. 0 = monotone line search, 1 = nonmonotone line search.
* **rate_gam** - The shrinking rate of gam.
* **normalization** - Whether the image should be normalized after reconstruction. Normalization consists of subtracting the minimum value such that the minimum value is zero. May improve reconstruction accuracy.

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
