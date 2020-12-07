# TVAL3
The TVAL3 implementation within the framework utilizes the TVAL3 algorithm source code from [Rice University](https://www.caam.rice.edu/~optimization/L1/TVAL3/).
Unused files such as demos have been removed from the source code.

The list of parameters for TVAL3 include the following:
* **TVL2** - Whether to use the TVL2 version of the algorithm, where TVL2 utilizes a penalty parameter ||Au-b||_2^2 instead of an equality constraint Au = b
  * Default: false
* **TVnorm** - The L-norm to use. Either 1 (anisotropic) or 2 (isotropic).
  * Default: 2
* **nonneg** - Whether to use non-negative models, such that by the termination of the algorithm u >= 0.
  * Default: true
* **mu** - The primary penalty parameter.
  * Default: 2^8
* **beta** - The secondary penalty parameter.
  * Default: 2^5
* **tol** - The outer stopping tolerance.
  * Default: 1e-4
* **tol_inn** - The inner stopping tolerance.
  * Default: 1e-5
* **maxit** - The maximum total iterations.
  * Default: 300
* **maxcnt** - The maximum outer iterations.
  * Default: 10
* **isreal** - If the signal is composed solely of real values.
  * Default: true
* **disp** - Whether information should be printed each iteration.
  * Default: false
* **init** - The initial guess for the algorithm.
  * Default: At(b) where b = A(x)
* **scale_A** - Whether A should be scaled.
  * Default: true
* **scale_b** - Whether b should be scaled.
  * Default: true
* **consist_mu** - Whether mu should be scaled along with A and b.
  * Default: false
* **mu0** - The initial value of mu. Defaults to mu. Set much lower than mu to trigger the 'continuation scheme,' which increases the number of iterations and could increase convergence.
  * Default: mu
* **beta0** - The initial value of beta. Defaults to beta. Set much lower than beta to trigger the 'continuation scheme,' which increases the number of iterations and could increase convergence.
  * Default: beta
* **rate_ctn** - The continuation rate of the penalty parameters.
  * Default: 2
* **c** - The nonmonotone line search tolerance modifier.
  * Default: 1e-5
* **gamma** - The nonmonotone line search alpha continuation parameter.
  * Default: 0.6
* **gam** - Controls the degree of nonmonotonicity. 0 = monotone line search, 1 = nonmonotone line search.
  * Default: 0.9995
* **rate_gam** - The shrinking rate of gam.
  * Default: 0.9
* **normalization** - Whether the image should be normalized after reconstruction. Normalization consists of subtracting the minimum value such that the minimum value is zero. May improve reconstruction accuracy.
  * Default: false

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
  * Default: none
* **colored_reconstruction_mode** - The manner through which to reconstruct multi-channel images. Can be either 'channelwise' or 'vectorized'.
  * Default: 'channelwise'