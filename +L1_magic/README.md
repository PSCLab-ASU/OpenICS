# L1-Magic Toolbox
The L1-Magic toolbox implementation within the framework utilizes both the L1 and TV (Total Variation) methods from the [original code](https://statweb.stanford.edu/~candes/software/l1magic/).
Note that the L1 methods are generally ill-suited for image reconstruction, so the TV methods will likely perform better. However, TV methods only work with square images.
Also, the sensing method *sensing_linemasked_uhp_fourier* generates a radial line mask, so *m* in this context should not be the measurement size of the vector, but the number of radial lines to draw. The actual size of the measurements is somewhat unpredictable.
Unused files such as the examples, the notes folder, and the data folder have been removed from the source code.

# L1 Methods
The L1 methods have four different implementations which it can run, which are specified under the 'constraint' parameter. These include:
* Equality constraints, or **'eq'**
* Quadratic constraints, or **'qc'**
* Dantzig Selector, or **'dantzig'**
* Minimum error approximation, or **'decode'**

The default constraint is 'eq'.

Additional parameters include the following:
* **pdtol** - The tolerance of the primal-dual algorithm. 'qc' utilizes the log-barrier algorithm, so it does not apply when using quadratic constraints.
  * Default: 1e-3
* **pdmaxiter** - The maximum number of iterations of the primal-dual algorithm. Similarly, it does not apply to the 'qc' constraint.
  * Default: 50
* **cgtol** - The tolerance of the Conjugated Gradients algorithm, which is used to solve the system of linear equations.
  * Default: 1e-8
* **cgmaxiter** - The maximum number of iterations of the Conjugated Gradients algorithm.
  * Default: 200
* **epsilon** - The allowed error for the initial point x0 of the 'qc' and 'dantzig' constraints. For 'dantzig' it is also used as the correlation constraints. Must be a scalar for 'qc', but can be a scalar or an Nx1 vector for 'dantzig', where N is the original size of the signal.
  * Default: 5e-3
* **lbtol** - The tolerance of the log-barrier algorithm. This is only applied to the 'qc' constraint.
  * Default: 1e-3
* **mu** - The factor by which to increase the barrier constant per iteration. This is only applied on the 'qc' constraint.
  * Default: 10
* **normalization** - Whether the image should be normalized prior to sensing and reconstruction. Normalization consists of dividing by the L2-norm and subtracting the mean, and then inverting this transformation after reconstruction. This allows images to become sparser and may improve reconstruction accuracy.
  * Default: false

# TV Methods
Note that the TV methods will **not** work with non-square images due to the way the algorithm calculates total variation.

The TV methods have three different implementations which it can run, which are specified under the 'constraint' parameter. These include:
* Equality constraints, or **'eq'**
* Quadratic constraints, or **'qc'**
* Dantzig Selector, or **'dantzig'**

The default constraint is 'eq'.

Additional parameters include the following:
* **lbtol** - The tolerance of the log-barrier algorithm.
  * Default: 1e-2
* **mu** - The factor by which to increase the barrier constant per iteration.
  * Default: 2
* **lintol** - The tolerance of the linear equation solving algorithm.
  * Default: 1e-8
* **linmaxiter** - The maximum number of iterations of the linear equation solving algorithm. For 'eq', it is the Symmetric LQ method, while 'qc' and 'dantzig' utilize the Conjugated Gradients method.
  * Default: 200
* **epsilon** - The allowed error for the initial point x0 of the 'qc' and 'dantzig' constraints. Must be a scalar.
  * Default: 5e-3
* **normalization** - Whether the image should be normalized prior to sensing and reconstruction. Normalization consists of dividing by the L2-norm and subtracting the mean, and then inverting this transformation after reconstruction. This allows images to become sparser and may improve reconstruction accuracy.
  * Default: false

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
  * Default: none
* **colored_reconstruction_mode** - The manner through which to reconstruct multi-channel images. Can be either 'channelwise' or 'vectorized'.
  * Default: 'channelwise'
* **workers** - The number of parallel instances of a method to run over a directory of images. Requires Parallel Computing Toolbox.
  * Default: 0