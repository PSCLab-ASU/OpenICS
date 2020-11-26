# L1-Magic Toolbox
The L1-Magic toolbox implementation within the framework utilizes both the L1 and TV (Total Variation) methods from the [original code](https://statweb.stanford.edu/~candes/software/l1magic/).
Note that the L1 methods are generally ill-suited for image reconstruction, so the TV methods will likely perform better. However, TV methods only work with square images.
Also, the sensing method *sensing_linemasked_uhp_fourier* generates a radial line mask, so *m* in this context should not be the measurement size of the vector, but the number of radial lines to draw. The actual size of the measurements is somewhat unpredictable.
Unused files such as the examples, the notes folder, and the data folder have been removed from the source code.

# L1 Methods
The L1 methods have four different implementations which it can run, which are specified under the 'constraints' parameter. These include:
* Equality constraints, or **'eq'**
* Quadratic constraints, or **'qc'**
* Dantzig Selector, or **'dantzig'**
* Minimum error approximation, or **'decode'**

Additional parameters include the following:
* **pdtol** - The tolerance of the primal-dual algorithm. 'qc' utilizes the log-barrier algorithm, so it does not apply when using quadratic constraints.
* **pdmaxiter** - The maximum number of iterations of the primal-dual algorithm. Similarly, it does not apply to the 'qc' constraint.
* **cgtol** - The tolerance of the Conjugated Gradients algorithm, which is used to solve the system of linear equations.
* **cgmaxiter** - The maximum number of iterations of the Conjugated Gradients algorithm.
* **epsilon** - The allowed error for the initial point x0 of the 'qc' and 'dantzig' constraints. For 'dantzig' it is also used as the correlation constraints. Must be a scalar for 'qc', but can be a scalar or an Nx1 vector for 'dantzig', where N is the original size of the signal.
* **lbtol** - The tolerance of the log-barrier algorithm. This is only applied to the 'qc' constraint.
* **mu** - The factor by which to increase the barrier constant per iteration. This is only applied on the 'qc' constraint.
* **normalization** - Whether the image should be normalized prior to sensing and reconstruction. Normalization consists of dividing by the L2-norm and subtracting the mean, and then inverting this transformation after reconstruction. This allows images to become sparser and may improve reconstruction accuracy.

# TV Methods
The TV methods have three different implementations which it can run, which are specified under the 'constraints' parameter. These include:
* Equality constraints, or **'eq'**
* Quadratic constraints, or **'qc'**
* Dantzig Selector, or **'dantzig'**

Note that the TV methods will not work with non-square images due to the way the algorithm calculates total variation.

Additional parameters include the following:
* **lbtol** - The tolerance of the log-barrier algorithm.
* **mu** - The factor by which to increase the barrier constant per iteration.
* **lintol** - The tolerance of the linear equation solving algorithm.
* **linmaxiter** - The maximum number of iterations of the linear equation solving algorithm. For 'eq', it is the Symmetric LQ method, while 'qc' and 'dantzig' utilize the Conjugated Gradients method.
* **epsilon** - The allowed error for the initial point x0 of the 'qc' and 'dantzig' constraints. Must be a scalar.
* **normalization** - Whether the image should be normalized prior to sensing and reconstruction. Normalization consists of dividing by the L2-norm and subtracting the mean, and then inverting this transformation after reconstruction. This allows images to become sparser and may improve reconstruction accuracy.

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
