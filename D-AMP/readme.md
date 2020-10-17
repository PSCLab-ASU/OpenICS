# D-AMP
The D-AMP implementation within the framework utilizes the D-AMP_toolbox source code from Rice University.
Note that the 'sensing_guassian_random_columnwise' and 'sensing_guassian_random_rowwise' sensing methods also come from the same source code.
These two sensing methods work significantly better for D-AMP than the 'sensing_guassian_random' or 'sensing_uniform_random' sensing methods.

The list of parameters for D-AMP include the following:
* **denoiser** - The denoiser to use for D-AMP. Currently includes the following: 'NLM', 'Gauss', 'Bilateral', 'BLS-GSM', 'BM3D', 'fast-BM3D', and 'CBM3D'. Note that CBM3D is applied to colored images, which are not yet supported.
* **iters** - The total number of iterations of D-AMP to run.
