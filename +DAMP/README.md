# D-AMP
The D-AMP implementation within the framework utilizes the [D-AMP_toolbox source code](https://github.com/ricedsp/D-AMP_Toolbox) from Rice University.
Note that the 'sensing_guassian_random_columnwise' sensing method also comes from the same source code.
The 'sensing_guassian_random_columnwise' sensing method works significantly better for D-AMP than the 'sensing_guassian_random' sensing method.
Unused files such as demos and the util folder have been removed from the source code.

The list of parameters for D-AMP include the following:
* **denoiser** - The denoiser to use for D-AMP. Currently includes the following: 'NLM', 'Gauss', 'Bilateral', 'BLS-GSM', 'BM3D', 'fast-BM3D', and 'CBM3D'. Note that CBM3D is applied to colored images, which are not yet supported.
* **iters** - The total number of iterations of D-AMP to run.

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
