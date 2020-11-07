# NLR-CS
The NLR-CS implementation uses the source code from the Xidian University website.
Please beware that random guassian sensing methods will **NOT** work with this method, due to its use of a DCT recovery method for the initial guess.
Additionally, note that in the original source code for this method, the measurement size is approximately double the sensing rate (i.e. a sensing rate of 0.1 on a 256x256 image would yield a measurement size of ~13107 instead of ~6554).
However, for this framework's implementation, we instead define the measurement size as the sensing rate multiplied by the signal size (i.e. a sensing rate of 0.1 on a 256x256 image would yield a measurement size of ~6554).
Therefore, in this framework, the size of the measurement vector should be approximately the value *m* defined by the user, not double. This may lead to discrepancies in performance between the source code and the framework.
Lastly, the sensing method *sensing_masked_uhp_fourier* generates a random mask, so the size of the measurements may not match the user-defined value.

The list of parameters for NLR-CS include the following:
* **win** - The size of each patch.
* **nblk** - The number of similar patches in each grouping.
* **step** - The steps between patches.
* **K0** - Iterations of NLR-CS after which to apply adaptive weights to SVT.
* **K** - Iterations of NLR-CS to run.
* **t0** - Threshold for DCT-thresholding.
* **nSig** - Threshold modifier for DCT-thresholding.
* **c0** - Threshold for non-weighted SVT.
* **c1** - Threshold for weighted SVT.

Other non-method specific parameters include the following:
* **slice_size** - The size of each slice of the image to reconstruct. Scalar or 2-element vector ordered [width, height].
