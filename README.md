# CS-Framework

## hierarchy

parameters:

* sensing:string
    * which sensing method to use: 
        1. Guassian: generate guassian sensing matrix N(0,1/m), 
        2. specified
* reconstruction:string
    * which reconstruction method to use: 
        * ldamp
        * ISTA-Net
        * ReconNet
        * LAPRAN
        * csgm
        * cs-gan
* stage:string
    * training
    * testing
* default: bool
    * using all the default setups to reproduce the results in original paper. Will override all the other parameters.
* dataset: string
    * which dataset to use
* input_channel:int
    * color channel
* input_width:int
    * the width of the input image
* input_height:int
    * the height of the input image
* m:int
    * number of measurements
* n:int
    * dimensionality of the original signal.
* specifics:dict
    * additional parameters specific to the used methods.

## Included models:

* Data-driven: 
    * ISTA-Net
    * LDAMP
    * ReconNet
    * LAPRAN
    * CSGM
    * CS-GAN.
* model based: 
    * L1(lasso)
    * TVAL-3, DAMP
    * NLR-CS.