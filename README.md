
# Section 0: Table of Benchmarks
<table>
<thead>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>Compression Ratio</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>Reconstruction Speed</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="30">ISTANETPlus</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>47.99</td>
    <td>0.9999</td>
    <td>0.0107</td>
  </tr>
  <tr>
    <td>4</td>
    <td>44.18</td>
    <td>0.9987</td>
    <td>0.0181</td>
  </tr>
  <tr>
    <td>8</td>
    <td>34.96</td>
    <td>0.9903</td>
    <td>0.0132</td>
  </tr>
  <tr>
    <td>16</td>
    <td>27.32</td>
    <td>0.9536</td>
    <td>0.0132</td>
  </tr>
  <tr>
    <td>32</td>
    <td>19.29</td>
    <td>0.7411</td>
    <td>0.0121</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>37.43</td>
    <td>0.9798</td>
    <td>0.0839</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.14</td>
    <td>0.9297</td>
    <td>0.0699</td>
  </tr>
  <tr>
    <td>8</td>
    <td>27.07</td>
    <td>0.8499</td>
    <td>0.2076</td>
  </tr>
  <tr>
    <td>16</td>
    <td>23.77</td>
    <td>0.7406</td>
    <td>0.1361</td>
  </tr>
  <tr>
    <td>32</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>34.12</td>
    <td>0.9703</td>
    <td>0.0332</td>
  </tr>
  <tr>
    <td>4</td>
    <td>27.66</td>
    <td>0.8932</td>
    <td>0.0301</td>
  </tr>
  <tr>
    <td>8</td>
    <td>23.41</td>
    <td>0.7632</td>
    <td>0.0318</td>
  </tr>
  <tr>
    <td>16</td>
    <td>20.25</td>
    <td>0.5979</td>
    <td>0.0386</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.95</td>
    <td>0.435</td>
    <td>0.0379</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>33.63</td>
    <td>0.9679</td>
    <td>0.0072</td>
  </tr>
  <tr>
    <td>4</td>
    <td>27.46</td>
    <td>0.8886</td>
    <td>0.0147</td>
  </tr>
  <tr>
    <td>8</td>
    <td>23.15</td>
    <td>0.7501</td>
    <td>0.0133</td>
  </tr>
  <tr>
    <td>16</td>
    <td>20.25</td>
    <td>0.5911</td>
    <td>0.0123</td>
  </tr>
  <tr>
    <td>32</td>
    <td>18.13</td>
    <td>0.4406</td>
    <td>0.0116</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>37.28</td>
    <td>0.9393</td>
    <td>0.0631</td>
  </tr>
  <tr>
    <td>4</td>
    <td>33</td>
    <td>0.8686</td>
    <td>0.0512</td>
  </tr>
  <tr>
    <td>8</td>
    <td>29.88</td>
    <td>0.7823</td>
    <td>0.0453</td>
  </tr>
  <tr>
    <td>16</td>
    <td>24.73</td>
    <td>0.584</td>
    <td>0.0390</td>
  </tr>
  <tr>
    <td>32</td>
    <td>24.67</td>
    <td>0.5864</td>
    <td>0.0336</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>38.49</td>
    <td>0.95</td>
    <td>0.0173</td>
  </tr>
  <tr>
    <td>4</td>
    <td>34.06</td>
    <td>0.8874</td>
    <td>0.0130</td>
  </tr>
  <tr>
    <td>8</td>
    <td>30.69</td>
    <td>0.8007</td>
    <td>0.0121</td>
  </tr>
  <tr>
    <td>16</td>
    <td>27.66</td>
    <td>0.7035</td>
    <td>0.0135</td>
  </tr>
  <tr>
    <td>32</td>
    <td>25.16</td>
    <td>0.6074</td>
    <td>0.0153</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>


# Section 1: Setting up environment

### Datadriven using conda:
`conda create -n CS_methods_environement python=3.7.9` </br>
`conda activate CS_methods_environement`</br>
</br>
`conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch`</br>
`conda install tensorflow-gpu=2.1.0`</br>
`conda install opencv==3.4.2`</br>
`conda install scikit-image==0.17.2`</br>
 
### Matlab:
IN MATLAB:<br>
`cd \<path to CS_Framework directory\>`<br>
`run set_up.m`<br>





# Section 2: Parameters

sensing: Method of sensing</br>
reconstruction: Method of reconstruction</br>
stage: Training or testing</br>
default: [True] use original paper's parameters [False] manually set parameters</br>
dataset: Dataset to be used</br>
input_channel: # of channels training/testing images have</br>
input_width: Width of training/testing images</br>
input_height: Height of training/testing images</br>
m: # of measurements/outputs of sensing matrix</br>
n: # of inputs to sensing matrix</br>
specifics: Specific parameter settings of chosen reconstruction method</br>

### Matlab:
To quickly modify parameters, make changes to demo.m. Refer to the README.md
located in the directory of each Matlab reconstruction method for more details
on method-specific parameters.




# Section 3: List of Methods
### Model-based:
TVAL3</br>
(Reference: https://www.caam.rice.edu/~zhang/reports/tr1213.pdf)</br>
(Repository: https://www.caam.rice.edu/~optimization/L1/TVAL3/)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

NLR-CS</br>
(Reference: https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/NLR-CS-TIP.pdf)</br>
(Repository: http://see.xidian.edu.cn/faculty/wsdong/Code_release/NLR_codes.rar)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

D-AMP</br>
(Reference: https://arxiv.org/pdf/1406.4175.pdf)</br>
(Repository: https://github.com/ricedsp/D-AMP_Toolbox)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

L1</br>
(Reference: https://statweb.stanford.edu/~candes/software/l1magic/)</br>
(Repository: https://statweb.stanford.edu/~candes/software/l1magic/)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance


### Datadriven:
ISTANET </br>
(Reference: https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper.pdf)</br>
(Repository: https://github.com/jianzhangcs/ISTA-Net)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* compare to one in original paper
* merged ISTANet and ISTANetPlus, parameter used to control which one to train/test

LDAMP</br>
(Reference: https://papers.nips.cc/paper/2017/file/8597a6cfa74defcbde3047c891d78f90-Paper.pdf)</br>
(Repository: https://github.com/ricedsp/D-AMP_Toolbox/tree/master/LDAMP_TensorFlow)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* compare to one in original paper
* only supports layer-by-layer and denoiser-by-denoiser training (not end-to-end)
* supports gaussian, complex-gaussian, and coded-diffraction sensing for Layer-by-Layer (not Fast-JL)

ReconNet</br>
(Reference: https://openaccess.thecvf.com/content_cvpr_2016/papers/Kulkarni_ReconNet_Non-Iterative_Reconstruction_CVPR_2016_paper.pdf)</br>
(Repository: https://github.com/KuldeepKulkarni/ReconNet)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

LAPRAN</br>
(Reference: https://openaccess.thecvf.com/content_ECCV_2018/papers/Kai_Xu_LAPCSRA_Deep_Laplacian_ECCV_2018_paper.pdf)</br>
(Repository: https://github.com/PSCLab-ASU/LAPRAN-PyTorch)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

CSGM</br>
(Reference: http://proceedings.mlr.press/v70/bora17a/bora17a.pdf)</br>
(Repository: https://github.com/AshishBora/csgm)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance

CSGAN</br>
(Reference: http://proceedings.mlr.press/v97/wu19d/wu19d.pdf)</br>
(Repository: https://github.com/deepmind/deepmind-research/tree/master/cs_gan)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* IF THERE IS NO DIFFERENCE IN RESULTS, SKIP THESE TWO POINTS
* compare to one in original paper
* significant modifications that we made to the code that affect the performance
