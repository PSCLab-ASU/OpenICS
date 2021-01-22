


# Section 0: Table of Benchmarks
<table>
<thead>
  <tr>
    <th>Method</th>
    <th>Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Learned DAMP</td>
    <td>20.72298384</td>
  </tr>
  <tr>
    <td>ISTA-Net Plus</td>
    <td>16.68532509</td>
  </tr>
  <tr>
    <td>CSGAN</td>
    <td>19.77839493</td>
  </tr>
  <tr>
    <td>LAPRAN</td>
    <td>25.51094228</td>
  </tr>
  <tr>
    <td>CSGM</td>
    <td>5.935114887</td>
  </tr>
  <tr>
    <td>TVAL3</td>
    <td>12.2197737</td>
  </tr>
  <tr>
    <td>TV</td>
    <td>7.859866999</td>
  </tr>
  <tr>
    <td>D-AMP</td>
    <td>7.160057161</td>
  </tr>
  <tr>
    <td>NLR-CS</td>
    <td>6.601654874</td>
  </tr>
</tbody>
</table>


# Section 1: Setting up environment

### Data-driven using conda (PyTorch):
`conda create -n CS_methods_environement python=3.7.9` </br>
`conda activate CS_methods_environement`</br>
</br>
`conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch`</br>
`conda install opencv==3.4.2`</br>
`conda install scikit-image==0.17.2`</br>
`conda install pandas`</br>
 
 ### Data-driven using conda (Tensorflow):
`conda create -n CS_methods_environement python=3.7.9` </br>
`conda activate CS_methods_environement`</br>
</br>
`conda install tensorflow-gpu=1.15 cudatoolkit=10.0`</br>
`conda install opencv==3.4.2`</br>
`conda install scikit-image==0.17.2`</br>
`pip install pypng`</br>
`pip install PyWavelets`</br>
`pip install scipy==1.1.0`</br>
`pip install matplotlib`</br>
`pip install scikit-learn`</br>
`pip install cvxopt`</br>
`pip install Pillow`</br>

### Model-based using Matlab:
`cd \<path to CS-Framework directory\>`<br>
`run set_up.m`<br>

### Using the framework:
#### Data-driven using Python
To run the methods in Python, you may either modify the `main.py` file in each method's folder and run `python main.py`, or call it from another file and pass in custom arguments. Details about the specifics parameter may be found in each method's folder, and will require modification to properly work with different file structures. Note that the correct conda environment must be active for it to function properly.

Pre-trained models for each data-driven method may be downloaded from [this link](https://google.com). <!--ADD IN GOOGLE DRIVE LINK-->

#### Model-based using Matlab
To run the methods in Matlab, you may either modify the `demo.m` file in the framework root directory and run it in Matlab, or call the `main.m` function from another file and pass in custom arguments. Details about the specifics parameter may be found in each method's folder. Note that `set_up.m` must be ran in each new session for it to function properly.



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
LDAMP </br>
(Reference: https://papers.nips.cc/paper/2017/file/8597a6cfa74defcbde3047c891d78f90-Paper.pdf)</br>
(Repository: https://github.com/ricedsp/D-AMP_Toolbox/tree/master/LDAMP_TensorFlow)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* compare to one in original paper
* only supports layer-by-layer training
* supports gaussian, complex-gaussian, and coded-diffraction sensing for Layer-by-Layer (not Fast-JL)

ISTA-Net Plus </br>
(Reference: https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper.pdf)</br>
(Repository: https://github.com/jianzhangcs/ISTA-Net)</br>
* list main (we choose specific dataset and specific cs ratio) benchmark results
* compare to one in original paper
* merged ISTANet and ISTANetPlus, parameter used to control which one to train/test

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

# Section 4: Method Results
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
    <td rowspan="30">LDAMP<br>Layer-by-Layer</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>48</td>
    <td>0.9999</td>
    <td>0.2493</td>
  </tr>
  <tr>
    <td>4</td>
    <td>47.2738</td>
    <td>0.9997</td>
    <td>0.2468</td>
  </tr>
  <tr>
    <td>8</td>
    <td>30.4165</td>
    <td>0.9704</td>
    <td>0.2455</td>
  </tr>
  <tr>
    <td>16</td>
    <td>24.7801</td>
    <td>0.9287</td>
    <td>0.1686</td>
  </tr>
  <tr>
    <td>32</td>
    <td>14.0546</td>
    <td>0.5527</td>
    <td>0.053</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>40.5767</td>
    <td>0.9986</td>
    <td>0.9729</td>
  </tr>
  <tr>
    <td>4</td>
    <td>32.1129</td>
    <td>0.9911</td>
    <td>0.9403</td>
  </tr>
  <tr>
    <td>8</td>
    <td>27.5632</td>
    <td>0.9759</td>
    <td>0.9421</td>
  </tr>
  <tr>
    <td>16</td>
    <td>23.7575</td>
    <td>0.9409</td>
    <td>0.9446</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.6439</td>
    <td>0.7866</td>
    <td>0.1988</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>35.1305</td>
    <td>0.995</td>
    <td>0.2303</td>
  </tr>
  <tr>
    <td>4</td>
    <td>28.6455</td>
    <td>0.9793</td>
    <td>0.2411</td>
  </tr>
  <tr>
    <td>8</td>
    <td>22.7897</td>
    <td>0.9323</td>
    <td>0.2393</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.395</td>
    <td>0.823</td>
    <td>0.1454</td>
  </tr>
  <tr>
    <td>32</td>
    <td>13.7681</td>
    <td>0.4758</td>
    <td>0.0194</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>34.5538</td>
    <td>0.9945</td>
    <td>0.245</td>
  </tr>
  <tr>
    <td>4</td>
    <td>28.1537</td>
    <td>0.9775</td>
    <td>0.2466</td>
  </tr>
  <tr>
    <td>8</td>
    <td>23.3344</td>
    <td>0.938</td>
    <td>0.2582</td>
  </tr>
  <tr>
    <td>16</td>
    <td>17.2754</td>
    <td>0.7614</td>
    <td>0.069</td>
  </tr>
  <tr>
    <td>32</td>
    <td>13.7009</td>
    <td>0.4419</td>
    <td>0.0369</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>37.2695</td>
    <td>0.9951</td>
    <td>1.0422</td>
  </tr>
  <tr>
    <td>4</td>
    <td>33.8795</td>
    <td>0.9879</td>
    <td>0.9857</td>
  </tr>
  <tr>
    <td>8</td>
    <td>31.0234</td>
    <td>0.9742</td>
    <td>0.9805</td>
  </tr>
  <tr>
    <td>16</td>
    <td>28.01</td>
    <td>0.9524</td>
    <td>0.9892</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.51</td>
    <td>0.7634</td>
    <td>0.2282</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>40.4972</td>
    <td>0.9954</td>
    <td>1.1842</td>
  </tr>
  <tr>
    <td>4</td>
    <td>37.3146</td>
    <td>0.9881</td>
    <td>1.1625</td>
  </tr>
  <tr>
    <td>8</td>
    <td>34.1416</td>
    <td>0.975</td>
    <td>1.1563</td>
  </tr>
  <tr>
    <td>16</td>
    <td>32.9576</td>
    <td>0.9651</td>
    <td>1.574</td>
  </tr>
  <tr>
    <td>32</td>
    <td>13.5395</td>
    <td>0.5392</td>
    <td>0.2627</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>

<br>

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
    <td rowspan="30">ISTA-Net Plus</td>
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
    <td>21.13</td>
    <td>0.6295</td>
    <td>0.057</td>
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

<br>

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
    <td rowspan="30">TVAL3</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>47.995</td>
    <td>1.000</td>
    <td>0.061</td>
  </tr>
  <tr>
    <td>4</td>
    <td>33.233</td>
    <td>0.879</td>
    <td>0.081</td>
  </tr>
  <tr>
    <td>8</td>
    <td>20.587</td>
    <td>0.542</td>
    <td>0.081</td>
  </tr>
  <tr>
    <td>16</td>
    <td>15.291</td>
    <td>0.299</td>
    <td>0.063</td>
  </tr>
  <tr>
    <td>32</td>
    <td>13.076</td>
    <td>0.163</td>
    <td>0.062</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>32.335</td>
    <td>0.959</td>
    <td>1.373</td>
  </tr>
  <tr>
    <td>4</td>
    <td>26.592</td>
    <td>0.889</td>
    <td>0.799</td>
  </tr>
  <tr>
    <td>8</td>
    <td>22.863</td>
    <td>0.801</td>
    <td>0.796</td>
  </tr>
  <tr>
    <td>16</td>
    <td>19.919</td>
    <td>0.703</td>
    <td>0.709</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.345</td>
    <td>0.599</td>
    <td>0.713</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>29.584</td>
    <td>0.936</td>
    <td>0.213</td>
  </tr>
  <tr>
    <td>4</td>
    <td>24.001</td>
    <td>0.822</td>
    <td>0.217</td>
  </tr>
  <tr>
    <td>8</td>
    <td>20.621</td>
    <td>0.690</td>
    <td>0.208</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.286</td>
    <td>0.573</td>
    <td>0.191</td>
  </tr>
  <tr>
    <td>32</td>
    <td>16.401</td>
    <td>0.476</td>
    <td>0.187</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>29.766</td>
    <td>0.900</td>
    <td>0.069</td>
  </tr>
  <tr>
    <td>4</td>
    <td>24.189</td>
    <td>0.742</td>
    <td>0.076</td>
  </tr>
  <tr>
    <td>8</td>
    <td>20.778</td>
    <td>0.577</td>
    <td>0.073</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.343</td>
    <td>0.446</td>
    <td>0.065</td>
  </tr>
  <tr>
    <td>32</td>
    <td>16.661</td>
    <td>0.362</td>
    <td>0.062</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>35.830</td>
    <td>0.960</td>
    <td>1.256</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.084</td>
    <td>0.905</td>
    <td>0.763</td>
  </tr>
  <tr>
    <td>8</td>
    <td>27.632</td>
    <td>0.845</td>
    <td>0.752</td>
  </tr>
  <tr>
    <td>16</td>
    <td>24.754</td>
    <td>0.786</td>
    <td>0.700</td>
  </tr>
  <tr>
    <td>32</td>
    <td>22.109</td>
    <td>0.729</td>
    <td>0.703</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>36.154</td>
    <td>0.915</td>
    <td>0.608</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.368</td>
    <td>0.814</td>
    <td>0.278</td>
  </tr>
  <tr>
    <td>8</td>
    <td>27.871</td>
    <td>0.711</td>
    <td>0.256</td>
  </tr>
  <tr>
    <td>16</td>
    <td>24.954</td>
    <td>0.620</td>
    <td>0.234</td>
  </tr>
  <tr>
    <td>32</td>
    <td>22.054</td>
    <td>0.545</td>
    <td>0.244</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>
<br>
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
    <td rowspan="30">TV</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>47.911</td>
    <td>0.999</td>
    <td>1.167</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.010</td>
    <td>0.803</td>
    <td>1.213</td>
  </tr>
  <tr>
    <td>8</td>
    <td>19.881</td>
    <td>0.489</td>
    <td>1.213</td>
  </tr>
  <tr>
    <td>16</td>
    <td>14.158</td>
    <td>0.225</td>
    <td>0.987</td>
  </tr>
  <tr>
    <td>32</td>
    <td>11.762</td>
    <td>0.100</td>
    <td>0.849</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>32.578</td>
    <td>0.961</td>
    <td>32.568</td>
  </tr>
  <tr>
    <td>4</td>
    <td>27.003</td>
    <td>0.897</td>
    <td>20.542</td>
  </tr>
  <tr>
    <td>8</td>
    <td>23.359</td>
    <td>0.816</td>
    <td>19.367</td>
  </tr>
  <tr>
    <td>16</td>
    <td>20.721</td>
    <td>0.731</td>
    <td>18.818</td>
  </tr>
  <tr>
    <td>32</td>
    <td>18.235</td>
    <td>0.633</td>
    <td>21.241</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>30.078</td>
    <td>0.943</td>
    <td>3.341</td>
  </tr>
  <tr>
    <td>4</td>
    <td>24.486</td>
    <td>0.837</td>
    <td>3.154</td>
  </tr>
  <tr>
    <td>8</td>
    <td>21.115</td>
    <td>0.709</td>
    <td>3.580</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.697</td>
    <td>0.584</td>
    <td>3.158</td>
  </tr>
  <tr>
    <td>32</td>
    <td>16.746</td>
    <td>0.476</td>
    <td>2.675</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>30.231</td>
    <td>0.910</td>
    <td>1.271</td>
  </tr>
  <tr>
    <td>4</td>
    <td>24.619</td>
    <td>0.758</td>
    <td>0.997</td>
  </tr>
  <tr>
    <td>8</td>
    <td>21.328</td>
    <td>0.596</td>
    <td>1.011</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.822</td>
    <td>0.452</td>
    <td>0.981</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.046</td>
    <td>0.344</td>
    <td>0.895</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>36.084</td>
    <td>0.961</td>
    <td>28.520</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.600</td>
    <td>0.910</td>
    <td>19.144</td>
  </tr>
  <tr>
    <td>8</td>
    <td>28.475</td>
    <td>0.855</td>
    <td>18.062</td>
  </tr>
  <tr>
    <td>16</td>
    <td>25.951</td>
    <td>0.802</td>
    <td>17.608</td>
  </tr>
  <tr>
    <td>32</td>
    <td>23.821</td>
    <td>0.754</td>
    <td>16.688</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>36.425</td>
    <td>0.919</td>
    <td>11.601</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.881</td>
    <td>0.823</td>
    <td>6.424</td>
  </tr>
  <tr>
    <td>8</td>
    <td>28.687</td>
    <td>0.727</td>
    <td>6.090</td>
  </tr>
  <tr>
    <td>16</td>
    <td>26.128</td>
    <td>0.640</td>
    <td>5.747</td>
  </tr>
  <tr>
    <td>32</td>
    <td>24.035</td>
    <td>0.570</td>
    <td>5.986</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>

<br>

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
    <td rowspan="30">D-AMP</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>43.450</td>
    <td>0.972</td>
    <td>2.871</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.787</td>
    <td>0.891</td>
    <td>2.836</td>
  </tr>
  <tr>
    <td>8</td>
    <td>22.582</td>
    <td>0.724</td>
    <td>2.935</td>
  </tr>
  <tr>
    <td>16</td>
    <td>13.238</td>
    <td>0.322</td>
    <td>3.054</td>
  </tr>
  <tr>
    <td>32</td>
    <td>6.530</td>
    <td>0.093</td>
    <td>3.131</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>47.126</td>
    <td>0.998</td>
    <td>17.592</td>
  </tr>
  <tr>
    <td>4</td>
    <td>37.495</td>
    <td>0.985</td>
    <td>18.618</td>
  </tr>
  <tr>
    <td>8</td>
    <td>31.095</td>
    <td>0.950</td>
    <td>21.815</td>
  </tr>
  <tr>
    <td>16</td>
    <td>26.328</td>
    <td>0.877</td>
    <td>22.199</td>
  </tr>
  <tr>
    <td>32</td>
    <td>21.517</td>
    <td>0.723</td>
    <td>19.991</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>40.401</td>
    <td>0.993</td>
    <td>4.567</td>
  </tr>
  <tr>
    <td>4</td>
    <td>31.595</td>
    <td>0.957</td>
    <td>4.752</td>
  </tr>
  <tr>
    <td>8</td>
    <td>25.851</td>
    <td>0.867</td>
    <td>5.061</td>
  </tr>
  <tr>
    <td>16</td>
    <td>20.248</td>
    <td>0.645</td>
    <td>5.314</td>
  </tr>
  <tr>
    <td>32</td>
    <td>8.307</td>
    <td>0.113</td>
    <td>5.427</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>31.682</td>
    <td>0.929</td>
    <td>2.773</td>
  </tr>
  <tr>
    <td>4</td>
    <td>25.616</td>
    <td>0.789</td>
    <td>2.838</td>
  </tr>
  <tr>
    <td>8</td>
    <td>19.996</td>
    <td>0.537</td>
    <td>2.943</td>
  </tr>
  <tr>
    <td>16</td>
    <td>16.365</td>
    <td>0.342</td>
    <td>2.926</td>
  </tr>
  <tr>
    <td>32</td>
    <td>8.016</td>
    <td>0.074</td>
    <td>2.943</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>42.510</td>
    <td>0.992</td>
    <td>21.075</td>
  </tr>
  <tr>
    <td>4</td>
    <td>37.565</td>
    <td>0.977</td>
    <td>19.227</td>
  </tr>
  <tr>
    <td>8</td>
    <td>33.950</td>
    <td>0.949</td>
    <td>19.371</td>
  </tr>
  <tr>
    <td>16</td>
    <td>30.973</td>
    <td>0.908</td>
    <td>20.815</td>
  </tr>
  <tr>
    <td>32</td>
    <td>27.300</td>
    <td>0.832</td>
    <td>20.164</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>38.205</td>
    <td>0.940</td>
    <td>8.172</td>
  </tr>
  <tr>
    <td>4</td>
    <td>34.138</td>
    <td>0.870</td>
    <td>8.185</td>
  </tr>
  <tr>
    <td>8</td>
    <td>30.774</td>
    <td>0.786</td>
    <td>8.224</td>
  </tr>
  <tr>
    <td>16</td>
    <td>27.234</td>
    <td>0.670</td>
    <td>8.193</td>
  </tr>
  <tr>
    <td>32</td>
    <td>20.727</td>
    <td>0.435</td>
    <td>8.226</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>

<br>

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
    <td rowspan="30">NLR-CS</td>
    <td rowspan="5">MNIST</td>
    <td>2</td>
    <td>40.062</td>
    <td>0.912</td>
    <td>3.177</td>
  </tr>
  <tr>
    <td>4</td>
    <td>29.452</td>
    <td>0.778</td>
    <td>3.064</td>
  </tr>
  <tr>
    <td>8</td>
    <td>18.787</td>
    <td>0.472</td>
    <td>3.058</td>
  </tr>
  <tr>
    <td>16</td>
    <td>15.058</td>
    <td>0.312</td>
    <td>3.080</td>
  </tr>
  <tr>
    <td>32</td>
    <td>12.052</td>
    <td>0.132</td>
    <td>3.069</td>
  </tr>
  <tr>
    <td rowspan="5">CelebA</td>
    <td>2</td>
    <td>35.932</td>
    <td>0.978</td>
    <td>42.007</td>
  </tr>
  <tr>
    <td>4</td>
    <td>29.288</td>
    <td>0.930</td>
    <td>42.841</td>
  </tr>
  <tr>
    <td>8</td>
    <td>23.970</td>
    <td>0.831</td>
    <td>43.301</td>
  </tr>
  <tr>
    <td>16</td>
    <td>21.211</td>
    <td>0.743</td>
    <td>42.828</td>
  </tr>
  <tr>
    <td>32</td>
    <td>17.848</td>
    <td>0.601</td>
    <td>43.07</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10</td>
    <td>2</td>
    <td>30.627</td>
    <td>0.938</td>
    <td>9.811</td>
  </tr>
  <tr>
    <td>4</td>
    <td>24.975</td>
    <td>0.843</td>
    <td>9.482</td>
  </tr>
  <tr>
    <td>8</td>
    <td>21.008</td>
    <td>0.700</td>
    <td>9.476</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.564</td>
    <td>0.576</td>
    <td>9.409</td>
  </tr>
  <tr>
    <td>32</td>
    <td>16.928</td>
    <td>0.477</td>
    <td>9.32</td>
  </tr>
  <tr>
    <td rowspan="5">CIFAR10 Gray</td>
    <td>2</td>
    <td>31.140</td>
    <td>0.912</td>
    <td>3.217</td>
  </tr>
  <tr>
    <td>4</td>
    <td>25.336</td>
    <td>0.781</td>
    <td>3.019</td>
  </tr>
  <tr>
    <td>8</td>
    <td>21.215</td>
    <td>0.598</td>
    <td>3.035</td>
  </tr>
  <tr>
    <td>16</td>
    <td>18.810</td>
    <td>0.463</td>
    <td>3.029</td>
  </tr>
  <tr>
    <td>32</td>
    <td>16.921</td>
    <td>0.353</td>
    <td>3.014</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset</td>
    <td>2</td>
    <td>38.114</td>
    <td>0.973</td>
    <td>41.701</td>
  </tr>
  <tr>
    <td>4</td>
    <td>33.706</td>
    <td>0.938</td>
    <td>41.190</td>
  </tr>
  <tr>
    <td>8</td>
    <td>30.003</td>
    <td>0.883</td>
    <td>40.988</td>
  </tr>
  <tr>
    <td>16</td>
    <td>27.201</td>
    <td>0.831</td>
    <td>40.240</td>
  </tr>
  <tr>
    <td>32</td>
    <td>24.191</td>
    <td>0.758</td>
    <td>40.191</td>
  </tr>
  <tr>
    <td rowspan="5">Bigset Gray</td>
    <td>2</td>
    <td>38.697</td>
    <td>0.942</td>
    <td>13.891</td>
  </tr>
  <tr>
    <td>4</td>
    <td>34.293</td>
    <td>0.874</td>
    <td>13.732</td>
  </tr>
  <tr>
    <td>8</td>
    <td>30.449</td>
    <td>0.781</td>
    <td>13.770</td>
  </tr>
  <tr>
    <td>16</td>
    <td>27.393</td>
    <td>0.692</td>
    <td>13.501</td>
  </tr>
  <tr>
    <td>32</td>
    <td>24.426</td>
    <td>0.592</td>
    <td>13.443</td>
  </tr>
  <tr>
    <td colspan="6">THIS IS EXAMPLE TEXT, PUT THE TOTAL SCORE CALCULATION HERE</td>
  </tr>
</tbody>
</table>

<br>
