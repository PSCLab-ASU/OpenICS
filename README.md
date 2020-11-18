
# Section 0:
	TABLE OF BENCHMARKS


# Section 1: Setting up environment

### Datadriven using conda:
conda create -n CS_methods_environement python=3.7.9 </br>
conda activate CS_methods_environement</br>
</br>
conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch</br>
conda install scipy==1.5.2</br>
conda install opencv==3.4.2</br>
conda install scikit-image==0.17.2</br>
conda install tensorflow-gpu==1.15</br>
 
### Matlab:





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