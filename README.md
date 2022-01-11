
HIDS: Hypothesis testing influence functions for data subsampling
=====

Summary
-----

For image denoising with non-local methods, using the choice of which non-local patches to use dramatically impacts denoising quality. For the Video Non-Local Bayes (VNLB) image denoisig method, if the patches are carefully chosen (explained below) the method's resulting PSNR can approximately match and is sometimes better than state-of-the-art deep learning based methods. This library, HIDS, implements a class of solutions to select the correct patches for the Bayes filter in image denoising.

The VNLB method uses sum-of-squared difference metric (SSD) commonly used to select the 60 or 100 patches from a search space of about 2,000 patches. Ideally, these patches should be "iid". Since the patches are corrupted with Gaussian noise, standard methods rank the quality of patches with respect to the mode of the resulting chi-squared distribution, e.g. <img src="https://render.githubusercontent.com/render/math?math=\|x_i - x_{ref}\|_2^2 - \sigma^2">. We consider the SSD as the t-statistics from a Hotelling T-test using known covariance terms (<img src="https://render.githubusercontent.com/render/math?math=\sigma^2 I">). Then we can apply some of the zoo of exisiting hypothesis testing methods to this problem to improve the set of 60 - 100 patches used to construct the Bayes filter.

Since the power of hypothesis tests usually increase with dataset size, brute-force searching the combinatorial number of options seems unreasonable in this context. To reduce the search complexity, we use a continuous-relaxation of our discrete problem to identify a "good" (e.g. iid) subset of 60 - 100. 

This package implements some of these hypotheses testing methods for the data subsampling problem.
