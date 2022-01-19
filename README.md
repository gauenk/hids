
Patch Subsetting
=====

Summary
-----

For image denoising with non-local methods, using the choice of which non-local patches to use dramatically impacts denoising quality. For the Video Non-Local Bayes (VNLB) image denoisig method, if the patches are carefully chosen (explained below) the method's resulting PSNR can approximately match and is sometimes better than state-of-the-art deep learning based methods. This library implements a methods to subset the correct patches for the Bayes filter in image denoising.

The VNLB method uses sum-of-squared difference metric (SSD) commonly used to select the 60 or 100 patches from a search space of about 2,000 patches. Ideally, these patches should be "iid". Since the patches are corrupted with Gaussian noise, standard methods rank the quality of patches with respect to the mode of the resulting chi-squared distribution, e.g. <img src="https://render.githubusercontent.com/render/math?math=\|x_i - x_{ref}\|_2^2 - \sigma^2">. We want to use more than pairwise information to improve the patch matching quality.

