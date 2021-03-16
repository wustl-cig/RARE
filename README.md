# [RARE: Image Reconstruction using Deep Priors Learned without Ground Truth](https://ieeexplore.ieee.org/abstract/document/9103213)

Regularization by denoising (RED) is an image reconstruction framework that uses an image denoiser as a prior. Recent work has shown the state-of-the-art performance of RED with learned denoisers corresponding to pre-trained convolutional neural nets (CNNs). In this work, we propose to broaden the current denoiser-centric view of RED by considering priors corresponding to networks trained for more general artifact-removal. The key benefit of the proposed family of algorithms, called regularization by artifact-removal (RARE), is that it can leverage priors learned on datasets containing only undersampled measurements. This makes RARE applicable to problems where it is practically impossible to have fully-sampled groundtruth data for training. We validate RARE on both simulated and experimentally collected data by reconstructing a free-breathing whole-body 3D MRIs into ten respiratory phases from heavily undersampled k-space measurements. Our results corroborate the potential of learning regularizers for iterative inversion directly on undersampled and noisy measurements. The supplementary material of this paper can be found[here](https://wustl.app.box.com/s/bntylvjqets7dhhf7tm83io7k90r6193). The talk is available [here](https://www.youtube.com/watch?v=dOqNbsQbpxc).

![visualpipline](figs/pipline.png "Visual illustration of reconstructed images of RARE")

## How to run the code

### Prerequisites for numpy-mcnufft

tqdm
python 3.6  
tensorflow 1.13 or lower  
scipy 1.2.1 or lower  
numpy v1.17 or lower  
matplotlib v3.1.0
### Prerequisites for torch-mcnufft

above prerequisites +
pytorch 1.13 or lower

It is better to use Conda for installation of all dependecies.

### Run the Demo

to demonstrate the performance of RARE with freath-breath 4D MRI, you can run the RARE by typing

```
$ python demo_RARE_np.py
```

or

```
$ python demo_RARE_torch.py
```

The per iteration results will be stored in the ./Results folder. The torch-mcnufft is a more efficient implementation using gpu backend. (Thanks [wjgancn](https://github.com/wjgancn) for his help in pytorch-mcnufft.)

Visual results of RARE
----------
![visualExamples](figs/rareVSn2n.png "Visual illustration of reconstructed images of RARE")

### CNN model
The training code for artifact-to-artifact (A2A) convolutional neural network is coming soon. The pre-trained models are stored under the ./models folder. Feel free to download and test it.

### Citation
If you find the paper useful in your research, please cite the paper:
```BibTex
@ARTICLE{Liu.etal2020,
  author={J. {Liu} and Y. {Sun} and C. {Eldeniz} and W. {Gan} and H. {An} and U. S. {Kamilov}},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={RARE: Image Reconstruction using Deep Priors Learned without Ground Truth}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},}

```