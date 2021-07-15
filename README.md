# COAST: COntrollable Arbitrary-Sampling NeTwork for Compressive Sensing [PyTorch]

This repository is for COAST introduced in the following paper

Di You, [Jian Zhang](http://jianzhang.tech/), Jingfen Xie, Bin Chen, and Siwei Ma. COAST: Controllable arbitrary-sampling network for compressive sensing. IEEE Transactions on Image Processing, 2021. [[pdf]](https://ieeexplore.ieee.org/abstract/document/9467810)

The code is built on **PyTorch** and tested on Ubuntu 16.04/18.04 and Windows 10 environment (Python3.x, PyTorch>=0.4) with 1080Ti GPU.

## Introduction
Recent deep network-based compressive sensing (CS) methods have achieved great success. However, most of them regard different sampling matrices as different independent tasks and need to train a specific model for each target sampling matrix. Such practices give rise to inefficiency in computing and suffer from poor generalization ability. In this paper, we propose a novel COntrollable Arbitrary-Sampling neTwork, dubbed COAST, to solve CS problems of arbitrary-sampling matrices (including unseen sampling matrices) with one single model. Under the optimization-inspired deep unfolding framework, our COAST exhibits good interpretability. In COAST, a random projection augmentation (RPA) strategy is proposed to promote the training diversity in the sampling space to enable arbitrary sampling, and a controllable proximal mapping module (CPMM) and a plug-and-play deblocking (PnP-D) strategy are further developed to dynamically modulate the network features and effectively eliminate the blocking artifacts, respectively. Extensive experiments on widely used benchmark datasets demonstrate that our proposed COAST is not only able to handle arbitrary sampling matrices with one single model but also to achieve state-of-the-art performance with fast speed.

![COAST](./Figs/framework.png)

Figure 1. Illustration of the proposed COAST framework.


## Contents
1. [Test-CS](#test-cs)
2. [Results](#results)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)


## Test-CS
### Quick start
1. All models for our paper have been put in './model'.
2. Please download sampling matrices from [BaiduPan [code: rgd9]](https://pan.baidu.com/s/1AFza-XCyTqRIVTdaYwjT3w).
3. Run the following scripts to test COAST model.

    **You can use scripts in file 'TEST_COAST_scripts.sh' to produce results for our paper.**

    ```bash
    # test scripts
    python TEST_COAST.py  --cs_ratio 10 --layer_num 20
    python TEST_COAST.py  --cs_ratio 20 --layer_num 20
    python TEST_COAST.py  --cs_ratio 30 --layer_num 20
    python TEST_COAST.py  --cs_ratio 40 --layer_num 20
    python TEST_COAST.py  --cs_ratio 50 --layer_num 20
    ```
    

### The whole test pipeline
1. Prepare test data.

    The original test set11 is in './data'

2. Run the test scripts. 

    See **Quick start**
3. Check the results in './result'.


## Results
### Quantitative Results

![Table_Results](./Figs/Table_Results.png)

### Visual Results

![visual_Results](./Figs/result.png)

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{you2021coast,
  title={COAST: COntrollable Arbitrary-Sampling NeTwork for Compressive Sensing},
  author={You, Di and Zhang, Jian and Xie, Jingfen and Chen, Bin and Ma, Siwei},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```
## Acknowledgements