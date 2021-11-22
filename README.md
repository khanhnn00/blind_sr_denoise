# A Demonstration of Single Image Super-Resolution
### Setup
0. **Minimum requirements.** This project was originally developed with Python 3.6, PyTorch 1.7 and CUDA 11.1. The training requires at least a Titan X GPUs (12Gb memory each).
1. **Setup your Python environment.** Please, clone the repository and install the dependencies. We recommend using Anaconda 3 distribution:
    ```
    conda create -n <environment_name> --file requirements.txt
    ```

3. **Download pre-trained models.** For running the code, you need to download the two pre-trained weights of Kernel Prior (KP) and Noise Prior (NP) module. Here we use FKP as KP and VDNet as NP. The pretrained weight can be found [here](https://drive.google.com/drive/folders/1PVGobcRyYqHwnT2MhUUSLb6xZK5h0-B1?usp=sharing)

### How to Run

First you need to place the pre-trained weight of KP and NP to anywhere you want.

Next, run the following command:
    ```
    python ./DIPFKP/main.py
    ```

**BEFORE RUN**: please make sure that you edit the **overwritting paths, DECLARE PATH AT THIS SECTION** in the above file.

By default, the results should appear in the following path: ./data/log_DIPFKP/{dataset_name}_x{scale}_{noise_level}

## Acknowledgements
We thank Jingyun Liang for releasing his [code](https://github.com/JingyunLiang/FKP) that helped in the early stages of this project.
We also thank Zongsheng Yue for the work at [this](https://github.com/zsyOAOA/VDNet) which contributes to the NP module of our project.



## Requirements
- Python 3.6, PyTorch >= 1.6 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5


## Quick Run
To run the code without preparing data, run this command:
```bash
cd DIPFKP
python main.py --SR --sf 4 --dataset Test
```

---

## Data Preparation
To prepare testing data, please organize images as `data/datasets/DIV2K/HR/0801.png`, and run this command:
```bash
cd data
python prepare_dataset.py --model DIPFKP --sf 2 --dataset Set5
python prepare_dataset.py --model KernelGANFKP --sf 2 --dataset DIV2K
```
Commonly used datasets can be downloaded [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets). Note that KernelGAN/KernelGAN-FKP use analytic X4 kernel based on X2, and do not support X3.

## FKP

To train FKP, run this command:

```bash
cd FKP
python main.py --train --sf 2
```
Pretrained FKP and [USRNet](https://github.com/cszn/KAIR) models are already provided in `data/pretrained_models`.


## DIP-FKP

To test DIP-FKP (no training phase), run this command:

```bash
cd DIPFKP
python main.py --SR --sf 2 --dataset Set5
```


## KernelGAN-FKP

To test KernelGAN-FKP (no training phase), run this command:

```bash
cd KernelGANFKP
python main.py --SR --sf 2 --dataset DIV2K
```

## Results
Please refer to the [paper](https://arxiv.org/pdf/2103.15977.pdf) and [supplementary](https://github.com/JingyunLiang/FKP/releases) for results. Since both DIP-FKP and KernelGAn-FKP are randomly intialized, different runs may get slightly different results. The reported results are averages of 5 runs.



## Citation
```
@article{liang21fkp,
  title={Flow-based Kernel Prior with Application to Blind Super-Resolution},
  author={Liang, Jingyun and Zhang, Kai and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2103.15977},
  year={2021}
}
```


## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows), [DIP](https://github.com/DmitryUlyanov/deep-image-prior), [KernelGAN](https://github.com/sefibk/KernelGAN) and [USRNet](https://github.com/cszn/KAIR). Please also follow their licenses. Thanks for their great works.


