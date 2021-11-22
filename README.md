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
    cd DIPFKP
    python main.py --SR --sf 4 --dataset Test
    ```

**BEFORE RUN**: please make sure that you edit the **overwritting paths, DECLARE PATH AT THIS SECTION** in the above file.

By default, the results should appear in the following path: ./data/log_DIPFKP/{dataset_name}_x{scale}_{noise_level}

## Acknowledgements
We thank Jingyun Liang for releasing his [code](https://github.com/JingyunLiang/FKP) that helped in the early stages of this project.

We also thank Zongsheng Yue for the work at [this](https://github.com/zsyOAOA/VDNet) which contributes to the NP module of our project.



## Requirements
- Python 3.6/3.7.9, PyTorch >= 1.5.1 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 20.04, cuda-10.2 & cuDNN v-7.5

## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows), [DIP](https://github.com/DmitryUlyanov/deep-image-prior), [KernelGAN](https://github.com/sefibk/KernelGAN) and [USRNet](https://github.com/cszn/KAIR). Please also follow their licenses. Thanks for their great works.


