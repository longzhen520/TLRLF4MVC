# TLRLF4MVC

## Introduction

TLRLF4MVC is a scalable multi-view clustering method that efficiently handles large datasets by ensuring smooth intra-view representation and capturing high-order correlations across views. It introduces a **Tensor Low-Frequency Component (TLFC) operator** for intra-view consistency and a **Tensor Nuclear Norm (TNN) operator** for inter-view correlation balancing. Experimental results demonstrate that TLRLF4MVC outperforms state-of-the-art methods in both clustering accuracy and computational efficiency.

This repository provides the implementation of TLRLF4MVC and allows users to reproduce the results presented in our paper.

---

## Run the Code

To run the TLRLF4MVC model on the `CCV` dataset, execute the following script in MATLAB:

```matlab
run('demo.m');
```

This script will load the dataset, execute the clustering algorithm, and display the results.

---

## Datasets

The datasets used in this project are organized as follows:

- **`datasets/` folder:**
  - `CCV`
  - `ALOI`

Additional datasets are available via the following links due to storage limitations. These datasets require **Graph Fourier Transform (GFT)** preprocessing, which can be generated using the code in `demo.m` or directly loaded from existing `transform` datasets. Note that if GFT is generated from the provided code, the optional parameters need to tune accrodingly.

### Download Links:
- **Google Drive:** [Download datasets](https://drive.google.com/drive/folders/1WSdsMHInccSFEEAbSjcCqhFkqQsc3-vE?usp=sharing)
- **Baidu Netdisk:** [Download datasets](https://pan.baidu.com/s/12fSX996FbfTSdIsXQMe8Mg?pwd=3u7q) (Password: 3u7q)

Once downloaded, place them in the appropriate folders:

- **`datasets/` folder:**
  - `Caltech102`
  - `NUSWIDEOBJ`
  - `AwAfea`
  - `cifar10`
- **`transform/` folder:**
  - `CCV_transform`
  - `ALOI_transform`
  - `Caltech102_transform`
  - `NUSWIDEOBJ_transform`
  - `AwAfea_transform`
  - `cifar10_transform`

---

## Contact

For any questions or issues related to this repository, please feel free to contact us at [zhen.long@uestc.edu.cn](mailto:zhen.long@uestc.edu.cn) or open an issue in the repository.

---

Enjoy using TLRLF4MVC!

