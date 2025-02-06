# When Heterophily Meets Heterogeneous Graphs: Latent Graphs Guided Unsupervised Representation Learning

This is the source code of "When Heterophily Meets Heterogeneous Graphs: Latent Graphs Guided Unsupervised Representation Learning" (LatGRL), accepted by IEEE Transactions on Neural Networks and Learning Systems 2025 (TNNLS).

Paper Link: [https://arxiv.org/abs/2409.00687](https://arxiv.org/abs/2409.00687)

# Available Data

Datasets: https://drive.google.com/file/d/12nWwcrufexpU1n6W7YTyb-6sL1wQ-WrY/view?usp=sharing

**Place the 'data' folder from the downloaded files into the 'LatGRL' directory.**

# Requirements

This code requires the following:

* Python==3.9.16
* PyTorch==1.13.1
* Numpy==1.24.2
* Scipy==1.10.1
* Scikit-learn==1.2.1
* Munkres==1.1.4

# Training

`python main.py -dataset acm`
Here, "acm" can be replaced by "dblp", "yelp","imdb".


# BibTeX

```
@article{shen2024latgrl,
  title={When Heterophily Meets Heterogeneous Graphs: Latent Graphs Guided Unsupervised Representation Learning},
  author={Shen, Zhixiang and Kang, Zhao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}

```
