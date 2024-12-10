
<a id="readme-top"></a>

<!-- HEADER -->
<br />
<div align="center">
  <h3 align="center">Cross Attentive PET Image
Reconstruction Methods
  </h3>

  <p align="center">
    The Role of the Cross-Attention in the LPD
Reconstruction Algorithm
    <br />
  </p>
</div>
<p align="center">
  <a href="https://www.python.org/downloads/release/python-3119/"><img alt="python-version" src="https://img.shields.io/badge/python-3.11.9-blue"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Abstract

Positron Emission Tomography (PET) is a key diagnostic tool in oncology, cardiology and neurology. However, the inherent noise and sparsity of the acquisition process pose significant challenges for image reconstruction. State-of-the-art reconstruction methods often produce low-resolution images and struggle to preserve small details. This thesis studies and develops new reconstruction methods based on the Learned Primal-Dual (LPD) Reconstruction and enhanced by the Cross-Attention mechanism. A new synthetic generator capable of producing a wide variety of shapes was implemented, along with a new loss function, leading to improvements in both metrics and generalization power. Using these new elements, four different LPD architectures incorporating Cross-Attention were tested, achieving comparable performance to previous implementations. Although the Cross-Attention mechanism did not significantly improve the LPD reconstruction algorithm, the results suggest its potential in effectively integrating different information and are promising for future applications.

### Built With

- 🐍 [Python](https://www.python.org/)
- 🔥 [PyTorch](https://pytorch.org/)
- 🟰 [Odl](https://odlgroup.github.io/odl/)
- 🌐 [Parallelproj](https://parallelproj.readthedocs.io)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

Create the virtual environment

```sh
conda env create -n venv -f environment.yml
```

**(Optional)** Setup the Telegram integration by creating a `.env` file with the variables present `example.env`. This allows you to receive a notification from a Telegram bot when the training is finished.

### Usage

1. Select an existing `config.yml` file in the `./configs` folder or create a new one.
2. Train the model:
   ```sh
   python train_models.py <config_file_path>
   ```
The results are automatically saved in the  `./outputs` folder.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Simone Bonino - ✉️ [send an mail](mailto:simone@binarypillow.dev)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project was inspired by these repositories:

* [Deep Learning for PET Imaging](https://github.com/AlessandroGuazzo/Deep-Learning-for-PET-Imaging)
* [Transformer Based LPD Reconstruction for PET](https://github.com/antonadelow/Transformer-Based-Learned-Primal-Dual-Reconstruction-for-PET)
*  [Synthmorph implemented with PyTorch](https://github.com/matt-kh/synthmorph-torch)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
