# [Mitigating Emergent Robustness Degradation on Graphs while Scaling-up](https://openreview.net/forum?id=Koh0i2u8qX)

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository hosts the official implementation of the paper "[Mitigating Emergent Robustness Degradation on Graphs while Scaling-up](https://openreview.net/forum?id=Koh0i2u8qX)", as presented at **ICLR 2024**.

## Introduction

Our research presents innovative strategies to address the challenge of emergent robustness degradation in graph neural networks as scale increases. This repository is structured to facilitate both reproduction of our results and application of our methods to new datasets and problems.

## Getting Started

### Prerequisites

- Python 3.x
- Pip package manager

### Installation

Clone the repository and install the required Python packages:

```shell
git clone https://github.com/chunhuizng/emergent-degradation.git
cd emergent-degradation
pip install -r requirements.txt
```

### Training Models

To train the model on a specific task, navigate to the appropriate directory and run the desired script. For example, to perform link prediction:

```shell
python training.py
```

### Adversarial Training and Attacks

To perform adversarial training or to simulate attacks using the moedp model:

```shell
python adv_training.py
python pipeline*.py
```

Replace `adv_training.py` with `attack.py` to execute an attack.


## Extending the Code

The repository is organized like GRB to facilitate easy extensions and modifications. Feel free to adapt the code to your requirements and contribute back any useful changes.

## Citation

Please cite our work if it helps your research:

```bibtex
@inproceedings{yuan2024mitigating,
  title={Mitigating Emergent Robustness Degradation on Graphs while Scaling-up},
  author={Xiangchi Yuan and Chunhui Zhang and Yijun Tian and Yanfang Ye and Chuxu Zhang},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Koh0i2u8qX}
}
```

## Questions and Contributions

For any questions about the code or contributions you'd like to make, please open an issue or a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.