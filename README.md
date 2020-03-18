# SINDySR3

Repository for the paper [``A unified sparse optimization framework to learn parsimonious physics-informed models from data``](https://arxiv.org/pdf/1906.10612.pdf) by Kathleen Champion, Peng Zheng, Aleksandr Y. Aravkin, Steven L. Brunton, and J. Nathan Kutz.

This work implements the [sparse regularized relaxed regression](https://ieeexplore.ieee.org/abstract/document/8573778) (SR3) optimization method for [sparse identification of nonlinear dynamics](https://www.pnas.org/content/113/15/3932) (SINDy), including a number of extensions to the algorithm for corrupt data trimming, incorporating physical constraints, and fitting parameterized library functions. Code examples can be found as jupyter notebooks in the `examples` directory.

## Dependencies

This code requires the installation of the [PySINDy](https://github.com/dynamicslab/pysindy) package.