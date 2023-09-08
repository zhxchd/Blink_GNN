# Blink: Link Local Differential Privacy in Graph Neural Networks via Bayesian Estimation

This is the code repository for our paper "Blink: Link Local Differential Privacy in Graph Neural Networks via Bayesian Estimation" to appear in CCS '23. You can read the preprint paper [here](https://arxiv.org/abs/2309.03190).

## Dependencies

To run the experiments in this repo, you need `numpy`, `matplotlib`, `sklearn`, `torch`, `torch_sparse`, `torch_geometric`. You can install all the dependencies is through `conda` and `pip` (please use the `CUDA` version applicable to your system):

```
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install torch_geometric
pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip3 install scikit-learn
```

## File structure
- `./src`: the source directory of all the mechanisms, datasets and models we have experimented with.
  - `./src/blink` implements our main result, the Blink framework.
  - `./src/rr` implements vanilla randomized response as a baseline.
  - `./src/ldpgcn` implement a LDP variant of DPGCN from [Wu et al (2022)](https://ieeexplore.ieee.org/document/9833806).
  - `./src/solitude` tries to implement Solitude from [Lin et al (2022)](https://ieeexplore.ieee.org/document/9855440).
  - `./src/data` contains all the code to download, pre-process and load graph datasets including Cora, CiteCeer and LastFM.
  - `./src/models` contains all the code to build GNN models including GCN, GraphSage and GAT.
- `./scripts` is the directory of Python scripts to run experiments.
  - `./scripts/run_blink.sh` runs the Blink framework with specified settings.
  - `./scripts/run_baselines.sh` runs baseline methods with specified settings.
  - `./scripts/log` stores all the log files when running the scripts above.
  - `./scripts/output` stores all the results (hyperparameter choices and final accuracy).
- `./doc` is the root directory for the paper describing the proposed method.

## Running
Inside directory, you can run experiments with `python3 run_blink.py {variant name} {dataset} {model_name} --eps {epsilon_list}`, like:
```
python3 run_blink.py hybrid cora gcn --eps 1
```

## Citation
Please cite our paper as follows:
```
@inproceedings{zhu2023blink,
  author      = {Zhu, Xiaochen and Tan, Vincent Y. F. and Xiao, Xiaokui},
  title       = {Blink: Link Local Differential Privacy in Graph Neural Networks via Bayesian Estimation},
  year        = {2023},
  booktitle   = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  location    = {Copenhagen, Denmark},
  series      = {CCS '23}
}
```

## License
The code and documents are licensed under the MIT license.
```
MIT License

Copyright (c) 2022 Xiaochen Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```