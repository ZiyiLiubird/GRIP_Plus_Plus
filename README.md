# GRIP++

This repository is the re-implementation of [GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving](https://arxiv.org/abs/1907.07792) on the Baidu Apollo Trajectory dataset. GRIP++ is an enhanced version of GRIP ([GRIP: Graph-based Interaction-aware Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/8917228)).

The official codes can be found here [GRIP++](https://github.com/xincoder/GRIP).

Our implementation has several advantages over official codes:

1. User friendly graph data preprocessing.

2. Using graph convolutional network rather than Conv2d: The official codes implemented GCN using Conv2d.

3. Better performance: Our model has better performance than the official implementation under same hyperparameters.

|                         | WSADE | ADEv  | ADEp  | ADEb  |
|-------------------------|-------|-------|-------|-------|
| Our implementation      | **1.133** | 1.705 | **0.696** | **1.762** |
| official implementation | 1.176 | 1.652 | 0.764 | 1.829 |


---

1. data generation. 

```python

python data_process.py

```

2. training model.

```python

python main.py

```
