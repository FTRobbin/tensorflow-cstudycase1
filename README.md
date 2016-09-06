# Tensorflow C Study - Case 1

With the most basic understanding of Tensorflow architecture in mind, one becomes realized that what python and C++ do in Tensorflow.

## Introduction

This is a tiny tutorial / example to show the following by creating a computational graph in python and then run / train the graph in c.

1. Tensorflow builds the computational graph in python, including three parts:
   1. Inference: Minimal graph to run the model forward
   2. Loss: Calculate defined loss function
   3. Training: Calculate the derivative and update the parameters according to an optimizer
2. Tensorflow has control on training in python.
3. Tensorflow has the actual computation done in c(e.g. the Session). But c API just calculates the computational graph built in python that passed by protobuf and it's unaware of any higher abstractions.
   1. You have to explicitly name the things you would like to see in c in python or you will have trouble finding them(tf just gives them names like "Variable_1")

## Usage

1. Install `python`, `numpy`, `tensorflow` and `bazel`
2. Run `buildgraph.py` and get `graph.pb` which is the binary protobuf file describes the graph. (Another file `text.pb` is the same graph in human-friendly text)
3. Clone the tensorflow [repo](https://github.com/tensorflow/tensorflow)
4. Create a new folder `tensorflow/tensorflow/trainer` and put `trainer.cc` and `BUILD` file in it
5. In `trainer` folder, run `bazel build :trainer`. This should take about ten minutes
6. Copy the `graph.pb` to `repo_root/bazel-bin/tensorflow/trainer`
7. Run the program by `./trainer` in the `repo_root/bazel-bin/tensorflow/trainer` folder
8. It will print the results of approximating the function $y = x * 0.1 + 0.3$ by $y = W * x + b$

---

Authored by [Haobin Ni](https://github.com/FTRobbin) Aug-Sep 2016
