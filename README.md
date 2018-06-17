# Intro to Machine Learning Labs

## Lab 1: Intro to ML Packages: SKLearn
Lab 1 can be found at `lab1`.

First, we'll walk through `lab1/sentiment_analysis_tutorial.ipynb` together, and cover how to build and tune models on SKLearn. Next, you'll apply what you learned, and work through `lab1/news_group_exercise.ipynb` yourself.

We provide a sample solution in `lab1/news_group_exercise.ipynb`.

## Lab 2: Intro to DNN Packages: PyTorch
Lab 2 can be found at `lab2`.

First, we'll walk through `lab2/mnist_tutorial.ipynb` together, and cover how to build and tune models on PyTorch. Next, you'll apply what you learned, and work through `lab1/cifar_exercise.ipynb` yourself.

We provide a sample solution in `lab1/news_group_exercise.ipynb`.


## Running the labs
The labs are jupyter notebooks, so run `jupyter notebook` in this directory, and then click on the proper file name in jupyter to open it.


## Requirments:
Please install `python3.6` and `pip` before coming to lab. If you have any problems please post on piazza.


## Lab1 Install Instructions
Please copy the following commands to your bash terminal:

```
git clone https://github.com/yala/MLCodeLab.git;
cd MLCodeLab;
pip install virtualenv;
virtualenv intro_ml;
source intro_ml/bin/activate;
pip install --upgrade pip;
pip install -r requirements.txt;
```

Then visit the [PyTorch website](https://pytorch.org/), and follow their instalation instructions your OS and python 3.6.
Instead of `pip3`, please use `pip`.


Finally, run:
```jupyter notebook```

### If you don't have git
- Just click "Clone or Download Button" and then click 'Download Zip' on this page: [https://github.com/yala/MLCodeLab](https://github.com/yala/MlCodeLab)
