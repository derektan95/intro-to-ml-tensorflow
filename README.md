# ND230 Introduction to Machine Learning with TensorFlow Nanodegree Program
## Courses: Supervised Learning and Deep Learning

Projects and exercises for the Udacity Intro to Machine Learning with TensorFlow course.

 <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.

## Installation
Creating conda environment from scratch:
```bash
conda create env -n tf-intro python==3.8
conda activate tf-intro
conda install tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0  # Depends on computer (Works on RTX 2060)
conda install matlibplot tensorflow-datasets jupyter notebook
pip install h5py   # Current version not compatible with tf for model saving and loading...
```

Creating conda environment from scratch:
```bash
cd intro_to_ml_tensorflow
conda env create -f tf_intro_env.yml
```
