# PUGAN

The pytorch implemtation of the paper [on positive-unlabeled classification in gan](https://arxiv.org/abs/2002.01136). 

The code is based on the code provided by https://github.com/AlexiaJM/RelativisticGAN.

# How to run the code?

**Requirement**
* lastest pytorch
* python 3.7.3
* lastest numpy

**Before you run the code**
* make sure all required folder are create
* if you want to use CAT dataset: Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)

**to run**
* run 'GAN_losses_iter.py' (please check the arguements in the code carefully, including the argument for change the model and hyperprarmeter)
