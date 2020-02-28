# PUGAN

The pytorch implemtation of the paper [on positive-unlabeled classification in gan](https://arxiv.org/abs/2002.01136). 

The code is based on the code provided by https://github.com/AlexiaJM/RelativisticGAN.

# How to run the code?

**Requirement**
* lastest pytorch
* python 3.7.3
* lastest numpy
* lastest tensorflow (if you want to use our FID code)
* [Cat Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd) (if you want to use it)


**Before you run the code**
* make sure all required folders are created (check the argument help of code)
* if you want to use CAT dataset: Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)

**to run**
* run 'GAN_losses_iter.py' (please check the arguements in the code carefully, including the argument for change the model and hyperprarmeter)

**to calculate the FID sorce**
* make sure you save the generated images for calculation
* run 'fid.py' (please check the arguements before you run)

# The algorithm of PUGAN

```python

# pre processes
x.data.resize_as_(images).copy_(images)
y_pred = D(x)
y.data.resize_(current_batch_size).fill_(1)
y2.data.resize_(current_batch_size).fill_(0)
z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
fake = G(z)
x_fake.data.resize_(fake.data.size()).copy_(fake.data)
y_pred_fake = D(x_fake.detach())


# calculating the D loss for PUGAN
errD_real = BCE_stable(y_pred, y)
errD_real_f = BCE_stable(y_pred,y2)
errD_fake = BCE_stable(y_pred_fake, y2)

errD_positive_risk = prior * errD_real
errD_negative_risk = errD_fake - prior * errD_real_f

zero.data.fill_(0)
errD = errD_positive_risk + torch.max(zero, errD_negative_risk)
```
