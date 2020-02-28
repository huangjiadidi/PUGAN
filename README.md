# PUGAN

The pytorch implemtation of the paper [on positive-unlabeled classification in gan](https://arxiv.org/abs/2002.01136). 

The code is based on the code provided by https://github.com/AlexiaJM/RelativisticGAN.

# Implementation

**Requirement**
* pytorch (lastest version)
* python 3.7.3
* numpy (lastest version)
* tensorflow (lastest version, don't need to download it you don't need to calculcate fid)
* [Cat Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd) (if you want to use it)


**Before you run the code**
* make sure all required folders are created, including a output folder to save model, a extra folder to save generated images and a inception folder for inception model. 
* if you want to use CAT dataset: Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)

**usage**
* run 'GAN_losses_iter.py' (please check the arguements in the code carefully, including the argument for change the model and hyperprarmeter)

e.g. to train a PUSGAN model with 64x64 size images: `python3 GAN_losses_iter.py --image_size=64 --loss_D=5 --prior=0.3 --prior_increase_mode=1 --input_folder=/path/to/input_image/dir/ --output_folder=/path/to/output/dir --extra_folder=/path/to/generated_image/dir --inception_folder=/path/to/inception/dir`

**to calculate the FID sorce**
* make sure you save the generated images in the extra folder for calculation
* run `python fid.py "/path/to/saved_generated_image/dir/01" "/path/to/real_image/dir" -i "/path/to/Inception/dir" --gpu "0"`

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


# the hyper parameter prior, indicating the proportion of real data in the mixed data (you can change it such as increasing it during the training processes)
prior = 0.3

# calculating the D pu loss for standard GAN 
errD_real = BCE_stable(y_pred, y)
errD_real_f = BCE_stable(y_pred,y2)
errD_fake = BCE_stable(y_pred_fake, y2)

errD_positive_risk = prior * errD_real
errD_negative_risk = errD_fake - prior * errD_real_f

zero.data.fill_(0)
errD = errD_positive_risk + torch.max(zero, errD_negative_risk)

# G loss for PUSGAN
errG = BCE_stable(y_pred_fake, y)


# calculating the D pu loss for LSGAN
errD_real = torch.mean((y_pred - y) ** 2)
errD_real_f = torch.mean((y_pred + y) ** 2)
errD_fake =  torch.mean((y_pred_fake + y) ** 2)
errD_positive_risk = prior * errD_real
errD_negative_risk = errD_fake - prior * errD_real_f
zero.data.fill_(0)
errD = errD_positive_risk + torch.max(zero, errD_negative_risk)

# G loss for PULSGAN
errG = torch.mean((y_pred_fake - y) ** 2)
```
