# PUGAN

The pytorch implemtation of the paper [on positive-unlabeled classification in gan](https://arxiv.org/abs/2002.01136). 

The paper has been accepted by CVPR 2020.

# Implementation

**Requirement**
* pytorch (lastest version)
* python 3.7.3
* numpy (lastest version)
* tensorflow (lastest version, don't need to download it you don't need to calculcate fid)

**Before you run the code**
* make sure all required folders are created, including a output folder to save model, an extra folder to save generated images and an inception folder for inception model. 

* if you want to use [Cat Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd): Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)

**Usage**
* run 'GAN_losses_iter.py' (please check the arguements in the code carefully, including the argument for change the model and hyperprarmeter)

* e.g. to train a PUSGAN model with 64x64 size images: `python3 GAN_losses_iter.py --image_size=64 --loss_D=5 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --prior=0.3 --prior_increase_mode=1 --seed=1 --input_folder=/path/to/input_image/dir/ --output_folder=/path/to/output/dir --extra_folder=/path/to/generated_image/dir --inception_folder=/path/to/inception/dir`

* notes: similar to RelativisticGAN, the random seed for training input is 1 constantly. It is important to notice that, although the random seedes are same, but the trained result will still be various.

**to calculate the FID sorce**
* make sure you save the generated images in the extra folder for calculation
* run `python fid.py "/path/to/saved_generated_image/dir/01" "/path/to/real_image/dir" -i "/path/to/Inception/dir" --gpu "0"`

# The algorithm of PUGAN

* PUGAN combines the concept of positive unlabeled classification with GAN. In the later stage of training, some generated data could be very similar to real data which we shouldn't consider them as the fake data and distinguish them strictly. Instead, let's consider the generated dataset as the mixed dataset, including positive data (high quality generated data) and negative data (the low quality generated data). Thus, the target of discriminator will be changed to: find the positive data in the mixed data based on the real data and separate them with negative data. In this case, we expect that the generator will focus on improving the low-quality results, and less modification on high-quality results.

* the general loss function will be:
![](https://latex.codecogs.com/gif.latex?%5Cunderset%7BG%7D%7Bmax%7D%5C%20%5Cunderset%7BD%7D%7Bmin%7DV%28D%2C%20G%29%20%3D%20%5Cpi%5Cmathbb%7BE%7D_%7Bp_%7Bdata%7D%7D%5Bf_1%28D%28x%29%29%5D%20&plus;%20max%5C%7B0%2C%20%5Cmathbb%7BE%7D_%7Bp_z%7D%5Bf_2%28D%28G%28z%29%29%29%5D%5C%7D%20-%20%5Cpi%5Cmathbb%7BE%7D_%7Bp_%7Bdata%7D%7D%5Bf_2%28D%28x%29%29%5D%5C%7D)

It is easy to adapt PU concept into different framework of GAN.



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

# Citation
``@inproceedings{guo2020positiveunlabeled,
    title={On Positive-Unlabeled Classification in GAN},
    author={Tianyu Guo and Chang Xu and Jiajun Huang and Yunhe Wang and Boxin Shi and Chao Xu and Dacheng Tao},
    year={2020},
    booktitle={CVPR}
}``
