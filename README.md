It is an important step in brain science research to segment the  brain in medical image. Considering the time and energy consumption of artificial marking, the current research objective is to achieve automatic segmentation of target brain images via artificial intelligence technology.
We used U-Net to extract rubrum, substantia nigra in 3D brain images, and the results are below. 

![image](https://github.com/Charlotte12L/UNet-Brain-Segmentation/blob/master/results.jpg)

The visuliaztion interface:
![image](https://github.com/Charlotte12L/UNet-Brain-Segmentation/blob/master/visualization.png)

subj1.nii is a original image and subj1_OUT.nii is the result of the model. trainlog.out shows the training process and we achieved dice loss below 25%.

This is our first project related to deep learning (June-Oct 2018), so as newbies, we employed a crude method: slicing 3D brain images to 2D for training and predicting, then reconstructing the 3D images.

## Usage

### Prediction

To see all options:
`python predict.py -h`

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To specify which model file to use with `--model MODEL.pth`.

### Training

`python train.py -h` 
