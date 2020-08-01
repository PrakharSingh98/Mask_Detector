# Mask Detector
## Using transfer learning and openCV
This uses a pre-trained caffe model to extract features from a live video or any clip of your choice. I did not use opencv's haar cascade as it was not very accurate on my device.
The extracted ROI is sent through a Inception model using imagenet weights whoose top layer was trained on a dataset of 2000 images where half of them belong to 'mask' class and other half to 'no_mask'. 
