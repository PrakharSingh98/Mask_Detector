# Mask Detector
## Using transfer learning and openCV
1). This uses a pre-trained caffe model to extract features from a live video or any clip of your choice. I did not use opencv's haar cascade as it was not very accurate on my device. <br>
2). The extracted ROI is sent through a Inception model using imagenet weights whoose top layer was trained on a dataset of 2000 images where half of them belong to 'mask' class and other half to 'no_mask'. 
### How to use
Run the train.ipynb file if you want to train the model from scratch. <br>
Run the detect.ipynb file to run the mask detector from webcam or video (Default is set to webcam, if you want to change follow the instructions in comments) <br>
I have given the saved model trained on my device which took almost 8 hrs. <br>
### Requirements
<ul>
  <li>OpenCV</li>
  <li>Tensorflow == 1.14.x </li>
  <li>Keras</li>
  </ul>
 
