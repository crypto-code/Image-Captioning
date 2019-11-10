# Image-Captioning

Uses a Convolutional Neural Network along with a Long Short Term Memory model Recurrent Neural Network to Caption Images.

## Neural Network Model 

The CNN first reduces the image to a feature matrix which is then passed through the LSTM network which forms the caption.

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/model.png" align="middle" />   </p>

## Usage

- First, download the required training and validation data along with their annotations from the below links:
  - [Training Dataset](http://images.cocodataset.org/zips/train2014.zip)
  - [Validation Dataset](http://images.cocodataset.org/zips/val2014.zip)
  - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
