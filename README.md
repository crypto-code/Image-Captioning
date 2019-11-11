# Image-Captioning

Uses a Convolutional Neural Network along with a Long Short Term Memory model Recurrent Neural Network to Caption Images.

## Neural Network Model 

- **The Encoder**: The CNN encodes the images into a feature matrix which is then reduced linearly to fit the LSTM input. In this, the CNN used is the resnet-152 model which is already trained on ILSVRC-2012-CLS image classification dataset.

- **The Decoder**: The LSTM decodes the linearly transformed input matrix into a list of sequenced words. In this model, the words generated in the previous iteration is passed on to the next until it reaches <end>. 

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/model.png" align="middle" />   </p>

## Usage

- First, download the required training and validation data along with their annotations from the below links and extract to the data folder:
  - [Training Dataset](http://images.cocodataset.org/zips/train2014.zip)
  - [Validation Dataset](http://images.cocodataset.org/zips/val2014.zip)
  - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
