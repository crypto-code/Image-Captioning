# Image-Captioning

Uses a Convolutional Neural Network along with a Long Short Term Memory model Recurrent Neural Network to Caption Images.

## Neural Network Model 

- **The Encoder**: The CNN encodes the images into a feature matrix which is then reduced linearly to fit the LSTM input. In this, the CNN used is the resnet-152 model which is already trained on ILSVRC-2012-CLS image classification dataset.

- **The Decoder**: The LSTM decodes the linearly transformed input matrix into a list of sequenced words. In this model, the words generated in the previous iteration is passed on to the next until it reaches the end. 

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/model.png" align="middle" />   </p>

## Usage

- First, download the required training and validation data along with their annotations from the below links and extract to the data folder:
  - [Training Dataset](http://images.cocodataset.org/zips/train2014.zip)
  - [Validation Dataset](http://images.cocodataset.org/zips/val2014.zip)
  - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

- To install the coco module download the files from [here](https://github.com/pdollar/coco.git) and run the following in the main folder.
```
python setup.py build
python setup.py install
```

- Now download all the files from this repository and run the following to pre-process the dataset.
```
python resize.py
python build_vocab.py
```


