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

- To train the image captioning, run
```
python train.py
```

- To test the trained model, run
```
python test.py --image=[image path]
```

## All Possible Arguments

- For resize.py
```
usage: resize.py [-h] [--image_dir IMAGE_DIR] [--output_dir OUTPUT_DIR]
                 [--image_size IMAGE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        directory for train images
  --output_dir OUTPUT_DIR
                        directory for saving resized images
  --image_size IMAGE_SIZE
                        size for image after processing
```

- For build_vocab.py
```
usage: build_vocab.py [-h] [--caption_path CAPTION_PATH]
                      [--vocab_path VOCAB_PATH] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --caption_path CAPTION_PATH
                        path for train annotation file
  --vocab_path VOCAB_PATH
                        path for saving vocabulary wrapper
  --threshold THRESHOLD
                        minimum word count threshold
```

- For train.py
```
usage: train.py [-h] [--model_path MODEL_PATH] [--crop_size CROP_SIZE]
                [--vocab_path VOCAB_PATH] [--image_dir IMAGE_DIR]
                [--caption_path CAPTION_PATH] [--log_step LOG_STEP]
                [--save_step SAVE_STEP] [--embed_size EMBED_SIZE]
                [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path for saving trained models
  --crop_size CROP_SIZE
                        size for randomly cropping images
  --vocab_path VOCAB_PATH
                        path for vocabulary wrapper
  --image_dir IMAGE_DIR
                        directory for resized images
  --caption_path CAPTION_PATH
                        path for train annotation json file
  --log_step LOG_STEP   step size for prining log info
  --save_step SAVE_STEP
                        step size for saving trained models
  --embed_size EMBED_SIZE
                        dimension of word embedding vectors
  --hidden_size HIDDEN_SIZE
                        dimension of lstm hidden states
  --num_layers NUM_LAYERS
                        number of layers in lstm
  --num_epochs NUM_EPOCHS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --learning_rate LEARNING_RATE
```

- For test.py
```
usage: test.py [-h] [--image IMAGE] [--encoder_path ENCODER_PATH]
               [--decoder_path DECODER_PATH] [--vocab_path VOCAB_PATH]
               [--embed_size EMBED_SIZE] [--hidden_size HIDDEN_SIZE]
               [--num_layers NUM_LAYERS]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         input image for generating caption
  --encoder_path ENCODER_PATH
                        path for trained encoder
  --decoder_path DECODER_PATH
                        path for trained decoder
  --vocab_path VOCAB_PATH
                        path for vocabulary wrapper
  --embed_size EMBED_SIZE
                        dimension of word embedding vectors
  --hidden_size HIDDEN_SIZE
                        dimension of lstm hidden states
  --num_layers NUM_LAYERS
                        number of layers in lstm
```


## Examples

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/dog.jpg" align="middle" />   </p>
<p align = "center"><b>\<start\> a dog is playing with a frisbee in a field . \<end\></b></p>

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/baseball.jpg" align="middle" />   </p>
<p align = "center"><b>\<start\> a baseball player swinging a bat at a ball \<end\></b></p>

<p align="center">
<img src="https://github.com/crypto-code/Image-Captioning/blob/master/assets/plane.jpg" align="middle" />   </p>
<p align = "center"><b>\ a large jetliner flying through the sky with a sky background \<end\></b></p>


# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
