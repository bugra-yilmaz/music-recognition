# music-recognition

A computer vision application for automatically playing the music provided in a music sheet image.

## Install
### Python dependencies
This application requires ```Python 3.9``` installation. In order to install required Python libraries, run:
```bash
pip install -r requirements.txt
```

### Pretrained musical object detection model
Also, you need to download a pretrained musical object detection model. You can download this model with the following [link](https://github.com/apacha/MusicObjectDetector-TF/releases/download/full-page-detection-v2/2018-07-30_faster-rcnn_inception-resnet-v2_full-page_muscima.pb). Then, save the ```.pb``` file with the name *model.pb* in the *resources* directory.

### Proto files
In order to use TensorFlow's object detection libraries, you need to compile the ```.proto``` files. 

If you don't have the proto compiler, install it first with Homebrew:
```bash
brew install protobuf
```

Then run:
```bash
protoc object_detection/protos/*.proto --python_out=.
```
