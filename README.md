# few-shot-project

This project demonstrates Few Shot Image Classification. Traditionally, if you want to classify two images, you need thousands of images to train a model. But now, using few shot learning algorithms models can learn to classify images with just a few images.

## To run:

`python main.py`

## How to tweak and experiment:

The model will learn from the adaptation images, and then it can predict the labels of images in the folder `testset`.

```
.
├── ...
├── adaptset
│   ├── 0.jpg         # adaptation image of label 0
│   ├── 1.jpg         # adaptation image of label 1
│   └── testset       # Test Images
│       ├──testimage1.jpg       # Label of this image will be predicted by the model
└── ...
```

You can put other images in the testset folder and run the code, which will predict its label.

You can also change images `0.jpg` and `1.jpg`, which can enable you to predict different classes of images.


0.jpg             |  1.jpg             |  testimage1.jpg
:-------------------------:|:-------------------------:|:-------------------------:
![](/adaptset/0.jpg)  |  ![](/adaptset/1.jpg)  |  ![](/adaptset/testset/testimage1.jpg)
