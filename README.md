# Traffic-Sign-AI
Uses Tensorflow Keras API to create a Convolutional Neural Network that can detect and classify 43 types of traffic signs in an image. Data is provided by German Traffic Sign Recognition Benchmark Dataset.

First install packages:
    
    pip3 install opencv-python scikit-learn tensorflow

Download Dataset at http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads

Run by:
    
    python3 traffic.py dataset_dir model.h5

where dataset_dir is the location of the folder containing the dataset and model.h5 is the filename that the model should be saved to.

The model that has been created with current code is on repository, this is its evaluation and training:
![Image of Code](https://github.com/ivinable2000/Traffic-Sign-AI/blob/master/images/code.png)
