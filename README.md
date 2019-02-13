# Python + TensorFlow model for the card game Set
	
The Set card game is awesome! However, sometimes it's tricky to check if a set exits on the board. The goal of this code is to make an image recognition project to do just that: to be able to take a photo of the board and output all possible sets. 

### Desired components
A brief rundown of what I want to build into this

#### Completed
- Code to take in raw images of single cards and resize/pad
- Image augmentation to take a small number of photos of Set cards and generate more ad infinitum using imgaug library (which is rad, go use it!)
- Basic CNN model in TensorFlow. It sucks (see below)

#### Current WIP
-  Basic CNN sucks and prone to overfitting (I am unsurprise). The usual suspects of dropout etc don't seem to help (other than to bring down the training accuracy). Fix this
- Batch normalisation
- Move all the model definition stuff to a class to enable more modular model building 
- Basic unit tests inspired by https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
- Watered down version of CNN: map to grey scale and drop colour requirement, un-flatten classes i.e. fit on shape, style, and fill separately instead of different classes for each card. Presumably the CNN would do this anyway in feature extraction, but it might need some help.

#### Planned
- Need a way to split up the board into individual cards, so may need a more general card/edge-detector
- Flask plugin/API