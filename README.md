# Image registration and recognition model for the card game Set
	
The Set card game is awesome! However, sometimes it's tricky to check if a set exits on the board. The goal of this code is to make an image recognition project to do just that: to be able to take a photo of the board and output all possible sets. 

### Desired components
A brief rundown of what I want to build into this

#### Tasks
[x] Code to take in raw images of single cards and resize/pad
[x] Image augmentation to take a small number of photos of Set cards and generate more ad infinitum using imgaug library (which is rad, go use it!)
[x] Basic CNN model in TensorFlow (deprecated on its own branch now; I made a dogs breakfast of the git history there too, totes soz intrepid reader)
[x] Basiscally the same CNN model in Pytorch (current version)
[x] Basic image registration using countour detection from OpenCV2 (thanks to [here](https://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python) for code snippets and ideas)
[] Connect all the bits into a real pipeline which can be run in real time
[] Experiment with reducing the number of branches in the CNN by doing colour extraction based on pixel ratios (might be better, might be worse, not sure)
[] Flask plugin/API
[] Mobile app