# Development of the program "Object recognition by shape"
The code contains a set of functions that are used to classify fruit images.

To do this, the image is preprocessed using the Canny filter, converting it into a black-and-white image and calculating the distances from white pixels to the center of the image.

Then, using the distance variance, it is determined which type of fruit corresponds to this image.

The code also contains functions for calculating accuracy, sensitivity, specificity and other indicators that are used to assess the quality of classification.

Most functions take as parameters the paths to the files with training and test images, the threshold value of brightness, the radius of the circle in the center of the image, which should be excluded from the calculation of distances, as well as the threshold value of the variance of distances to determine the type of fruit.
