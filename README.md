## Music Recommendation based on emotion with Flask

### Datset
  Dataset based on FER-2013 dataset consisting of 48x48 grayscale images stored as pixel values (csv). 
### Model Layers
  Keras Sequential Model
  My model uses Conv2D to map over the image and extract features of it into a featue map.
  Then through Max pooling the most important features are extracted and parsed into the next layer.
  It is then Flattened out before parsing through the Dense layer. 
  Dropout is also implemented to generalize and prevent overfitting.
### Overall accuracy
  65% - 75% accuracy
### Flask
  Flask is used to render the HTML pages and control which URL displays them.
  The main app.py file contains code to capture video content using openCV and render to html page.
### Spotify API
  Based on emotion output a playlist and song tracks along with a link.
  Emotion is stored as a global variable which is passed into the music page.
### Docker and Travis CI
  Docker container is used by Travis CI to deploy it onto AWS Elastic Beanstalk
  **Issue with docker container getting feed from camera