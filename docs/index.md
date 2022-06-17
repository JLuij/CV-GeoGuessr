# Computer Vision by Deep Learning (CS4245)
*by Sjoerd Groot, Douwe den Blanken & Joost Luijmes (June 2022)*
Group 15

This blog post describes the project we undertook for the course Computer Vision by Deep Learning. We describe the inspiration, process, results and, at the end, reflect on these. The blog is structured as follows:
* Objective of the project
* The data
* Methods and process
* Results
* Discussion


## The objective

(We can all check this)

We were inspired by the game GeoGuessr in which a player is presented with a Google Streetview interface and is tasked to infer from their surroundings their location in the world. 

In a previous project by other students, it was shown that CV models could accuratly distinguish google street view images between 10 diffrent cities. In this project we tackle the problem of locating an image within one of the cities, namely the city center of London.

Our hypothesis was that, although images within a city will be way more simular, it should be possible to to distinguish between diffrent neighbourhood.


## The data

(Sjoerd will add data density plot)

In this project the dataset collected by a previous student project was used. This data was collected by Laura Muntenaar, Jetse Spijkstra, Maarten Mens, Rik van der Hoorn and Sjoerd Groot. 

This dataset consisted of a training set of 100.000 images and a test set of 20.000 images across 10 diffrent cities. These images where randomly sampled within a predefined zone from Google Street View.

Only the 10.000 images from London where used in this project. This results in a image density of at most roughly 2 images form each street.
![](https://i.imgur.com/Ac9czcr.png)






## Methods and process

### Grid creation and considered prediction output types

Outputing a grid of probabilities gives an idea of how
(Sjoerd)

### Data augmentation
(Joost)

### Used model

(Douwe)

(Using a pretrained model and locking (part of) its weights)

(talk about diff resnets)

### Training

(Douwe, which loss etc., how do we do the setup)


## Results

### ImageNet vs Places pretrained weights

(Douwe)

### Unlocking different amount of layers

(Sjoerd)

### Comparison of grid division
Iets over underrepresentation in square cells


### Data augmentation vs no data augmentation

(Joost)

### Interactive prediction outputs: neighbourhood simularity


<iframe width="100%" height="500" src="http://www.sjoerdgroot.ml/Projects/London/" frameborder="0"></iframe>

## Conclusion ^^^ processen in de stukjes hierboven ^^^
- With a dataset limited to 10.000 images its more efficient to only unlock part of a pretrained model.
- Simularity of neighbourhoods can somewhat be infeard from the model predicitons.
- Data augmentations helps?
- Its better to fintune a model with more images than more relevant images.

## Discussion/future work
**Instead of classification problem, approach as similarity problem (metric learning)**
**Instead of euclidian distance, geodesic distance**

(Douwe + ... + Joost :)

### Group Member Contributions
- *Sjoerd Groot*
- *Douwe den Blanken*
- *Joost Luijmes*


### Reproducing our results
bring your own data ;p
just run notebook


### References
- [1] 