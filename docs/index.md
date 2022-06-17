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

For this project, we are using Pytorch.

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
Data augmentation is often used to virtually increase the size of and diversify the dataset that a network is trained on to reduce overfitting phenomena. In our project we used data augmentation not only to reduce overfitting but also to make our data more compatible with the pretrained model. 
The following transformations are performed to the train set:
1. RandomPerspective
2. Resize(256)
3. CenterCrop(224, 224)
4. ColorJitter
5. Normalize(IMAGENET_MEAN, IMAGENET_STD)

Point 2, 3 and 5 are there to achieve the aforementioned compatibility with the pretrained model. This model was trained on imagenet and thus we thought it would be important to deliver our data in a similar format such that the network's fitting efforts aren't spent on learning the format of this new dataset.  
Point 1 and 4 are the actual data augmentations that have a probability and random strength associated with them. They were picked to prevent overfitting but also to stay close to the input domain. For example we don't flip the images vertically or rotate them since this would present the network with input that does not reflect true Streetview images. Instead, we alter the colour space of the images (hue, brightness, contrast) and change their perspective. In our opinion, these augmentations give images that stay somewhat close to the true input domain but still diversify the dataset.
We note that transformations 1 and 4 are performed only on the train set, not on the test set.


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


### Data augmentation vs. no data augmentation
We now compare the learning performance of the model with and without data augmentation. We will refer to the figure below in our analysis. In this figure the blue plot 'devout-morning-26' is where the dataset is not augmented. The brown plot 'different-flower-30' is where augmentation *is* employed.
<p align="center">
  <img src="https://i.imgur.com/xWPIzsg.png" />
</p>
Comparing the left and right image, it is clear that the brown plot represents an overfitting model. Indeed, its performance on the train dataset improves up to 100% accuracy but its performance on the test set reaches its limit rather quickly.  
The blue plot shows a more true representation of the model's capabilities. Although its ability to perform on the validation set is somewhat limited, we observe that its performance on the training set plateaus less severely.  
We can conclude that data augmentation is indeed essential for deep neural network training.

### Interactive prediction outputs: neighbourhood simularity


<iframe width="100%" height="500" src="http://www.sjoerdgroot.ml/Projects/London/" frameborder="0"></iframe>

## Conclusion ^^^ processen in de stukjes hierboven ^^^
- With a dataset limited to 10.000 images its more efficient to only unlock part of a pretrained model.
- Simularity of neighbourhoods can somewhat be infeard from the model predicitons.
- Its better to fintune a model with more images than more relevant images.

## Discussion/future work
### Approach
Upon reflection our approach deviates somewhat from what we initially set out to do. Currently, the model essentialy determines the similarity between different neighbourhoods. It however is not necessarily motivated to construct a latent space in which two nearby images that may be similar are placed together.  
An approach more in line with our initial intentions would have been to approach the problem not as one of classification but of similarity. This especially came to mind as we learned more about the metric learning work and in converstations with professor Seyran Khademi.  
Perhaps combined with the previous point, the graph-like structure of Streetview image locations could have been exploited in constructing the metric space. In e.g. a graph network representing the streets of London we would have not considered the Euclidian distance as a metric but Geodesic distance.  
^^ Mschn nog concreter omschrijven hoe we dit precies hadden aangepakt?


A more in-depth analysis of different data augmentations can be performed. An obvious augmentation we overlooked was horizontally flipping the images.

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