<head>
    <style>
        .photos img{
          display: inline;
          vertical-align: top;
          float: none;
        }
    </style>
</head>

# Playing GeoGuessr with machine learning

*By: Sjoerd Groot, Douwe den Blanken & Joost Luijmes*

*Course: Computer Vision by Deep Learning (CS4245)*

*Date: 17th of June 2022*

*Group: 15*

---

This blog post describes the work we have done for the Computer Vision by Deep Learning project. We will set out the inspiration, process and results of this project, and at the end, reflect on these. This blog is structured as follows:

* Objective of the project
* Data
* Methods and process
* Results
* Discussion

With the introduction out of the way, we just want to mention that we hope that you will enjoy reading our blog :smile:.


## The objective

For this project, we were inspired by the game [GeoGuessr](https://www.geoguessr.com/), in which a player is presented with a Google Street View photo and is tasked to infer from the surroundings, the photo's location in the world. The closer the player guesses to the actual location, the higher the score in the game will be.

In a previous course project (EWI3615TU), in which one of the authors of this article collaborated, it was already shown that Computer Vision models can accurately distinguish between Google Street View images of 10 different cities. In this project however, we will tackle the problem of locating an image within one of the cities, namely the city (center) of London.

We hypothesized that, although images within a city will be way more similar to each other (than images from different cities), it should be possible to distinguish between different neighbourhoods/areas quite well. However, we do not expect to get street-level prediction accuracy.

## The data 👨‍💻👩‍💻

In this project the dataset collected by the previous course project was used. This data was collected by Laura Muntenaar, Jetse Spijkstra, Maarten Mens, Rik van der Hoorn and Sjoerd Groot. 

The dataset consisted of a training set of 100.000 images and a test set of 20.000 images across 10 different cities. These images were randomly sampled within a predefined zone from Google Street View. From this dataset the 10.000 images from London were used in this project. This results in an image density of at most roughly 2 images for each street.

![](https://i.imgur.com/Ac9czcr.png)

Below is an overview of the datapoints. Red are queries without corresponding Streetview images. Green are positions that had a streetview image downloaded at the nearest blue point. Finally, purple points are positions where no image was downloaded because an image had already been downloaded from that point (to prevent duplicates and overlap between the train and test set).

![](https://i.imgur.com/b8IJH38.jpg)


## Methods and process

In this section, we will go over the various methods and procedures that were applied for different parts of the project. These parts were used interchangeably throughout our experiments. Please note that we built all our models using Pytorch.

### Used models

For this problem, many different machine learning could be used. However, in [1], one of the most-cited papers in geo-location estimation using machine learning, a ResNet-50 architecture is used. Combined with the fact that we were also interested in transfer learning and given that many applications of transfer learning use the ResNet-50 architecture, we decided to go with this architecture as well. 

However, even when limiting the choice to ResNet-50 models, there is still a large set of possible models: the most popular being ResNet-50 models pre-trained on [ImageNet](https://www.image-net.org/). Thus, for this project we decided to go with this model as well.

However, one of the things we were interested in checking, is the effect of the type of data that was used for pre-training on the model's final performance. In order to do this, we also used a ResNet-50 pre-trained on [Places365](http://places2.csail.mit.edu/download.html).

### Grid creation and considered prediction output types

In our first approach, we divided up the data into square cells for the sake of simplicity. These cells were all of equal area however this did not mean that they encapsulated a similar number of images. This is the case for the cells that lie on the boundary of the dataset and therefore only partly overlap with the location of images.

<div style="text-align: center; margin:0; auto;">
    <div class="photos">
<!--         <img src="https://i.imgur.com/TFfHc02.png" height="auto" width="300"/> -->
<!--         <img class="photos" src="https://i.imgur.com/40cue5J.png"/ height="auto" width="300"/> -->      
        <img class="photos" src="https://i.imgur.com/RJ0bW0D.png"/ height="auto" width="350"/>
        <img class="photos" src="https://i.imgur.com/GrLALYv.png"/ height="auto" width="350"/>
    </div>
</div>
<!-- ![](https://i.imgur.com/GrLALYv.png) -->

<!-- ![](https://i.imgur.com/RJ0bW0D.png) -->


Instead we decided to look into a way of subdividing the city boundary in a more equal way. We settled on Voronoi subdivision. For this we randomly sample points within the outer boundary. If a point does not fall within a margin range from the desired distance from the map border or existing points, the random point will be rejected. Otherwise the point is added to the list of Voronoi centers. Once no more points can be placed, these points are used to generate the Voronoi cells with scipy's Voronoi function. These are then turned into cell geometry, the end result beings cells that are roughly equally sized and completely fall within the dataset boundary.

Next to this, we also experimented with predicting the coordinates of an image directly. To do this, the final layer in the ResNet-50 was replaced with a 256 neuron layer, followed by a ReLU, followed by a 2 neuron layer: one neuron predicting latitude, the other one predicting longitude. Both of these values were scaled to always be between 0 and 1.

**resultaten toevoegen?**

### (Un)locking different amounts of layers

Only training the last layer doesn't provide the maximum possible accuracy while unlocking all layers suffers from overfitting unless you have a sufficiently large dataset. That's why we wanted fine grained control on how many layers we would keep locked so that we could experiment on where the balance lies between underfitting and overfitting.

Whereas in TensorFlow you can iterate over the individual layers as shown [here](https://www.tensorflow.org/tutorials/images/transfer_learning#un-freeze_the_top_layers_of_the_model), this was not easily possible in PyTorch with our Resnet-50 model.

```python3
# Example of locking layers in Tensorflow
for layer in model.layers[:fine_tune_at]:
  layer.trainable = False
```

We couldn't find an easy way of being able to individually lock/unlock the 174 layers shown by `torchsummary`. We therefore adapted the `torchsummary` code to return its ordered dict with layers. After this we could lock the desired amount of layers.

```python3
layers = model_layers_from_torchsummary(model, input_shape)

n = len(layers)

for i, layer in enumerate(layers):
    for param in layers[layer]["params"]:
        param.requires_grad = i > n * lock_factor
```


### Data augmentation
Data augmentation is often used to virtually increase the size of and diversify the dataset that a network is trained on to reduce overfitting phenomena. In our project we used data augmentation not only to reduce overfitting but also to make our data more compatible with the pretrained model. 
The following transformations are performed to the train set:
1. RandomPerspective
2. Resize(256)
3. CenterCrop(224, 224)
4. ColorJitter
5. Normalize(`IMAGENET_MEAN`, `IMAGENET_STD`)

Point 2, 3 and 5 are there to achieve the aforementioned compatibility with the pretrained model (see "Used models" section).

Point 1 and 4 are the actual data augmentations that have a probability and random strength associated with them. They were picked to prevent overfitting but also to stay close to the input domain. For example, we don't flip the images vertically or rotate them since this would present the network with input that does not reflect true Streetview images. Instead, we alter the colour space of the images (hue, brightness, contrast) and change their perspective. In our opinion, these augmentations yield images that stay somewhat close to the true input domain but still diversify the dataset.
We note that transformations 1 and 4 are performed only on the train set, not on the test set.


### Training

The training procedure for all experiments is the same. Firstly, we always apply data augmentation, except in experiments in which we want to quantify the effect of not using data augmentation.

Next, for the versions of the model where we try to classify in which grid cell the photo was taken, we use cross entropy loss, whereas for the coordinate regression version, we use mean squared error.

As optimizer, we use stochastic gradient descent with a learning rate of 0.001 and a momentum of 0.9. Furthermore, we decay the learning rate with a factor 0.99 every epoch. An upper limit of 25 epochs was maintained, as this gave the model plenty time to converge to its optimum.

Finally, in order to keep track of all of our experiments, we made use of [Weights and Biases](https://wandb.ai/site) to track the train and validation performance of each model we tried.


## Results

In this section, we will go over the results of the experiments that we have run for this project.

### ImageNet vs Places365 pretrained weights

The first experiment that we ran was comparing the ImageNet based ResNet with the Places365 based ResNet. In both cases, the Voronoi grid setup was used for 252 grid cells (size 0.01 coordinate). The results of this comparison can be seen below. Please note that `volcanic-sponge-18` is Places365 and`skilled-feather-20` is ImageNet.

![](https://i.imgur.com/itHiqFc.png)
![](https://i.imgur.com/2dYbLFg.png)


It can clearly be seen that the validation loss and accuracy are almost the same: however, the training loss and accuracy tell a different story. Here, it seems as if the ImageNet based model is better able continuously learn under data augmentation. This could be explained by the fact that the ImageNet-based model is pre-trained on almost twice as much data as the Places365-based model and is thus better able to keep extracting new information out of the training data.


### Directly predicting coordinates

In the next experiment, we tried to, instead of predicting the class label of the cell in which the photo was taken, directly predict the photo's latitude and longitude. In the table below, you can find the average distance error expressed in degrees after 15 epochs for the cells-based model and after one epoch for the coordinate based model.

| Cells | Coordinate |
| -------- | -------- |
| 0.063 degrees (6.99 km)    | 0.035 degrees (3.89 km)   |

Only one epoch was run for the coordinate-based model, as we were unable to improve the performance of this model after the first epoch: even after tuning the learning rate and playing with unlock factors, we did not manage to go lower.

However, it can be seen that when directly trying to find the coordinate where a picture was taken, a lower distance error can be achieved.

So although direct prediction leads to a lower average distance error, a possible reason not to use it might that it is harder to debug this system. For example, in the section "Interactive neighborhood similarity plot", it is very easy to see what the model 'is thinking' when using grid cells. So in the case that two locations are very similar and that the model is having a hard time distinguishing between the two, this can be easily recognized in the cell-based setup, whereas in the direct prediction setup, this is almost impossible to see.


### Comparison of grid division

To verify the influence on the results we ran the same experiment with both square grid cells and Voronoi grid cells. Here data augmentation was used, weights pretrained with ImageNet were used with 70% of the layers locked and the learning schedule as described in the training section was used. Note that there are 250 square cells whereas there are 252 Voronoi cells which gives the square cells a slight advantage.

![](https://i.imgur.com/41ucsXu.png)


In these experiments we see that the test accuracy for Voronoi cells are marginally better but the cross entropy loss for the square cells is better. Our weighted distance metric clearly favours the Voronoi cells which could be because the random cell centers of the Voronoi cells are better distributed.

When examining how many train images a single cell has it becomes clear how the square grid cells caused 7 cells to have no data points at all, and a number of cells only have a low amount of images. The Voronoi grid cells are still not perfect, however they are already a bit better with only a few cells with between 10 and 20 images.

<div style="text-align: center; margin:0; auto;">
    <div class="photos">
<!--         <img src="https://i.imgur.com/TFfHc02.png" height="auto" width="300"/> -->
<!--         <img class="photos" src="https://i.imgur.com/40cue5J.png"/ height="auto" width="300"/> -->      
        <img class="photos" src="https://i.imgur.com/Nairo9J.png"/ height="auto" width="350"/>
        <img class="photos" src="https://i.imgur.com/4cpKMDc.png"/ height="auto" width="350"/>
    </div>
</div>

### Interactive neighborhood similarity plot

In the iFrame below, you can hover over a certain grid cell to see which cells the model predicted for photos taken in the focused grid cell on average. Red indicates that the model is more confident that pictures from the selected cell were taken there, blue means the model is pretty confident the pictures taken in the focused cell were not taken in those cells.

<iframe width="100%" height="500" src="https://jluij.github.io/CV-GeoGuessr/output_visualization/" frameborder="0"></iframe>


### Interesting results

Our original hypothesis was that images within a neighbourhood have similar characteristics and the model would therefore be able to predict the image locations. From examining the average model predictions from grid cells certain grid cells are confused with each other insinuating that their images are visually similar. Below are some intersting example of similar neighbourhoods with explanations what could make them similar.

#### Thames
Images around the river Thames seem to be similar. This is of course because images near the water side have the common element of the Thames. However there are also streetview images taken on boats on top of the Thames which is a completely different environment from the streets and buildings on other images.
![](https://i.imgur.com/n3WEG96.jpg)

#### Parks
Images within Hyde Park and Battersea Park are visually similar
![](https://i.imgur.com/jof3Q44.png)


#### City center
Images within the city center enclosed by the A501 and A4202 seem to be simular

![](https://i.imgur.com/HNp6dMF.png)




### (Un)locking different amount of layers

For different lock factors we trained and evaluated the ImageNet-based ResNet-50 model. From this, it became clear that locking 70% (122/174) of the layers provided the best test accuracy. 

![](https://i.imgur.com/t9ZLYgr.png)

### Data augmentation vs. no data augmentation
We now compare the learning performance of the model with and without data augmentation. We will refer to the figure below in our analysis. In this figure the blue plot `devout-morning-26` is where the dataset is not augmented. The brown plot `different-flower-30` is where augmentation *is* employed.
<p align="center">
  <img src="https://i.imgur.com/xWPIzsg.png" />
</p>

Comparing the left and right image, it is clear that the brown plot represents an overfitting model. Indeed, its performance on the train dataset improves up to 100% accuracy but its performance on the test set reaches its limit rather quickly.  
The blue plot shows a more true representation of the model's capabilities. Although its ability to perform on the validation set is somewhat limited, we observe that its performance on the training set plateaus less severely.  
We can conclude that data augmentation is indeed essential for deep neural network training.


## Conclusion

- With a dataset limited to 10.000 images its more efficient to only unlock part of a pretrained model.
- Simularity of neighbourhoods can somewhat be infeard from the model predicitons.
- Its better to fintune a model with more images than more relevant images.

## Discussion/future work
### Approach
Upon reflection our approach deviates somewhat from what we initially set out to do. Currently, the model essentialy determines the similarity between different neighbourhoods. It however is not necessarily motivated to construct a latent space in which two nearby images that may be similar are placed together.
An approach more in line with our initial intentions would have been to approach the problem not as one of classification but of similarity. This especially came to mind as we learned more about the metric learning work and in converstations with professor Seyran Khademi.
Perhaps combined with the previous point, the graph-like structure of Streetview image locations could have been exploited in constructing the metric space. In e.g. a graph network representing the streets of London we would have not considered the Euclidian distance as a metric but Geodesic distance.
<!-- ^^ Mschn nog concreter omschrijven hoe we dit precies hadden aangepakt? -->


A more in-depth analysis of different data augmentations can be performed. An obvious augmentation we overlooked was horizontally flipping the images.


### Group Member Contributions
- *Sjoerd Groot: gridcell partitioning, (un)locking pretrained weights, WandB integration, interactive result widget, checkpoint saving/loading*
- *Douwe den Blanken: Places365, coordinate prediction, pipeline setup*
- *Joost Luijmes: Distance error metric, Data augmentation, Google Cloud setup*


### Reproducing our results

Our results are fully reproducible using [our code](https://github.com/JLuij/CV-GeoGuessr), which we make public in order to aid future work in this area. Please do note however, that you will have to 'bring' your own data, as Google Streetview images may not be stored and/or published.

### References
- [1] Müller-Budack, E., Pustu-Iren, K., Ewerth, R. (2018). Geolocation Estimation of Photos Using a Hierarchical Model and Scene Classification. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11216. Springer, Cham. https://doi.org/10.1007/978-3-030-01258-8_35

