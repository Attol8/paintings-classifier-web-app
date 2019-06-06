![paintings-classifier](https://c.pxhere.com/photos/c4/0e/people_museum_bench_couple_paintings_crocker-394893.jpg!d)

<!-- [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com) -->

# Paintings Classifier Web App

A clean Web app deployed on GCP using fastai to classify paintings styles 

Click <a href="https://rare-habitat-240317.appspot.com/">here</a> if you want to try the app <span>&#128072;</span> 

## Table of Contents 

- [Model Training](#Model-Training)
- [Methods Used](#Methods-used)
- [Production](#production)

---

### Model Training
The model has been trained on ... paintings using the <a href="https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset" target="_blank">Wiki Art Dataset</a>. The model beat the state-of-the-art result that was obtained by the creators of the dataset, which can be found in their academic paper <a href = "http://web.fsktm.um.edu.my/~cschan/doc/ICIP2016.pdf">Ceci n’est pas une pipe: A Deep Convolutional Network for Fine-art Paintings Classification</a> by WR Tan et al.(2017).

this model reached  accuracy(%) in style classification while WR tan et al. CNN-finetune model reached 54.50%

### Methods Used

* Machine Learning
* Data Visualization
* Predictive Modeling
* Deep Learning
* Image recognition
* Data Mining

### Technologies 
* Python
* Pandas
* Jupyter
* Google Cloud Storage
* Fastai Library

- Clone this repo to your local machine using `https://github.com/fvcproductions/SOMEREPO`

### Production

The model has been put into production by deploying a web app on GCP. The following <a href="https://course.fast.ai/deployment_google_app_engine.html">guide</a> has been used. 
