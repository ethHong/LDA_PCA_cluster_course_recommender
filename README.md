# LDA_PCA_cluster_course_recommender
LDA and PCA Clustering combined course recommender

This project tries to apply PCA cluster with course description, but basically identical to :https://github.com/ethHong/LDA_course_recommender

## Data Description
* Data are collected from Yonsei University, UIC courses
* Linkedin Data had been colleced manually and using selenium, anaymously

## Projecy Description
* Using LDA and Jensen Shannen Divergence, this projects tries to conpute disatnce between each courses and company, based on course description, and skillsets owned by workers in the company. Unlike https://github.com/ethHong/LDA_course_recommender, this one clsuters courses and find the group of courses relevant to specific company, or job role

## This is very initial prototype of the model, and attempts to improve this project in progress by:
* Re-collecting data by 'job description' (ex. 'Marketing'), instead of by specific company
* Trying to examine meta-learning

## How to Use
* First need to go through data processing with 'process_data.py'. It uses cleansing, stopword filtering and filtering irrelevant words by using TF-IDF scores. You could adjust parameters of how strictly filter irrelevant words based on TF-IDF Score

* Run modeling by 'main_modeling.py'. With perplexity and Coherence, put optimal topic number
* Also, with scree plot it asks to put optimal number for PCA
