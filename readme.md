# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of Jupyter notebooks.

_Note: Data used in the projects is for demonstration purposes only._

## Instructions for Running Python Notebooks Locally
1. Install dependencies using requirements.txt.
```
pip -r requirements.txt
```
2. Run notebooks as usual by using a jupyter notebook server, Vscode etc.

## Contents
- ### Data Mining lab practices
    - preprocessing of data: based on various csv dataset files, practices focus on these tasks  (folder: data_preprocess)
        - handling missing values 
        - handling noisy data (outliers and duplicate data) 
        - data normalisation 
        - time series data aggregation
        - sampling
        - discretisation
        - principal component analysis

    - Data exploration and data visualisation: 
        - pre-processing features of the dataset
        - data summarization
        - data visualization: table, histograms, pie charts, box plots, scatter plots, distance matrices (folder: data_vis) 

    - Data Warehousing and On-line Analytical Processing (OLAP), more specifically with the concepts of data cubes, data cube measures, typical OLAP operations, and data cube computation (folder:data_warehouse)

    - Classification and clustering: based on the MNIST handwritten digits dataset creats the following models to predict the digit from 0 to 9 （folder:ml_knn_kmeans)
        - create a KNN classifier with gridsearchcv for 5-fold cross validation
        - creat a K-Means Clustering model

    -  Association Analysis: using a simple supermarket transaction records example to explore Apriori Algorithm, support threshold,  Kulczynski measure and imbalance ratio (folder:association_analysis)

    - Outlier Detection: (folder: outlier_detection)
        - using parametric methods - Mahalanobis distance to detect anomalous changes in the daily closing prices of various stocks based on a given stock csv file dataset. This approach assumes that the majority of the data instances are governed by some well-known probability distribution, e.g. a Gaussian distribution. Outliers can then be detected by seeking for observations that do not fit the overall distribution of the data.
        - using proximity-based approaches: use a KNN model, we apply the distance-based approach with k=4 to identify the anomalous trading days from the stock market data. 
        - using classification-based methods: here, we will work with a different dataset on house prices. Use an one-class SVM classifier to detect outliers. When modeling one class, the algorithm captures the density of the majority class and classifies examples on the extremes of the density function as outliers.
    
    - Web Scraping: working on data extraction from the web using Python's Beautiful Soup module, tasks including: (folder:web_scraping)
        - converting an html table into a Pandas dataframe
        - scraping URLs
    - Text Mining and Timeseries Mining: 
        - text mining: use tf-idf and k-means clustering algorithm to group Wikipedia articles into different cluter groups
        - timeseries mining: 
            - trailing moving average smoothing, 
            - forcast future data using Autoregressive (AR) Models, Moving Average (MA) Models, Autoregressive Moving Average (ARMA) Models and Autoregressive Integrated Moving Average (ARIMA) Models
            - Transform data using Discrete Fourier Transform (DFT). 

    _Tools: scikit-learn, Pandas, Seaborn, Matplotlib, Numpy, cubes, sqlalchemy, mlxtend, bs4, urllib, wikipedia, statsmodels_

- ### Machine Learning lab practices
    -  polynomial regression model experiment(folder: ml_regression): to obtain the least squares linear fit of a polynomial regression model,  explore ways to overcome overfitting problems by experimenting with the order of a polynomial model from 1 to 10, and adding regularisation parameter. 

    - classification experiment I (folder:ml_classification/lab05.ipynb): 
        - create explore linear classifier,  which produces decision regions separated by linear boundaries
        - create a logistic regression classifier, which uses the logistic function to create a notion of classifier "certainty"
        - optimisation with gradient descent
    - classification experiemnt II (folder:ml_classification/lab06.ipynb): use the well-known Iris dataset to build and analyse the performance of multi-class classifiers
        - KNN classifier: first consider a kNN classifier and analyse the impact of changing the value of k on the resulting decision regions
        - a logistic regression classifier:  We will then train a logistic regression classifier and will analyse its performance using the confusion matrix
        - a Bayesian approach with Gaussian class densities

- ### Projects:
    - Audio Machien Learning Classifier (folder: ml_audio_classifier): build a supervised Machine Learning classifier to classifier an audio segment belongs to a specific song label based on four features: power, pitch mean, pich standard deviation, voice frequency

    - Big Data Project on Ethereum Analysis：(folder:ethereum) please find the detailed project report here. (link of the report)
        - Background of the project: Ethereum is a blockchain based distributed computing platform where users may exchange currency (Ether), provide or purchase services (smart contracts), mint their own coinage (tokens), as well as other applications.Whilst you would normally need a CLI tool such as GETH to access the Ethereum blockchain, recent tools allow scraping all block/transactions and dump these to csv's to be processed in bulk; notably Ethereum-ETL. These dumps are uploaded daily into a repository on the university cluster at the HDFS folder /data/ethereum. We have used this source as the dataset for this project.
        - Write a set of Map/Reduce and Spark jobs to perform the following tasks:
            - aggregate the total number of transactions in each month over the whole period of the dataset
            - find the top 10 aggregated value for each address which appears in both Transaction dataset and Contract dataset
            - top ten most active miners    
            - popular scams
            - to explore the impact on the number and average value of transactions as well as the gas price of the transaction from the DAO fork event which happened on 20/07/2016
            - to explore how has the supply of gas changed over time

    
    - NLP CRF sequence tagging for Movie Queries: (folder:nlp_CRF) this project involves optimizing the performance of a Conditional Random Field (CRF) sequence tagger for movie trivia questions and answers data, which consist of instances of data of word sequences with the target classes/labels for each word in a BIO (Beginning, Inside, Outside) tagging format.

    - NLP Vector Space Semantics for Similarity between Eastenders Characters: (folder nlp_vector) this project involves creating a vector representation of a document containing lines spoken by a character in the Eastenders script data, then improving that representation such that each character vector is maximially distinguished from the other character documents. 

    - NLP Sexism Detection on social media using deep learning methods (nlp_sexism_detection folder): The objective of the project is to implement English sexism content (tweets and gabs) classification with only very limited dataset (train on 3000 texts) by using deep learning methods, and to explore ways to improve the prediction performance.Please find (ppt) for detail explaination of the project. 







