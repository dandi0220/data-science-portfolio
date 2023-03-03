# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of Jupyter notebooks or .py files.

_Note: Data used in the projects is for demonstration purposes only._

## Instructions for Running Python Notebooks Locally
1. Install dependencies using requirements.txt.
```
pip -r requirements.txt
```
2. Run notebooks as usual by using a jupyter notebook server, Vscode etc.


## Table of Content 

* [Data Mining lab practices](#data-mining-lab-practices)
    * [Preprocessing of data](#preprocessing-of-data-based-on-various-csv-dataset-files-practices-focus-on-these-tasks)
    * [Data exploration and data visualisatio](#data-exploration-and-data-visualisation)
    * [Data Warehousing and On-line Analytical Processing (OLAP)](#data-warehousing-and-on-line-analytical-processing-olap-more-specifically-with-the-concepts-of-data-cubes-data-cube-measures-typical-olap-operations-and-data-cube-computation)
    * [Classification and Clustering](#classification-and-clustering-based-on-the-mnist-handwritten-digits-dataset-creats-the-following-models-to-predict-the-digit-from-0-to-9)
    * [Association Analysis](#association-analysis-using-a-simple-supermarket-transaction-records-example-to-explore-apriori-algorithm-support-threshold-kulczynski-measure-and-imbalance-ratio)
    * [Outlier Detection](#outlier-detection)
    * [Web Scraping](#web-scraping-working-on-data-extraction-from-the-web-using-pythons-beautiful-soup-module-tasks-including)
    * [Text Mining and Timeseries Mining](#text-mining-and-timeseries-mining)

* [Machine Learning lab practices](#machine-learning-lab-practices)
    * [Polynomial Regression Model Experiment](#polynomial-regression-model-experiment)
    * [Classification Experiment I](#classification-experiment-i)
    * [Classification Experiment II](#classification-experiemnt-ii-use-the-well-known-iris-dataset-to-build-and-analyse-the-performance-of-multi-class-classifiers)

* [Projects](#projects)
    * [Audio Machine Learning Classifier](#audio-machine-learning-classifier-build-a-supervised-machine-learning-classifier-to-classify-an-audio-segment-belongs-to-a-specific-song-based-on-four-input-features-power-pitch-mean-pich-standard-deviation-and-voice-frequency)
    * [MLOPS project on Wine Quality Prediction](#mlops-project-on-wine-quality-prediction-build-end-to-end-mlops-pipelines-for-a-simple-wine-quality-prediction-website-application-trained-base-on-elastic-net-algorithm-key-features-include)
    * [Big Data Project on Ethereum Analysis](#big-data-project-on-ethereum-analysis-please-find-the-detailed-project-report-here)
    * [NLP CRF sequence tagging for Movie Queries](#nlp-crf-sequence-tagging-for-movie-queries-this-project-involves-optimizing-the-performance-of-a-conditional-random-field-crf-sequence-tagger-for-movie-trivia-questions-and-answers-data-which-consist-of-instances-of-data-of-word-sequences-with-the-target-classeslabels-for-each-word-in-a-bio-beginning-inside-outside-tagging-format)
    * [NLP Vector Space Semantics for Similarity between Eastenders Characters](#nlp-vector-space-semantics-for-similarity-between-eastenders-characters-this-project-involves-creating-a-vector-representation-of-a-document-containing-lines-spoken-by-a-character-in-the-eastenders-script-data-then-improving-that-representation-such-that-each-character-vector-is-maximially-distinguished-from-the-other-character-documents)
    * [NLP Sexism Detection on social media using deep learning methods](#nlp-sexism-detection-on-social-media-using-deep-learning-methods-the-objective-of-the-project-is-to-implement-english-sexism-content-from-social-media-dataset-tweets-and-gabs-classification-by-using-deep-learning-methods-such-as-cnn-bert-and-lstm-and-to-explore-ways-to-improve-the-prediction-performance-please-find-the-slides-for-more-details-of-the-project)

## Contents


- ### Data Mining lab practices
    - ### [Preprocessing of data](https://github.com/dandi0220/data-science-portfolio/blob/main/data_preprocess/Lab02_V2.ipynb): based on various csv dataset files, practices focus on these tasks
        - handling missing values 
        - handling noisy data (outliers and duplicate data) 
        - data normalisation 
        - time series data aggregation
        - sampling
        - discretisation
        - principal component analysis

    - ### [Data exploration and data visualisation](https://github.com/dandi0220/data-science-portfolio/blob/main/data_vis/Lab03.ipynb) 
        - pre-processing features of the dataset
        - data summarization
        - data visualization: table, histograms, pie charts, box plots, scatter plots, distance matrices

    - ### [Data Warehousing and On-line Analytical Processing (OLAP)](https://github.com/dandi0220/data-science-portfolio/blob/main/data_warehouse/Lab04.ipynb): more specifically with the concepts of data cubes, data cube measures, typical OLAP operations, and data cube computation

    - ### [Classification and Clustering](https://github.com/dandi0220/data-science-portfolio/blob/main/ml_knn_kmeans/Lab05.ipynb): based on the MNIST handwritten digits dataset creats the following models to predict the digit from 0 to 9
        - create a KNN classifier with gridsearchcv for 5-fold cross validation
        - creat a K-Means Clustering model

    - ### [Association Analysis](https://github.com/dandi0220/data-science-portfolio/blob/main/association_analysis/association_analysis.ipynb): using a simple supermarket transaction records example to explore Apriori Algorithm, support threshold,  Kulczynski measure and imbalance ratio

    - ### [Outlier Detection](https://github.com/dandi0220/data-science-portfolio/blob/main/outlier_detection/Lab07.ipynb)
        - using parametric methods - Mahalanobis distance to detect anomalous changes in the daily closing prices of various stocks based on a given stock csv file dataset. This approach assumes that the majority of the data instances are governed by some well-known probability distribution, e.g. a Gaussian distribution. Outliers can then be detected by seeking for observations that do not fit the overall distribution of the data.
        - using proximity-based approaches: use a KNN model, we apply the distance-based approach with k=4 to identify the anomalous trading days from the stock market data. 
        - using classification-based methods: here, we will work with a different dataset on house prices. Use an one-class SVM classifier to detect outliers. When modeling one class, the algorithm captures the density of the majority class and classifies examples on the extremes of the density function as outliers.
    
    - ### [Web Scraping](https://github.com/dandi0220/data-science-portfolio/blob/main/web_scraping/Lab08.ipynb): working on data extraction from the web using Python's Beautiful Soup module, tasks including:
        - converting an html table into a Pandas dataframe
        - scraping URLs

    - ### [Text Mining and Timeseries Mining](https://github.com/dandi0220/data-science-portfolio/blob/main/time_series/Lab09.ipynb) 
        - text mining: use tf-idf and k-means clustering algorithm to group Wikipedia articles into different cluter groups
        - timeseries mining: 
            - trailing moving average smoothing, 
            - forcast future data using Autoregressive (AR) Models, Moving Average (MA) Models, Autoregressive Moving Average (ARMA) Models and Autoregressive Integrated Moving Average (ARIMA) Models
            - Transform data using Discrete Fourier Transform (DFT). 

 _Tools: scikit-learn, Pandas, Seaborn, Matplotlib, Numpy, cubes, sqlalchemy, mlxtend, bs4, urllib, wikipedia, statsmodels_

- ### Machine Learning lab practices
    - ### [Polynomial Regression Model Experiment](https://github.com/dandi0220/data-science-portfolio/blob/main/ml_regression/Lab04.ipynb) 
        - to obtain the least squares linear fit of a polynomial regression model
        - explore ways to overcome overfitting problems by experimenting with the order of a polynomial model from 1 to 10, and adding regularisation parameter

    - ### [Classification Experiment I](https://github.com/dandi0220/data-science-portfolio/blob/main/ml_classification/Lab05.ipynb) 
        - create and explore linear classifier,  which produces decision regions separated by linear boundaries
        - create a logistic regression classifier, which uses the logistic function to create a notion of classifier "certainty"
        - optimisation with gradient descent
    - ### [Classification Experiemnt II](https://github.com/dandi0220/data-science-portfolio/blob/main/ml_classification/Lab06.ipynb): use the well-known Iris dataset to build and analyse the performance of multi-class classifiers
        - KNN classifier: first consider a kNN classifier and analyse the impact of changing the value of k on the resulting decision regions
        - logistic regression classifier:  We will then train a logistic regression classifier and will analyse its performance using the confusion matrix
        - Bayesian approach with Gaussian class densities

_Tools: scikit-learn, Pandas, Seaborn, Matplotlib, Numpy_

- ### Projects:
    - ### [Audio Machine Learning Classifier](https://github.com/dandi0220/data-science-portfolio/tree/main/ml_audio_classifier): build a supervised Machine Learning classifier to classify an audio segment belongs to a specific song based on four input features: power, pitch mean, pich standard deviation and voice frequency

    - ### [MLOPS project on Wine Quality Prediction](https://github.com/dandi0220/-simple-dvc-demo): build end to end MLOPS pipelines for a simple wine quality prediction website application trained base on Elastic Net algorithm. Key features include:
        - Git and DVC for version control
        - Pytest and Tox for creating virtual environment to standardize the testing of the project
        - MLflow for automated model selection
        - flask framework for the web application
        - CI/CD pipline with GitHub Actions

    - ### [Big Data Project on Ethereum Analysis](https://github.com/dandi0220/data-science-portfolio/tree/main/ethereum): please find the detailed project report [here](https://github.com/dandi0220/data-science-portfolio/blob/main/ethereum/Project%20Report.pdf).

        - Background of the project: Ethereum is a blockchain based distributed computing platform where users may exchange currency (Ether), provide or purchase services (smart contracts), mint their own coinage (tokens), as well as other applications. Whilst you would normally need a CLI tool such as GETH to access the Ethereum blockchain, recent tools allow scraping all block/transactions and dump these to csv's to be processed in bulk; notably Ethereum-ETL. These dumps are uploaded daily into a repository on the university cluster at the HDFS folder /data/ethereum. We have used this source as the dataset for this project.
        - Write a set of Map/Reduce and Spark jobs to perform the following tasks:
            - aggregate the total number of transactions in each month over the whole period of the dataset
            - find the top 10 aggregated value for each address which appears in both Transaction dataset and Contract dataset
            - top ten most active miners    
            - popular scams
            - to explore the impact on the number and average value of transactions as well as the gas price of the transaction from the DAO fork event which happened on 20/07/2016
            - to explore how has the supply of gas changed over time
    
    - ### [NLP CRF sequence tagging for Movie Queries](https://github.com/dandi0220/data-science-portfolio/blob/main/nlp_CRF/CRF_tagging_in_Movie_Queries_FINAL.ipynb): this project involves optimizing the performance of a Conditional Random Field (CRF) sequence tagger for movie trivia questions and answers data, which consist of instances of data of word sequences with the target classes/labels for each word in a BIO (Beginning, Inside, Outside) tagging format.

    - ### [NLP Vector Space Semantics for Similarity between Eastenders Characters](https://github.com/dandi0220/data-science-portfolio/blob/main/nlp_vector/NLP_distributional_semantics_FINAL.ipynb): this project involves creating a vector representation of a document containing lines spoken by a character in the Eastenders script data, then improving that representation such that each character vector is maximially distinguished from the other character documents. 

    - ### [NLP Sexism Detection on social media using deep learning methods](https://github.com/dandi0220/data-science-portfolio/tree/main/nlp_sexism_detection): The objective of the project is to implement English sexism content from social media dataset (tweets and gabs) classification by using deep learning methods such as CNN, BERT and LSTM, and to explore ways to improve the prediction performance. Please find the [slides](https://github.com/dandi0220/data-science-portfolio/blob/main/nlp_sexism_detection/Project_presentation.pdf) for more details of the project. 

If you like what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at dandi0220@gmail.com or connect to me via [LinkedIn](https://www.linkedin.com/in/dandi-yu/) :)







