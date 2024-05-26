This is the first commit of the readme


Stages of project will be:
1. Prepare dataset by merging all the training data csv files
2. Prepare the pipelines
3. Define the metrics used to analyze the data (MSE)
4. Define a few parameters for finding the best hyperparameters (for KNN, RandomForset, MLP, SVM)
5. Do the pipelines and the gridsearchCV with TimeSeriesSplit as cv
6. For each "best" estimator model with the best hyperparameters, run the estimators on the final test set (last week of april)
7. Plot stuff up

DONE

reqs: scikit-learn matplotlib pandas