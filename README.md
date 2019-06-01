# Audit-Risk-Prediction-for-a-Firm

> Building a Predictive model that predicts the Audit risk for a financial firm which inturn tells us what are the chances of a firm being fraudulent.

## Table of contents
* [General info](#general-info)
* [Technologies and Tools](#technologies-and-tools)
* [Code Examples](#code-examples)
* [Contact](#contact)

## General info
We have a dataset of 700+ Financial firms and we are trying to find out the best predictive model to predict the Audit Risk for a firm.
In this process, we have preprocessed and cleaned the data, and then applied various regression models like KNN, LinearSVM, Kernelized SVM, Ridge, Lasso, Stochastic Gradient Regressor, Polynomial Regression, Linear Regression, Decision Tree, and Random Forest to fit the data.

We have found best parameters for each model using Grid Search Cross Validation and at the end compared all the models to find the best one out of all.

In second phase of this project we have used Ensemble models and Principal Component Analysis

## Technologies and Tools
* Python 
* mglearn
* Graphviz

## Code Examples

````
# Visualizing how each feature converges with the increase in Regularization parameter alpha in Ridge Regression

import numpy as np

x_range1 = np.linspace(0.001, 1, 100).reshape(-1,1)
x_range2 = np.linspace(1, 200, 10000).reshape(-1,1)

x_range = np.append(x_range1, x_range2)

coeff = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_train,y_train)
    coeff.append(ridge.coef_ )
    
coeff = np.array(coeff)
col=X.columns.values
for i in range(0,17):
    plt.plot(x_range, coeff[:,i], label = '{}'.format(col[i]))

plt.axhline(y=0, xmin=0.001, xmax=9999, linewidth=1, c ='gray')
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7),
          ncol=5, fancybox=True, shadow=True)
plt.show()
````
````

# Using Grid Search to find the best parameters for kernelized SVM

svr = SVR()

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','poly','linear','sigmoid'],'gamma':[0.001, 0.01, 0.1, 1, 10, 100],
      'C':[0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(svr,parameters,cv=10,return_train_score=True)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('Best Accuracy is {}'.format(best_accuracy))
print('Best Parameters {}'.format(best_parameters))
````
````
# Plotting Feature Importances as given by Decision Tree

tree=DecisionTreeRegressor(min_samples_split=2)

parameters={'max_depth':[10,20,50,100,150,200],'max_leaf_nodes':[30,100,200,400,500,700]}

grid_search = GridSearchCV(tree,parameters,cv=10,return_train_score=True)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(6, 6)
plt.figure(figsize=(10,10))
mglearn.tools.heatmap(scores, xlabel='max_depth', xticklabels=parameters['max_depth'], ylabel='max_leaf_nodes', yticklabels=parameters['max_leaf_nodes'], cmap="viridis")

def plot_feature_importances(model):
    plt.figure(figsize=(8,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, height=0.5,align='center')
    plt.yticks(np.arange(n_features), cols)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(tree)

````

## Contact
Created by me with my teammate [Manish Shukla](https://github.com/Manishms18)

If you loved what you read here and feel like we can collaborate to produce some exciting stuff, or if you
just want to shoot a question, please feel free to connect with me on 
<a href="mailto:nick22910@gmail.com">email</a> or 
<a href="https://www.linkedin.com/in/ashishsharma1993/" target="_blank">LinkedIn</a>
