from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
import time

def simple_grid_search(x_train, y_train, x_test, y_test, feature_engineering_pipeline):
    ''' 
    simple helper function to grid search an ExtraTreesClassifier model and 
    print out a classification report for the best param set.
    Best here is defined as having the best cross-validated accuracy on the training set
    '''
    
    params = {  # some simple parameters to grid search
        'max_depth': [10, None],
        'n_estimators': [10, 50, 100, 500],
        'criterion': ['gini', 'entropy']
    }

    base_model = ExtraTreesClassifier()

    model_grid_search = GridSearchCV(base_model, param_grid=params, cv=3)
    start_time = time.time()  # capture the start time
    if feature_engineering_pipeline:  # fit FE pipeline to training data and use it to transform test data
        parsed_x_train = feature_engineering_pipeline.fit_transform(x_train, y_train)
        parsed_x_test = feature_engineering_pipeline.transform(x_test)
    else:
        parsed_x_train = x_train
        parsed_x_test = x_test

    parse_time = time.time()
    print(f"Parsing took {(parse_time - start_time):.2f} seconds")

    model_grid_search.fit(parsed_x_train, y_train)
    fit_time = time.time()
    print(f"Training took {(fit_time - start_time):.2f} seconds")

    best_model = model_grid_search.best_estimator_

    print(classification_report(y_true=y_test, y_pred=best_model.predict(parsed_x_test)))
    end_time = time.time()
    print(f"Overall took {(end_time - start_time):.2f} seconds")
    
    return best_model


    def advanced_grid_search(x_train, y_train, x_test, y_test, ml_pipeline, params, cv=3, include_probas=False, is_regression=False):
        ''' 
        This helper function will grid search a machine learning pipeline with feature engineering included
        and print out a classification report for the best param set. 
        Best here is defined as having the best cross-validated accuracy on the training set
        '''
        
        model_grid_search = GridSearchCV(ml_pipeline, param_grid=params, cv=cv, error_score=-1)
        start_time = time.time()  # capture the start time

        model_grid_search.fit(x_train, y_train)

        best_model = model_grid_search.best_estimator_
        
        y_preds = best_model.predict(x_test)
        
        if is_regression:
            rmse = np.sqrt(mean_squared_error(y_pred=y_preds, y_true=test_set['pct_change_eod']))
            print(f'RMSE: {rmse:.5f}')
        else:
            print(classification_report(y_true=y_test, y_pred=y_preds))
        print(f'Best params: {model_grid_search.best_params_}')
        end_time = time.time()
        print(f"Overall took {(end_time - start_time):.2f} seconds")
        
        if include_probas:
            y_probas = best_model.predict_proba(x_test).max(axis=1)
            return best_model, y_preds, y_probas
        
        return best_model, y_preds

def distplot_features(df, feature, title, color = custom_colors[4], categorical=True):
    '''Takes a column from the dataframe and plots the distribution (after count)'''

    if categorical: values = df[feature].value_counts().values
    else: values = df[feature].values

    print('Mean: {:,}'.format(np.mean(values)), "\n"
          'Median: {:,}'.format(np.median(values)), "\n"
          'Max: {:,}'.format(np.max(values)))

    plt.figure(figsize = (18, 3))

    if categorical: sns.distplot(values, hist=False, color = color, kde_kws = {'lw':3})
    else:
        # To speed up the process
        if len(values) > 1000000: sns.distplot(values[::250000], hist=False, color = color, kde_kws = {'lw':3})
        else: sns.distplot(values, hist=False, color = color, kde_kws = {'lw':3})

    plt.title(title, fontsize=15)
    plt.show()    
    del values
    gc.collect()        