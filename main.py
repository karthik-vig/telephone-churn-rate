import os

import numpy as np
import pandas as pd
import plotly.express as px
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_data_df(datasplit_type):
    data_path = os.path.join('data', datasplit_type)
    data_df = pd.read_csv(data_path + '.csv')
    return data_df


def get_data_properties(data_df):
    print('First few rows are: ', data_df.head(5))
    print('The columns are: ', data_df.columns)
    print('unique values in each row are: ')
    for col in data_df.columns:
        print(col, ': ', len(data_df[col].unique()), ' DataType: ', data_df[col].dtypes
              )
    print('Check for class imbalance: ')
    num_pos_class = np.count_nonzero(data_df['churn_probability'])
    num_neg_class = len(data_df['churn_probability']) - num_pos_class
    print('Number of elements in postive class: ', num_pos_class)
    print('Number of elements in negative class: ', num_neg_class)
    print('Number of columns: ', len(data_df.columns))


def impute_missing_vals(data_df, cat_col):
    for col in cat_col:
        most_re_val = data_df[col].value_counts().idxmax()
        data_df[col] = data_df[col].fillna(most_re_val)
    data_df = data_df.fillna(data_df.mean())
    return data_df


def create_days_feat(data_df, date_col, prefix=''):
    for col in date_col:
        data_df[col] = pd.to_datetime(data_df[col])
    pre_col = data_df[date_col[0]]
    for index, col in enumerate(date_col[1:]):
        data_df[prefix + 'diff_' + str(index)] = data_df[col] - pre_col
        data_df[prefix + 'diff_' + str(index)] = data_df[prefix + 'diff_' + str(index)].dt.days
        pre_col = data_df[col]
    data_df = data_df.drop(columns=date_col)
    return data_df


def tsne_plot(data_df, load_tnse=False):
    tsne_val_path = 'tsne_vals.npy'
    if load_tnse:
        tsne_plot_vals = np.load(tsne_val_path)
    else:
        x_data_df = data_df.drop(columns=['churn_probability'])
        tsne_plot_vals = TSNE(n_components=3,
                              learning_rate='auto',
                              init='random',
                              perplexity=50).fit_transform(x_data_df)

        np.save(tsne_val_path, tsne_plot_vals)
    print('tsne values are ready')
    tsne_df = pd.DataFrame({'ax1': tsne_plot_vals[:, 0], 'ax2': tsne_plot_vals[:, 1], 'ax3': tsne_plot_vals[:, 2],
                            'churn_probability': data_df['churn_probability']})
    print(tsne_df)
    fig = px.scatter_3d(tsne_df,
                        x='ax1', y='ax2', z='ax3', color='churn_probability')
    fig.show()


def model_classfication(model, x_train_df, y_train_df, x_test_df, y_test_df):
    model.fit(x_train_df, y_train_df)
    model_preds = model.predict(x_test_df)
    print(accuracy_score(y_test_df, model_preds))
    return model


def svm_classification(x_train_df, y_train_df, x_test_df, y_test_df):
    svm_model = svm.SVC(kernel='rbf')
    svm_model = model_classfication(model=svm_model,
                                    x_train_df=x_train_df,
                                    y_train_df=y_train_df,
                                    x_test_df=x_test_df,
                                    y_test_df=y_test_df)
    dump(svm_model, 'svm_model.joblib')
    return svm_model


def logreg_classification(x_train_df, y_train_df, x_test_df, y_test_df):
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model = model_classfication(model=logreg_model,
                                       x_train_df=x_train_df,
                                       y_train_df=y_train_df,
                                       x_test_df=x_test_df,
                                       y_test_df=y_test_df)
    dump(logreg_model, 'logreg_model.joblib')
    return logreg_model


def split_dataset(data_df):
    y_data_df = data_df['churn_probability']
    x_data_df = data_df.drop(columns=['churn_probability'])
    x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(x_data_df.to_numpy(),
                                                                  y_data_df.to_numpy(),
                                                                  test_size=0.2,
                                                                  random_state=200
                                                                  )
    return x_train_df, y_train_df, x_val_df, y_val_df


def select_model(model_choice):
    model_options = (svm_classification, logreg_classification)
    return model_options[model_choice]


def dataset_process_pipeline(data_df):
    data_df = data_df.drop(
        columns=['id', 'circle_id', 'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8'])
    data_df = create_days_feat(data_df=data_df,
                               date_col=['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8'],
                               prefix='rech_')
    data_df = create_days_feat(data_df=data_df, date_col=['date_of_last_rech_data_6', 'date_of_last_rech_data_7',
                                                          'date_of_last_rech_data_8'], prefix='rech_data_')
    data_df = impute_missing_vals(data_df=data_df, cat_col=['fb_user_6', 'fb_user_7', 'fb_user_8'])
    return data_df


def get_use_input():
    user_input = {}
    compute_tsne_val = int(input('Show t-SNE values? = (1) yes, (2) no: '))
    if compute_tsne_val == 1:
        user_input['compute_tsne'] = True
    else:
        user_input['compute_tsne'] = False
    if user_input['compute_tsne']:
        load_tsne_option = int(input('Load existing t-SNE values? = (1) yes, (2) no: '))
        if load_tsne_option == 1:
            user_input['load_tsne'] = True
        else:
            user_input['load_tsne'] = False
    compute_model_option = int(input('Compute a new ML model? = (1) yes, (2) no: '))
    if compute_model_option == 1:
        user_input['compute_model'] = True
    else:
        user_input['compute_model'] = False
    select_model_option = int(input('Select a ML model = (1) SVM, (2) Logistic Regression: '))
    if select_model_option == 1:
        user_input['select_model'] = 'SVM'
    else:
        user_input['select_model'] = 'log'
    return user_input


def main():
    user_input = get_use_input()
    # process the training dataset and also separate it into training and validation dataset.
    train_df = get_data_df(datasplit_type='train')
    get_data_properties(data_df=train_df)
    train_df = dataset_process_pipeline(data_df=train_df)
    x_train_df, y_train_df, x_val_df, y_val_df = split_dataset(data_df=train_df)

    if user_input['compute_tsne']:
        tsne_plot(data_df=train_df, load_tnse=user_input['load_tsne'])

    if user_input['select_model'] == 'SVM':
        if user_input['compute_model']:
            model = svm_classification(x_train_df=x_train_df,
                                       y_train_df=y_train_df,
                                       x_test_df=x_val_df,
                                       y_test_df=y_val_df)
        else:
            model = load('svm_model.joblib')
    elif user_input['select_model'] == 'log':
        if user_input['compute_model']:
            model = logreg_classification(x_train_df=x_train_df,
                                          y_train_df=y_train_df,
                                          x_test_df=x_val_df,
                                          y_test_df=y_val_df)
        else:
            model = load('logreg_model.joblib')

    test_df = get_data_df(datasplit_type='test')
    test_df = dataset_process_pipeline(data_df=test_df)
    model_preds = model.predict(test_df.to_numpy())
    sol_df = pd.DataFrame({'id': np.arange(69999, 99999),
                           'churn_probability': model_preds})
    sol_df.to_csv('solution.csv')


if __name__ == '__main__':
    main()
