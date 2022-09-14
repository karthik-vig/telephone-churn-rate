import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class DataframeToTensor(Dataset):
    def __init__(self, x_data_df, y_data_df):
        self.x_data_df = x_data_df
        self.y_data_df = y_data_df

    def __len__(self):
        return len(self.x_data_df)

    def __getitem__(self, index):
        return torch.tensor([self.x_data_df[index]], dtype=torch.double), \
               torch.tensor([self.y_data_df[index]], dtype=torch.double)

class DataframeToTensorForPredict(Dataset):
    def __init__(self, x_data_df):
        self.x_data_df = x_data_df

    def __len__(self):
        return len(self.x_data_df)

    def __getitem__(self, index):
        return torch.tensor([self.x_data_df[index]], dtype=torch.double)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 3, 2)
        self.layer1 = nn.Linear(120, 80)
        self.layer2 = nn.Linear(80, 50)
        self.layer3 = nn.Linear(50, 30)
        self.layer4 = nn.Linear(30, 15)
        self.layer5 = nn.Linear(15, 10)
        self.layer6 = nn.Linear(10, 5)
        self.layer7 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, x_data_df):
        tensor_dataset = DataframeToTensorForPredict(x_data_df=x_data_df)
        dataloader = DataLoader(tensor_dataset, batch_size=len(x_data_df))
        for x in dataloader:
            x = x.to('cuda')
            x = self.forward(x.double()).cpu().detach().numpy()
            x = np.where( x > 0.5, 1, 0)
            x = x.reshape(len(x_data_df))
            return x


def train_cnn(x_data_df, y_data_df, x_test_df, y_test_df):
    cnn_model = CNN().to('cuda')
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)
    loss_fn = nn.BCELoss()
    tensor_dataset = DataframeToTensor(x_data_df=x_data_df,
                                       y_data_df=y_data_df)
    dataloader = DataLoader(tensor_dataset, batch_size=256, shuffle=True)
    test_acc_list = []
    cnn_model = cnn_model.double()
    for epoch in range(1, 50):
        avg_train_loss = 0
        for batch, label in dataloader:
            batch = batch.to('cuda')
            label = label.to('cuda')
            output = cnn_model(batch.double())
            optimizer.zero_grad()
            loss = loss_fn(output, label)
            avg_train_loss += loss
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch, ' Done!')
        print('Average training loss: ', avg_train_loss / len(x_data_df))
        cnn_preds = cnn_model.predict(x_test_df)
        test_acc = accuracy_score(y_test_df, cnn_preds)
        test_acc_list.append(test_acc)
    torch.save(cnn_model, 'cnn_model.pt')
    cnn_preds = cnn_model.predict(x_test_df)
    print('CNN model accuracy: ', accuracy_score(y_test_df, cnn_preds))
    plt.close()
    plt.plot(list(range(1, 50)), test_acc_list)
    plt.show()
    return cnn_model


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


def get_user_input(test_mode=False, **kwargs):
    user_input = {'compute_tsne': None,
                  'load_tsne': None,
                  'compute_model': None,
                  'select_model': None}
    if test_mode:
        for key, value in kwargs.items():
            user_input[key] = value
        return user_input
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
    select_model_option = int(input('Select a ML model = (1) SVM, (2) Logistic Regression, (3) CNN: '))
    if select_model_option == 1:
        user_input['select_model'] = 'SVM'
    elif select_model_option == 2:
        user_input['select_model'] = 'log'
    else:
        user_input['select_model'] = 'cnn'

    return user_input


def main():
    # user_input = get_user_input(test_mode=True,
    #                            compute_tsne=False,
    #                            compute_model=True,
    #                            select_model='cnn')
    user_input = get_user_input()
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
    elif user_input['select_model'] == 'cnn':
        if user_input['compute_model']:
            model = train_cnn(x_data_df=x_train_df,
                              y_data_df=y_train_df,
                              x_test_df=x_val_df,
                              y_test_df=y_val_df)
        else:
            model = torch.load('cnn_model.pt')

    test_df = get_data_df(datasplit_type='test')
    test_df = dataset_process_pipeline(data_df=test_df)
    model_preds = model.predict(test_df.to_numpy())
    sol_df = pd.DataFrame({'id': np.arange(69999, 99999),
                           'churn_probability': model_preds})
    if user_input['select_model'] == 'SVM':
        sol_df.to_csv('svm_solution.csv')
    elif user_input['select_model'] == 'log':
        sol_df.to_csv('logreg_solution.csv')
    elif user_input['select_model'] == 'cnn':
        sol_df.to_csv('cnn_solution.csv')


if __name__ == '__main__':
    main()
