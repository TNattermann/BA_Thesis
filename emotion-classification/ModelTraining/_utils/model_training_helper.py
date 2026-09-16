from ModelTraining.nn_training import train_nn
from ModelTraining.nn_sweep import Sweeper
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class TrainerUtils:
    def __init__(self, main_class_instance):
        self.instance = main_class_instance

    def load_cached_dataset(self):
        print('Loading dataset from local cache.')
        base_emotions = ['fear', 'joy', 'sadness', 'anger', 'no emotion']
        if self.instance.feature_set['pca'] or self.instance.dataset != 'GNE_SEM_GER' or self.instance.resample or self.instance.label_merge:
            print('Requirements for cache not met, loading new dataset.')
            return self.data_prep()
        X_train = pd.read_csv(f'dataset_cache/X_train_{self.instance.language}.csv')
        Y_train = pd.read_csv(f'dataset_cache/Y_train_{self.instance.language}.csv')
        df = pd.concat([X_train, Y_train], axis=1)
        if self.instance.base_emo:
            df = df[df['emotion'].isin(base_emotions)]
            X_train = df.drop(['emotion'], axis=1)
            Y_train = df[['emotion']]
        features = ''
        concat = pd.DataFrame()
        if self.instance.feature_set['ada']:
            features += 'ada&'
            concat = pd.concat([concat, X_train.iloc[:, :1536]], axis=1)
        if self.instance.feature_set['tf_idf']:
            features += f'tfidf_{self.instance.feature_set["n_tfidf"]}_dims&'
            concat = pd.concat([concat, X_train.iloc[:, 1536:2536]], axis=1)
        if self.instance.feature_set['linguistics']:
            features += f'linguistics&'
            concat = pd.concat([concat, X_train.iloc[:, 2536:2540]], axis=1)
        if self.instance.feature_set['nrc']:
            features += f'nrc&'
            concat = pd.concat([concat, X_train.iloc[:, 2540:]], axis=1)
        if features[-1:] == '&':
            features = features[:-1]
        X_train, X_val, Y_train, Y_val = train_test_split(concat, Y_train, test_size=0.2, random_state=42)
        return X_train, X_val, Y_train, Y_val, features

    @staticmethod
    def pca_decomp(data, components):
        print(f'Dimension reduced to {components}\n ------')
        pca = PCA()
        result = pca.fit_transform(data)
        exp_matrix = pca.explained_variance_ratio_[:components]
        var_exp = np.cumsum(exp_matrix)
        print(f'Explained Variance on train_set: {var_exp[-1]}')
        reduced = result[:, :components]
        return reduced

    @staticmethod
    def log_dataframe_to_html(df, file_path):
        html_table = df.to_html(index=False)
        with open(file_path, 'a') as html_file:
            if html_file.tell() == 0:
                html_file.write('<html><body>\n')
            html_file.write(html_table)

    def trigger_training(self):
        if self.instance.classifier == 'svm':
            print('Model: SVM\n ------')
            scaler_train = StandardScaler()
            scaler_val = StandardScaler()
            self.instance.X_train = scaler_train.fit_transform(self.instance.X_train)
            self.instance.X_val = scaler_val.fit_transform(self.instance.X_val)
            model = OneVsRestClassifier(SVC(probability=True, random_state=42, C=5, kernel='rbf'))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'rf':
            print('Model: RF\n ------')
            model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'gb':
            print('Model: Gradient Boosting\n ------')
            model = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'nn':
            if self.instance.nn_sweep:
                Sweeper(self.instance.X_train, self.instance.Y_train, self.instance.X_val, self.instance.Y_val, self.instance.configs)
            else:
                train_pred, Y_pred, train_proba, val_proba, classes = train_nn(self.instance.X_train, self.instance.Y_train, self.instance.X_val, self.instance.Y_val,
                                                                               self.instance.configs, self.instance.language, self.instance.model_id)
        else:
            if not self.instance.hypertuning:
                train_pred = model.predict(self.instance.X_train)
                train_proba = model.predict_proba(self.instance.X_train)
                Y_pred = model.predict(self.instance.X_val)
                val_proba = model.predict_proba(self.instance.X_val)
                classes = model.classes_
        if not self.instance.hypertuning:
            return train_pred, Y_pred, train_proba, val_proba, classes, model
        return model
