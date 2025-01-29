# models/meta_learning_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd

class MetaLearningIDS:
    def __init__(self, dataset_paths):
        self.models = {}
        self.label_encoder = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.dataset_paths = dataset_paths
        self.input_shape = None
        self.num_classes = None

    def load_and_preprocess(self):
        """Load and preprocess multiple CIC datasets"""
        dfs = []
        for path in self.dataset_paths:
            df = pd.read_csv(path)
            dfs.append(self._clean_data(df))
        
        full_df = pd.concat(dfs, axis=0)
        X = self.scaler.fit_transform(full_df.drop('labels', axis=1))
        y = self.label_encoder.fit_transform(full_df['labels'])
        return train_test_split(X, y, test_size=0.2)

    def _clean_data(self, df):
        """Dataset-specific cleaning"""
        df = df.dropna()
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df['labels'] = df['Label'].apply(lambda x: [x] if x != 'BENIGN' else [])
        return df

    def build_model(self):
        """Meta-learning model architecture"""
        inputs = Input(shape=(self.input_shape,))
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, epochs=50):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        self.input_shape = X_train.shape[1]
        self.num_classes = y_train.shape[1]
        
        model = self.build_model()
        model.fit(X_train, y_train, 
                 validation_data=(X_test, y_test),
                 epochs=epochs,
                 batch_size=512)
        
        self.models['base'] = model

    def save_model(self, path):
        self.models['base'].save(path)