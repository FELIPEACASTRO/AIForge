import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class BrazilianDataPipeline:
    def __init__(self):
        self.preprocessor = None
        self.numerical_features = []
        self.categorical_features = []

    def _identify_features(self, df):
        # Identify numerical and categorical features, excluding 'isFraud' and 'id'
        self.numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        if 'isFraud' in self.numerical_features:
            self.numerical_features.remove('isFraud')
        
        self.categorical_features = df.select_dtypes(include='object').columns.tolist()
        if 'id' in self.categorical_features:
            self.categorical_features.remove('id')
        if 'timestamp' in self.categorical_features: # Assuming timestamp will be handled separately or dropped
            self.categorical_features.remove('timestamp')

    def create_preprocessing_pipeline(self, df):
        self._identify_features(df)

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor

    def process_data(self, df, target_column='isFraud'):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Create or fit preprocessor if not already fitted
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(X)
            self.preprocessor.fit(X)
        
        X_processed = self.preprocessor.transform(X)
        return X_processed, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == '__main__':
    import numpy as np
    from external_dataset_integration import ExternalDatasetIntegration

    # 1. Integrate datasets
    integration_system = ExternalDatasetIntegration()
    integrated_dfs = integration_system.get_integrated_data()

    if integrated_dfs:
        print('\nStarting data pipeline for integrated datasets...')
        
        # Process Brazilian synthetic data
        if 'brazilian_synthetic_fraud_data' in integrated_dfs:
            brazilian_df = integrated_dfs['brazilian_synthetic_fraud_data']
            print(f'\nProcessing Brazilian Synthetic Data (Shape: {brazilian_df.shape})...')
            
            pipeline = BrazilianDataPipeline()
            X_processed, y = pipeline.process_data(brazilian_df)
            
            print(f'Processed X shape: {X_processed.shape}')
            print(f'Processed y shape: {y.shape}')
            
            X_train, X_test, y_train, y_test = pipeline.split_data(X_processed, y)
            print(f'Train/Test split: {X_train.shape}/{X_test.shape}')
            print(f'Fraud in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)')
            print(f'Fraud in testing: {y_test.sum()} ({y_test.mean()*100:.2f}%)')

        # Process creditcard_kaggle data
        if 'creditcard_kaggle' in integrated_dfs:
            creditcard_df = integrated_dfs['creditcard_kaggle']
            print(f'\nProcessing Credit Card Kaggle Data (Shape: {creditcard_df.shape})...')
            
            pipeline_cc = BrazilianDataPipeline()
            X_processed_cc, y_cc = pipeline_cc.process_data(creditcard_df)
            
            print(f'Processed X shape: {X_processed_cc.shape}')
            print(f'Processed y shape: {y_cc.shape}')
            
            X_train_cc, X_test_cc, y_train_cc, y_test_cc = pipeline_cc.split_data(X_processed_cc, y_cc)
            print(f'Train/Test split: {X_train_cc.shape}/{X_test_cc.shape}')
            print(f'Fraud in training: {y_train_cc.sum()} ({y_train_cc.mean()*100:.2f}%)')
            print(f'Fraud in testing: {y_test_cc.sum()} ({y_test_cc.mean()*100:.2f}%)')

        # Skip paysim_kaggle for now due to memory constraints
        # if 'paysim_kaggle' in integrated_dfs:
        #     paysim_df = integrated_dfs['paysim_kaggle']
        #     print(f'\nProcessing PaySim Kaggle Data (Shape: {paysim_df.shape})...')
        #     pipeline_ps = BrazilianDataPipeline()
        #     X_processed_ps, y_ps = pipeline_ps.process_data(paysim_df)
        #     print(f'Processed X shape: {X_processed_ps.shape}')
        #     print(f'Processed y shape: {y_ps.shape}')
        #     X_train_ps, X_test_ps, y_train_ps, y_test_ps = pipeline_ps.split_data(X_processed_ps, y_ps)
        #     print(f'Train/Test split: {X_train_ps.shape}/{X_test_ps.shape}')
        #     print(f'Fraud in training: {y_train_ps.sum()} ({y_train_ps.mean()*100:.2f}%)')
        #     print(f'Fraud in testing: {y_test_ps.sum()} ({y_test_ps.mean()*100:.2f}%)')

    else:
        print('No integrated dataframes found to process.')

