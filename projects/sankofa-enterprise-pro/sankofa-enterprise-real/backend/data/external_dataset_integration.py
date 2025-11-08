import pandas as pd
import os
import zipfile

class ExternalDatasetIntegration:
    def __init__(self, base_path='/home/ubuntu/sankofa-enterprise-real/data/external_datasets'):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _unzip_file(self, zip_path, extract_to_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f'Unzipped {zip_path} to {extract_to_path}')

    def load_and_preprocess(self, dataset_name, file_name, target_column=None):
        """Loads a dataset from a file and preprocesses it."""
        file_path = os.path.join(self.base_path, file_name)
        
        if not os.path.exists(file_path):
            print(f'Error: {file_path} not found. Please ensure the dataset is in the correct directory.')
            return None

        if file_name.endswith('.zip'):
            extract_path = os.path.join(self.base_path, dataset_name)
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            self._unzip_file(file_path, extract_path)
            
            if dataset_name == 'paysim_kaggle':
                csv_file_in_zip = 'PS_20174392719_1491204439457_log.csv'
            else:
                csv_files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]
                if not csv_files:
                    print(f'Error: No CSV file found in {extract_path}')
                    return None
                csv_file_in_zip = csv_files[0] 

            file_path = os.path.join(extract_path, csv_file_in_zip) 

        print(f'Loading and preprocessing {dataset_name} from {file_path}...')
        df = pd.read_csv(file_path)
        
        if dataset_name == 'creditcard_kaggle':
            df = df.rename(columns={'Class': 'isFraud'}) 
            if target_column and target_column != 'isFraud':
                df = df.rename(columns={'isFraud': target_column})
        
        elif dataset_name == 'paysim_kaggle':
            if target_column and target_column != 'isFraud':
                df = df.rename(columns={'isFraud': target_column})
        
        elif dataset_name == 'brazilian_synthetic_fraud_data':
            # Assuming the synthetic data already has an 'isFraud' column
            if target_column and target_column != 'isFraud':
                df = df.rename(columns={'isFraud': target_column})

        print(f'{dataset_name} preprocessed successfully.')
        return df

    def get_integrated_data(self, target_column='isFraud'):
        """Returns a dictionary of dataframes for all integrated datasets."""
        datasets_info = [
            {'name': 'creditcard_kaggle', 'file': 'creditcard.zip', 'target_column': target_column},
            # {'name': 'paysim_kaggle', 'file': 'paysim.zip', 'target_column': target_column},
            {'name': 'brazilian_synthetic_fraud_data', 'file': 'brazilian_synthetic_fraud_data.csv', 'target_column': target_column},
        ]
        
        integrated_data = {}
        for info in datasets_info:
            data = self.load_and_preprocess(info['name'], info['file'], info['target_column'])
            if data is not None:
                integrated_data[info['name']] = data
        
        return integrated_data

if __name__ == '__main__':
    integration_system = ExternalDatasetIntegration()
    integrated_dfs = integration_system.get_integrated_data()
    if integrated_dfs:
        print('\nIntegrated datasets info:')
        for name, df in integrated_dfs.items():
            print(f'\n--- Dataset: {name} ---')
            df.info()
            if 'isFraud' in df.columns:
                print(f'Total fraudulent transactions: {df["isFraud"].sum()}')
    else:
        print('No data integrated.')

