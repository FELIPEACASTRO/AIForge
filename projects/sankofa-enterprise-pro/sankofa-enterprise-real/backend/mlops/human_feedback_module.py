import logging
import pandas as pd
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HumanFeedbackModule:
    def __init__(self, feedback_storage_path='/home/ubuntu/sankofa-enterprise-real/data/human_feedback.csv'):
        self.feedback_storage_path = feedback_storage_path
        self._initialize_feedback_storage()

    def _initialize_feedback_storage(self):
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(self.feedback_storage_path), exist_ok=True)
        
        if not os.path.exists(self.feedback_storage_path):
            feedback_df = pd.DataFrame(columns=[
                'transaction_id', 'model_prediction', 'actual_label', 'feedback_timestamp', 'analyst_id', 'comments'
            ])
            feedback_df.to_csv(self.feedback_storage_path, index=False)
            logging.info(f'Arquivo de feedback criado em {self.feedback_storage_path}')
        else:
            logging.info(f'Arquivo de feedback existente em {self.feedback_storage_path}')

    def record_feedback(self, transaction_id: str, model_prediction: int, actual_label: int, analyst_id: str, comments: str = None):
        feedback_entry = {
            'transaction_id': transaction_id,
            'model_prediction': model_prediction,
            'actual_label': actual_label,
            'feedback_timestamp': datetime.now().isoformat(),
            'analyst_id': analyst_id,
            'comments': comments
        }
        try:
            # Obter feedbacks existentes
            feedback_df = self.get_feedback()
            
            # Adicionar novo feedback
            new_feedback_df = pd.DataFrame([feedback_entry])
            feedback_df = pd.concat([feedback_df, new_feedback_df], ignore_index=True)
            
            # Salvar de volta
            feedback_df.to_csv(self.feedback_storage_path, index=False)
            logging.info(f'Feedback registrado para transação {transaction_id}')
        except Exception as e:
            logging.error(f'Erro ao registrar feedback para transação {transaction_id}: {e}')

    def get_feedback(self) -> pd.DataFrame:
        try:
            if os.path.exists(self.feedback_storage_path) and os.path.getsize(self.feedback_storage_path) > 0:
                return pd.read_csv(self.feedback_storage_path)
            else:
                # Retornar DataFrame vazio com as colunas corretas
                return pd.DataFrame(columns=[
                    'transaction_id', 'model_prediction', 'actual_label', 'feedback_timestamp', 'analyst_id', 'comments'
                ])
        except Exception as e:
            logging.error(f'Erro ao carregar feedback: {e}')
            # Retornar DataFrame vazio com as colunas corretas em caso de erro
            return pd.DataFrame(columns=[
                'transaction_id', 'model_prediction', 'actual_label', 'feedback_timestamp', 'analyst_id', 'comments'
            ])

if __name__ == '__main__':
    # Exemplo de uso
    feedback_module = HumanFeedbackModule()

    # Registrar alguns feedbacks
    feedback_module.record_feedback('TRANS001', 1, 0, 'Analyst1', 'Falso positivo, transação legítima.')
    feedback_module.record_feedback('TRANS002', 0, 0, 'Analyst2', 'Transação legítima confirmada.')
    feedback_module.record_feedback('TRANS003', 1, 1, 'Analyst3', 'Fraude detectada e confirmada.')
    feedback_module.record_feedback('TRANS004', 0, 1, 'Analyst1', 'Falso negativo, fraude não detectada.')

    # Obter e exibir feedbacks
    all_feedback = feedback_module.get_feedback()
    print("\nTodos os Feedbacks:\n", all_feedback)

    # Exemplo de como o feedback pode ser usado para análise
    false_positives = all_feedback[(all_feedback['model_prediction'] == 1) & (all_feedback['actual_label'] == 0)]
    print("\nFalsos Positivos:\n", false_positives)

    false_negatives = all_feedback[(all_feedback['model_prediction'] == 0) & (all_feedback['actual_label'] == 1)]
    print("\nFalsos Negativos:\n", false_negatives)

