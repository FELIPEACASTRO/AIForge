#!/usr/bin/env python3
"""
Endpoints para o Módulo de Feedback Humano
Sankofa Enterprise Pro - Human Feedback API
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify
import pandas as pd
import os

from backend.mlops.human_feedback_module import HumanFeedbackModule

logger = logging.getLogger(__name__)

# Blueprint para os endpoints de feedback
feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')

# Instância global do módulo de feedback
feedback_module = HumanFeedbackModule()

@feedback_bp.route('/submit', methods=['POST'])
def submit_feedback():
    """
    Endpoint para submeter feedback de um analista sobre uma transação
    """
    try:
        data = request.get_json()
        
        # Validar campos obrigatórios
        required_fields = ['transaction_id', 'model_prediction', 'actual_label', 'analyst_id']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Campos obrigatórios faltando',
                'missing_fields': missing_fields
            }), 400
        
        # Validar tipos de dados
        try:
            transaction_id = str(data['transaction_id'])
            model_prediction = int(data['model_prediction'])
            actual_label = int(data['actual_label'])
            analyst_id = str(data['analyst_id'])
            comments = data.get('comments', '')
            
            # Validar valores binários
            if model_prediction not in [0, 1]:
                return jsonify({'error': 'model_prediction deve ser 0 ou 1'}), 400
            if actual_label not in [0, 1]:
                return jsonify({'error': 'actual_label deve ser 0 ou 1'}), 400
                
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Erro na validação dos dados: {str(e)}'}), 400
        
        # Registrar o feedback
        feedback_module.record_feedback(
            transaction_id=transaction_id,
            model_prediction=model_prediction,
            actual_label=actual_label,
            analyst_id=analyst_id,
            comments=comments
        )
        
        logger.info(f"✅ Feedback registrado para transação {transaction_id} por {analyst_id}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback registrado com sucesso',
            'transaction_id': transaction_id,
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"❌ Erro ao submeter feedback: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@feedback_bp.route('/list', methods=['GET'])
def list_feedback():
    """
    Endpoint para listar todos os feedbacks registrados
    """
    try:
        # Parâmetros de paginação
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        analyst_id = request.args.get('analyst_id', None)
        
        # Obter todos os feedbacks
        feedback_df = feedback_module.get_feedback()
        
        if feedback_df.empty:
            return jsonify({
                'feedbacks': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0
            })
        
        # Filtrar por analista se especificado
        if analyst_id:
            feedback_df = feedback_df[feedback_df['analyst_id'] == analyst_id]
        
        # Ordenar por timestamp mais recente
        feedback_df = feedback_df.sort_values('feedback_timestamp', ascending=False)
        
        # Paginação
        total = len(feedback_df)
        total_pages = (total + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paginated_df = feedback_df.iloc[start_idx:end_idx]
        
        # Converter para lista de dicionários
        feedbacks = paginated_df.to_dict('records')
        
        return jsonify({
            'feedbacks': feedbacks,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao listar feedbacks: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@feedback_bp.route('/analytics', methods=['GET'])
def get_feedback_analytics():
    """
    Endpoint para obter análises dos feedbacks coletados
    """
    try:
        feedback_df = feedback_module.get_feedback()
        
        if feedback_df.empty:
            return jsonify({
                'total_feedbacks': 0,
                'accuracy_metrics': {},
                'analyst_stats': {},
                'error_analysis': {}
            })
        
        # Métricas de acurácia do modelo
        total_feedbacks = len(feedback_df)
        correct_predictions = len(feedback_df[feedback_df['model_prediction'] == feedback_df['actual_label']])
        model_accuracy = correct_predictions / total_feedbacks if total_feedbacks > 0 else 0
        
        # Análise de erros
        false_positives = len(feedback_df[(feedback_df['model_prediction'] == 1) & (feedback_df['actual_label'] == 0)])
        false_negatives = len(feedback_df[(feedback_df['model_prediction'] == 0) & (feedback_df['actual_label'] == 1)])
        true_positives = len(feedback_df[(feedback_df['model_prediction'] == 1) & (feedback_df['actual_label'] == 1)])
        true_negatives = len(feedback_df[(feedback_df['model_prediction'] == 0) & (feedback_df['actual_label'] == 0)])
        
        # Métricas de performance
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Estatísticas por analista
        analyst_stats = {}
        for analyst_id in feedback_df['analyst_id'].unique():
            analyst_data = feedback_df[feedback_df['analyst_id'] == analyst_id]
            analyst_stats[analyst_id] = {
                'total_feedbacks': len(analyst_data),
                'fraud_confirmations': len(analyst_data[analyst_data['actual_label'] == 1]),
                'legitimate_confirmations': len(analyst_data[analyst_data['actual_label'] == 0]),
                'last_feedback': analyst_data['feedback_timestamp'].max()
            }
        
        # Tendências temporais (últimos 30 dias)
        feedback_df['feedback_timestamp'] = pd.to_datetime(feedback_df['feedback_timestamp'])
        recent_feedback = feedback_df[feedback_df['feedback_timestamp'] >= (datetime.now() - pd.Timedelta(days=30))]
        
        daily_feedback = recent_feedback.groupby(recent_feedback['feedback_timestamp'].dt.date).size().to_dict()
        daily_feedback = {str(k): v for k, v in daily_feedback.items()}  # Converter datas para string
        
        return jsonify({
            'total_feedbacks': total_feedbacks,
            'accuracy_metrics': {
                'model_accuracy': round(model_accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4)
            },
            'confusion_matrix': {
                'true_positives': true_positives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            },
            'analyst_stats': analyst_stats,
            'daily_feedback_trend': daily_feedback,
            'error_analysis': {
                'false_positive_rate': round(false_positives / total_feedbacks, 4) if total_feedbacks > 0 else 0,
                'false_negative_rate': round(false_negatives / total_feedbacks, 4) if total_feedbacks > 0 else 0
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter analytics de feedback: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@feedback_bp.route('/batch', methods=['POST'])
def submit_batch_feedback():
    """
    Endpoint para submeter múltiplos feedbacks em lote
    """
    try:
        data = request.get_json()
        
        if 'feedbacks' not in data or not isinstance(data['feedbacks'], list):
            return jsonify({'error': 'Campo "feedbacks" deve ser uma lista'}), 400
        
        results = []
        errors = []
        
        for i, feedback_data in enumerate(data['feedbacks']):
            try:
                # Validar campos obrigatórios
                required_fields = ['transaction_id', 'model_prediction', 'actual_label', 'analyst_id']
                missing_fields = [field for field in required_fields if field not in feedback_data]
                
                if missing_fields:
                    errors.append({
                        'index': i,
                        'error': f'Campos obrigatórios faltando: {missing_fields}'
                    })
                    continue
                
                # Registrar o feedback
                feedback_module.record_feedback(
                    transaction_id=str(feedback_data['transaction_id']),
                    model_prediction=int(feedback_data['model_prediction']),
                    actual_label=int(feedback_data['actual_label']),
                    analyst_id=str(feedback_data['analyst_id']),
                    comments=feedback_data.get('comments', '')
                )
                
                results.append({
                    'index': i,
                    'transaction_id': feedback_data['transaction_id'],
                    'status': 'success'
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': len(results),
            'errors': len(errors),
            'results': results,
            'error_details': errors
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao submeter feedback em lote: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@feedback_bp.route('/export', methods=['GET'])
def export_feedback():
    """
    Endpoint para exportar feedbacks em formato CSV
    """
    try:
        feedback_df = feedback_module.get_feedback()
        
        if feedback_df.empty:
            return jsonify({'error': 'Nenhum feedback disponível para exportar'}), 404
        
        # Converter para CSV
        csv_data = feedback_df.to_csv(index=False)
        
        from flask import Response
        
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao exportar feedbacks: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@feedback_bp.route('/transaction/<transaction_id>', methods=['GET'])
def get_transaction_feedback(transaction_id):
    """
    Endpoint para obter feedback específico de uma transação
    """
    try:
        feedback_df = feedback_module.get_feedback()
        
        if feedback_df.empty:
            return jsonify({'error': 'Nenhum feedback encontrado'}), 404
        
        # Filtrar por transaction_id
        transaction_feedback = feedback_df[feedback_df['transaction_id'] == transaction_id]
        
        if transaction_feedback.empty:
            return jsonify({'error': f'Nenhum feedback encontrado para a transação {transaction_id}'}), 404
        
        # Converter para lista de dicionários
        feedbacks = transaction_feedback.to_dict('records')
        
        return jsonify({
            'transaction_id': transaction_id,
            'feedbacks': feedbacks,
            'total_feedbacks': len(feedbacks)
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter feedback da transação {transaction_id}: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

