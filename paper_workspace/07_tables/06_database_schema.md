# Tabel Database Schema

| Tabel | Field utama | Fungsi |
|---|---|---|
| prediction_logs | id, image_path, predicted_label, confidence, top_k_predictions, needs_review, timestamp | Mencatat seluruh prediksi |
| feedback_logs | id, prediction_id, feedback_type, suggested_label, confidence, timestamp | Mencatat feedback user/evaluator |
| active_learning_queue | id, prediction_id, reason, status, validated_label | Queue validasi active learning |
| model_registry | model_version, artifact_path, accuracy, macro_f1, feedback_samples_used, status | Registry dan lineage model |
