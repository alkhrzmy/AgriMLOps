# README FINAL - AgriMLOps KTI

## Output Final

- `paper_workspace/09_final_docx/AgriMLOps_KTI_Final.docx`
- `report/AgriMLOps_KTI_Final.docx`

## Workspace Structure

- `00_format/`: template format `FORMAT.docx`.
- `01_current_draft/`: draft DOCX awal.
- `02_references_pdf/`: seluruh PDF paper referensi.
- `03_reference_notes/`: literature matrix dan catatan referensi.
- `04_project_evidence/`: metrics JSON, metadata model, CSV report, dan evidence project.
- `05_figures_source/`: sumber gambar/plot project.
- `06_figures_final/`: gambar final untuk laporan.
- `07_tables/`: tabel wajib dalam markdown.
- `08_report_markdown/`: versi markdown laporan dan script konversi.
- `09_final_docx/`: DOCX final.

## Gambar Final

Gambar yang dibuat/dikumpulkan:

1. `gambar_2_1_plant_disease_impact.png`
2. `gambar_2_2_dl_evolution.png`
3. `gambar_2_3_mlops_lifecycle.png`
4. `gambar_2_4_active_learning_flow.png`
5. `gambar_3_1_system_architecture.png`
6. `gambar_3_2_database_schema.png`
7. `gambar_3_3_api_endpoints.png`
8. `gambar_3_4_streamlit_pages.png`
9. `gambar_3_5_deployment_architecture.png`
10. `gambar_3_6_training_workflow.png`
11. `gambar_3_7_controlled_feedback_simulation.png`
12. `gambar_4_1_model_comparison.png`
13. `gambar_4_2_training_curves_v3.png`
14. `confusion_matrix_v3.png`
15. `sample_predictions.png`
16. `gambar_4_5_diagnosis_page.png`
17. `gambar_4_6_prediction_result.png`
18. `gambar_4_7_feedback_form.png`
19. `gambar_4_8_active_learning_queue.png`
20. `gambar_4_9_monitoring_dashboard.png`
21. `gambar_4_10_model_registry.png`
22. `gambar_4_11_fastapi_docs.png`
23. `gambar_4_12_fastapi_health.png`
24. `class_distribution.png`

## Tabel Wajib

Tabel yang dibuat dan/atau dimasukkan:

- Ringkasan penelitian terkait
- Gap penelitian
- Spesifikasi teknologi
- Dataset PlantWild subset
- Konfigurasi training v1-v2-v3
- Database schema
- API endpoints
- Perbandingan model v1-v2-v3
- Hasil controlled active learning simulation
- Fitur MLOps yang diimplementasikan
- Perbandingan AgriMLOps dengan paper terkait
- Limitasi dan future work

## Catatan Klaim Ilmiah Aman

- Model final adalah `v3`.
- Metrics final: accuracy `0.8288`, macro F1 `0.8127`, best validation macro F1 `0.8254`.
- `v3` memakai `9` validated feedback samples dari **controlled validated feedback simulation**.
- Feedback yang digunakan **bukan feedback petani asli** dan tidak diklaim sebagai validasi lapangan.
- Sistem **tidak mengklaim true open-world detection** seperti Dong et al. (2023).
- Sistem **tidak mengklaim domain adaptation eksplisit** seperti Wu et al. (2023).
- Sistem **tidak mengklaim multimodal retrieval/diagnosis** seperti PlantWild multimodal atau Snap and Diagnose.
- Retraining masih **semi-manual berbasis Kaggle GPU**, belum automated CI/CD retraining penuh.
- Deployment production digunakan untuk inference, logging, feedback collection, monitoring, dan model registry.

## Production Evidence

- API: http://159.65.139.148:8000
- Web: http://159.65.139.148:8501
- Screenshot production sudah disimpan di `report/images/` dan `paper_workspace/06_figures_final/`.

## Script Pendukung

- `scripts/generate_paper_like_assets.py`: regenerate diagram paper-like, panel screenshot evidence, dan analisis visual referensi tanpa menyalin gambar paper.
- `paper_workspace/08_report_markdown/convert_to_docx_v2.py`: convert markdown laporan ke DOCX final dengan gambar, caption, source note, dan tabel rapi.
- `paper_workspace/03_reference_notes/visual_reference_analysis.md`: catatan referensi visual dari paper yang digunakan sebagai inspirasi, bukan copy langsung.
