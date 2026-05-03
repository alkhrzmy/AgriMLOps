# Kaggle Training Workflow

Training berat dilakukan di Kaggle GPU, bukan di environment lokal.

## Langkah

1. Buka Kaggle Notebook.
2. Enable GPU pada menu accelerator.
3. Upload atau copy isi `notebooks/kaggle_train_plantwild.py` ke Kaggle Notebook.
4. Jalankan notebook/script sampai selesai.
5. Download file output:

```text
/kaggle/working/agrimlops_artifacts.zip
```

6. Extract ZIP ke root repo lokal `agrimlops-plantwild` sehingga folder `models/` dan `reports/` terisi artifact.
7. Jalankan verifikasi:

```bash
python scripts/verify_artifacts.py
```

8. Test API lokal:

```bash
uvicorn app.api.main:app --reload
```

9. Test Streamlit lokal di terminal lain:

```bash
streamlit run app/web/streamlit_app.py
```

## Artifact yang Diharapkan

Lihat kontrak lengkap di `artifacts_contract/ARTIFACTS.md`.

## Catatan Dataset PlantWild

Repository Hugging Face `uqtwei2/PlantWild` menyimpan dataset sebagai archive:

```text
plantwild.zip
plantwild_v2.zip
```

Script Kaggle default menggunakan `plantwild.zip` untuk MVP agar download dan training lebih ringan. Jika ingin memakai v2, ubah konfigurasi di awal script:

```python
ARCHIVE_FILENAMES = ["plantwild_v2.zip"]
```

Script akan menampilkan tree ringkas folder download, extract archive ke `/kaggle/working/plantwild_extracted`, lalu scan gambar dari folder hasil extract.
