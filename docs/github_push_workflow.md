# GitHub Push Workflow

Jangan push dataset mentah atau cache training besar ke GitHub.

## Perintah Awal

```bash
git init
git add app src notebooks scripts docs artifacts_contract requirements.txt Dockerfile.api Dockerfile.web docker-compose.yml README.md .gitignore
git add models/model_v1.pt models/label_map.json models/model_v1_metadata.json reports/metrics_v1.json reports/confusion_matrix_v1.png reports/classification_report_v1.csv
git commit -m "Initial AgriMLOps PlantWild MVP"
git branch -M main
git remote add origin <GITHUB_REPO_URL>
git push -u origin main
```

## Catatan Model Besar

Jika `models/model_v1.pt` lebih dari 100MB, jangan commit ke GitHub biasa. Gunakan salah satu opsi berikut:

- GitHub Release
- Hugging Face model repository
- Object storage

Kemudian simpan URL model sebagai `MODEL_URL` di `.env` atau secret deployment.
