# BAB V
# PENUTUP

## 5.1 Kesimpulan

Berdasarkan hasil penelitian dan pembahasan yang telah dilakukan, dapat disimpulkan bahwa:

1. **Sistem diagnosis penyakit tanaman berbasis deep learning berhasil diimplementasikan** menggunakan EfficientNetV2-B0 dengan dataset PlantWild subset (15 kelas, 5.645 gambar). Model baseline (v1) mencapai accuracy 0.8217 dan macro F1 0.8086, yang menunjukkan bahwa transfer learning dari ImageNet efektif untuk domain penyakit tanaman.

2. **MLOps lifecycle terintegrasi penuh** mencakup inference, prediction logging, feedback collection, active learning queue, monitoring dashboard, dan model registry. Implementasi ini memungkinkan pengelolaan model yang berkelanjutan dengan tracking version, comparison performa, dan deployment management yang sistematis.

3. **Active learning queue berhasil diimplementasikan** dengan 9 validated feedback samples dari controlled active learning simulation. Queue ini secara otomatis menambahkan prediksi dengan confidence rendah atau user unsure untuk validasi, kemudian data yang divalidasi digunakan untuk retraining model secara iteratif.

4. **Model v3 berhasil dilatih dengan active learning retraining** menggunakan 9 validated feedback samples sebagai transfer learning dari model v2. Model v3 mencapai accuracy 0.8288 (+0.0071 vs v1, +0.0035 vs v2), macro F1 0.8127 (+0.0041 vs v1, +0.0003 vs v2), dan Best Val F1 0.8254 (+0.0147 vs v2). Peningkatan Best Val F1 yang signifikan menunjukkan bahwa active learning dapat meningkatkan generalisasi model.

5. **Sistem berhasil dideploy pada DigitalOcean Droplet** dengan Docker Compose, menyediakan akses web (Streamlit) dan API (FastAPI) yang skalabel dan mudah diakses. Deployment menunjukkan performa yang baik dengan API response time < 500ms dan resource usage yang efisien pada Droplet basic plan.

6. **Penelitian ini mengisi gap literatur** dengan mengintegrasikan komponen MLOps (inference, logging, feedback, active learning, monitoring, model registry, deployment) ke dalam satu platform khusus diagnosis penyakit tanaman in-the-wild, yang sebelumnya belum ada dalam literatur.

## 5.2 Saran

Berdasarkan hasil penelitian dan pembahasan, beberapa saran untuk pengembangan selanjutnya adalah:

### 5.2.1 Integrasi dengan Aplikasi Mobile

Mengembangkan aplikasi mobile native (Android/iOS) atau Progressive Web App (PWA) untuk memudahkan petani mengakses sistem diagnosis di lapangan. Aplikasi mobile dapat:
- Menggunakan kamera device untuk capture gambar langsung
- Menyediakan offline mode dengan model lokal
- Integrasi dengan GPS untuk location tracking
- Push notification untuk hasil diagnosis

### 5.2.2 Real Farmer Feedback Collection

Implementasi feedback collection dari petani nyata melalui:
- Integrasi dengan aplikasi mobile yang digunakan petani
- Field app untuk extension workers
- SMS-based feedback untuk area dengan koneksi terbatas
- Quality control dengan validation oleh ahli pertanian

### 5.2.3 Automated Retraining Pipeline

Mengembangkan CI/CD pipeline untuk automated retraining:
- Trigger retraining berdasarkan jumlah feedback samples tertentu
- Automated testing dan validation sebelum deployment
- Canary deployment untuk testing gradual
- Rollback otomatis jika degradation terdeteksi

### 5.2.4 Model Explainability

Menambahkan interpretability model untuk meningkatkan trust user:
- Grad-CAM visualization untuk menunjukkan area gambar yang mempengaruhi prediksi
- Attention map untuk highlight symptoms
- Natural language explanation untuk diagnosis

### 5.2.5 Authentication dan Authorization

Implementasi security layer untuk produksi:
- OAuth2 atau JWT untuk user authentication
- Role-based access control (admin, farmer, extension worker)
- Rate limiting untuk mencegah abuse
- Audit logging untuk compliance

### 5.2.6 Multi-language Support

Menambahkan dukungan bahasa lokal:
- Bahasa Indonesia untuk petani lokal
- Bahasa Inggris untuk dokumentasi teknis
- Bahasa lain sesuai kebutuhan regional

### 5.2.7 Real-time Monitoring dengan Alerts

Upgrade monitoring dashboard dengan:
- Prometheus/Grafana integration untuk metrics collection
- Automated alerts untuk degradation atau drift
- Anomaly detection untuk prediksi yang tidak biasa
- SLA monitoring untuk availability

### 5.2.8 Edge Deployment

Optimasi model untuk deployment pada edge devices:
- TensorFlow Lite conversion untuk mobile
- Model quantization untuk mengurangi size
- ONNX deployment untuk inference cepat
- Hardware acceleration dengan GPU/NPU

### 5.2.9 Multi-modal Data Integration

Integrasi data tambahan untuk diagnosis yang lebih akurat:
- Environmental data (temperature, humidity, rainfall)
- Soil data (pH, moisture, nutrients)
- Satellite imagery untuk crop health monitoring
- IoT sensors untuk real-time monitoring

### 5.2.10 Dataset Expansion

Perluas dataset untuk mencakup:
- Lebih banyak kelas penyakit tanaman
- Variasi geografis dan musiman
- Berbagai kondisi pencahayaan dan latar belakang
- Multi-species crops (padi, jagung, kedelai, dll)

### 5.2.11 A/B Testing Framework

Implementasi A/B testing untuk:
- Compare performa model version secara live
- Test UI/UX improvements
- Validate active learning strategies
- Optimize confidence thresholds

### 5.2.12 Cost Optimization

Optimasi biaya deployment:
- Auto-scaling berdasarkan load
- Serverless deployment untuk spike traffic
- Spot instances untuk training
- CDN untuk static assets

Saran-saran ini diarahkan untuk meningkatkan kualitas, skalabilitas, dan dampak sosial dari sistem AgriMLOps, menjadikannya platform yang lebih komprehensif dan siap untuk produksi skala besar.
