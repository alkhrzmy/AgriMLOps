# Report Structure - AgriMLOps PlantWild

## Folder Structure

```
report/
├── front_matter/
│   ├── cover.md (Cover page)
│   ├── kata_pengantar.md
│   ├── abstrak.md
│   ├── daftar_isi.md
│   ├── daftar_gambar.md
│   └── daftar_tabel.md
├── bab1/
│   ├── pendahuluan.md
│   ├── latar_belakang.md
│   ├── rumusan_masalah.md
│   ├── tujuan_penelitian.md
│   └── manfaat_penelitian.md
├── bab2/
│   ├── kajian_pustaka.md
│   ├── penyakit_tanaman.md
│   ├── deep_learning.md
│   ├── mlops.md
│   ├── active_learning.md
│   ├── model_registry.md
│   └── deployment.md
├── bab3/
│   ├── rancangan_dan_metode.md
│   ├── pendekatan.md
│   ├── konsep_rancangan.md
│   ├── alat_bahan.md
│   ├── tahapan_implementasi.md
│   └── kelayakan_dampak.md
├── bab4/
│   ├── hasil_dan_pembahasan.md
│   ├── hasil_pelatihan.md
│   ├── implementasi_mlops.md
│   ├── hasil_deployment.md
│   └── pembahasan.md
├── bab5/
│   ├── penutup.md
│   ├── kesimpulan.md
│   └── saran.md
├── lampiran/
│   ├── lampiran1_orisinalitas.md
│   ├── lampiran2_biodata.md
│   ├── lampiran3_turnitin.md
│   ├── lampiran4_repository.md
│   ├── lampiran5_metrics.md
│   ├── lampiran6_api_docs.md
│   └── lampiran7_screenshots.md
└── daftar_pustaka.md
```

## Workflow

1. **Study Papers** - Read and analyze all 16 papers
2. **Write BAB I** - Pendahuluan (background, problem, objectives, benefits)
3. **Write BAB II** - Kajian Pustaka (literature review from papers)
4. **Write BAB III** - Rancangan dan Metode (methodology based on project)
5. **Write BAB IV** - Hasil dan Pembahasan (results from v1, v2, v3)
6. **Write BAB V** - Penutup (conclusions and recommendations)
7. **Write Front Matter** - Cover, preface, abstract, table of contents
8. **Write Daftar Pustaka** - References from papers
9. **Write Lampiran** - Appendices
10. **Convert to DOCX** - Merge all markdown files into single DOCX

## Format Recommendation

**Use Markdown first, then convert to DOCX at the end.**

**Reasons:**
- Markdown is easier to edit and version control
- AI can work with markdown more effectively
- Easier to review and make changes
- Conversion to DOCX is straightforward with python-docx
- Better for collaborative editing

## Next Steps

1. Start studying the papers (Priority 1 papers first)
2. Extract key points for each section
3. Begin writing BAB I (Pendahuluan)
