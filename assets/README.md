# assets

Taruh berkas milikmu sendiri di sini.

- **logo.png** — gambar watermark. Pakai PNG transparan, lebar 300–600 px.
  Aktifkan lewat `config.yaml`:

  ```yaml
  watermark:
    enabled: true
    path: assets/logo.png
    position: top-right   # top-left | top-right | bottom-left | bottom-right
    opacity: 0.75
    scale_width: 160      # lebar akhir di video 1080 px
  ```

  Atau sekali jalan dari terminal:

  ```bash
  python -m tukang_sayur clip video.mp4 --watermark assets/logo.png
  ```

Kalau berkasnya tidak ada, Tukang Sayur hanya memberi peringatan lalu
melanjutkan tanpa watermark — proses tidak akan berhenti.
