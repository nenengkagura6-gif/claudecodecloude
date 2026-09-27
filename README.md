# Tukang Sayur

**Potong video panjang jadi klip vertikal 9:16 siap unggah — otomatis.**

Kasih satu URL YouTube (atau file video di komputer), Tukang Sayur akan:

1. mengunduh dan merapikan videonya (yt-dlp + ffmpeg),
2. membuat transkrip dengan timestamp **per kata** (faster-whisper, jalan di CPU),
3. meminta LLM mencari 5–15 bagian paling menarik (potongan tepat di batas kalimat),
4. menilai tiap kandidat 0–100 (hook, emosi, kejelasan, berdiri sendiri),
5. membingkai ulang jadi 9:16 sambil mengikuti wajah (atau latar blur kalau tak ada wajah),
6. membakar subtitle animasi per kata + teks hook **di atas video yang tetap berjalan**,
7. menulis klip MP4, thumbnail, `metadata.json`, dan `metadata.csv` (judul, caption, hashtag, skor).

```
python -m tukang_sayur clip https://youtu.be/xxxxxxxx --max-clips 10
```

Tidak ada freeze-frame di awal klip: hook adalah teks yang ditumpuk di atas
gambar dan suara yang jalan sejak detik 0.

---

## Daftar isi

- [Kebutuhan sistem](#kebutuhan-sistem)
- [Instalasi Linux](#instalasi-linux-ubuntudebian)
- [Instalasi Windows 10](#instalasi-windows-10)
- [Isi file .env](#isi-file-env)
- [Cara pakai](#cara-pakai)
- [Antarmuka web (Gradio)](#antarmuka-web-gradio)
- [Dipanggil dari n8n](#dipanggil-dari-n8n)
- [Contoh config.yaml](#contoh-configyaml)
- [Hasil keluaran](#hasil-keluaran)
- [Menjalankan test](#menjalankan-test)
- [Troubleshooting](#troubleshooting)
- [Struktur proyek](#struktur-proyek)

---

## Kebutuhan sistem

| Kebutuhan | Minimum | Catatan |
|---|---|---|
| Python | 3.9+ | 3.10/3.11 paling nyaman |
| RAM | 8 GB | 12 GB nyaman untuk model Whisper `small` |
| Disk | ~3 GB | model Whisper `small` ±500 MB + file kerja |
| GPU | **tidak perlu** | semua default disetel untuk CPU (`int8`) |
| ffmpeg | wajib | beserta `ffprobe` |

Perkiraan waktu di CPU biasa (4 core), video 10 menit:
transkripsi 3–8 menit, render ±20–40 detik per klip.

---

## Instalasi Linux (Ubuntu/Debian)

```bash
# 1. Alat dasar
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg git

# 2. Ambil proyeknya
git clone <url-repo-ini> tukang-sayur
cd tukang-sayur

# 3. Lingkungan virtual (disarankan)
python3 -m venv .venv
source .venv/bin/activate

# 4. Paket Python
pip install --upgrade pip
pip install -r requirements.txt

# 5. Kunci API (boleh dilewati kalau pakai Ollama)
cp .env.example .env
nano .env

# 6. Cek semuanya sudah siap
python -m tukang_sayur doctor

# 7. Uji tanpa internet: bikin video contoh lalu proses
python -m tukang_sayur demo
```

Font untuk subtitle (opsional tapi disarankan, supaya gayanya sesuai config):

```bash
sudo apt install -y fonts-montserrat   # atau salin file .ttf ke ~/.fonts lalu: fc-cache -f
```

Kalau font tidak ada, libass otomatis memakai font pengganti — subtitle tetap muncul.

---

## Instalasi Windows 10

### 1. Python

Unduh dari <https://www.python.org/downloads/windows/> (3.10 atau 3.11).
Saat instalasi, **centang "Add python.exe to PATH"**.

### 2. ffmpeg

Cara cepat (butuh winget, bawaan Windows 10 versi baru):

```powershell
winget install --id=Gyan.FFmpeg -e
```

Cara manual:

1. Unduh build "release full" dari <https://www.gyan.dev/ffmpeg/builds/>.
2. Ekstrak ke `C:\ffmpeg`.
3. Tambahkan `C:\ffmpeg\bin` ke PATH:
   *Start* → ketik "environment variables" → **Edit the system environment variables** →
   **Environment Variables…** → pilih `Path` di *User variables* → **Edit** → **New** →
   isi `C:\ffmpeg\bin` → OK semua.
4. Tutup dan buka lagi PowerShell, lalu cek: `ffmpeg -version`

### 3. Proyek ini

```powershell
git clone <url-repo-ini> tukang-sayur
cd tukang-sayur

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Kalau PowerShell menolak menjalankan skrip:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
notepad .env

python -m tukang_sayur doctor
python -m tukang_sayur demo
```

### 4. Font (opsional)

Unduh Montserrat dari Google Fonts, klik kanan file `.ttf` → **Install for all users**.
Nama font di `config.yaml` harus sama persis dengan nama yang terdaftar di Windows.

> **Catatan Windows:** jangan taruh proyek di folder yang path-nya mengandung
> karakter aneh. Path dengan spasi aman (sudah diuji), tapi folder OneDrive yang
> sedang sinkron kadang mengunci file saat ffmpeg menulis.

---

## Isi file .env

`.env` **tidak pernah** ikut ter-commit (sudah masuk `.gitignore`). Isi yang dipakai saja:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
TUKANG_SAYUR_TOKEN=token-rahasia-untuk-webhook
```

Tidak punya kunci API sama sekali? Pakai LLM lokal:

```bash
# sekali saja
curl -fsSL https://ollama.com/install.sh | sh     # Linux
ollama pull llama3.1:8b

# lalu jalankan Tukang Sayur dengan:
python -m tukang_sayur clip video.mp4 --provider ollama
```

---

## Cara pakai

### Satu video

```bash
# dari URL
python -m tukang_sayur clip "https://youtu.be/xxxxxxxx" --max-clips 10

# dari file lokal
python -m tukang_sayur clip "D:\rekaman\podcast.mp4" --max-clips 5

# ganti provider dan model LLM sekali jalan
python -m tukang_sayur clip video.mp4 --provider openai --model gpt-4o-mini

# bahasa Sunda, model Whisper lebih besar
python -m tukang_sayur clip video.mp4 --lang su --whisper-model medium

# potong jeda hening + watermark
python -m tukang_sayur clip video.mp4 --remove-silence --watermark assets/logo.png

# lihat dulu kandidat & skornya tanpa merender (cepat)
python -m tukang_sayur clip video.mp4 --dry-run
```

Semua nilai di `config.yaml` bisa ditimpa lewat `--set`:

```bash
python -m tukang_sayur clip video.mp4 \
  --set subtitle.font_size=72 \
  --set subtitle.highlight_color="#00FFAA" \
  --set hook.duration=2.5 \
  --set render.crf=18
```

### Banyak video (batch, bisa dilanjutkan)

`daftar.txt`:

```
# satu URL atau path per baris, '#' untuk komentar
https://youtu.be/aaaaaaaaaaa
https://youtu.be/bbbbbbbbbbb
D:\rekaman\wawancara.mp4
```

```bash
python -m tukang_sayur batch daftar.txt --max-clips 5
```

Mati listrik atau ditekan Ctrl+C? Jalankan lagi perintah yang sama — yang sudah
selesai dilewati (statusnya disimpan di `output/batch_state.json`).
Mau ulang dari nol: tambahkan `--no-resume`.

### Perintah lain

```bash
python -m tukang_sayur doctor        # cek ffmpeg, paket, kunci API
python -m tukang_sayur demo          # uji pipeline penuh tanpa internet
python -m tukang_sayur transcribe video.mp4 -o transkrip.json
python -m tukang_sayur ui            # antarmuka web
python -m tukang_sayur serve         # endpoint HTTP untuk n8n
python -m tukang_sayur clip --help   # semua opsi
```

---

## Antarmuka web (Gradio)

```bash
pip install gradio
python -m tukang_sayur ui
```

Buka <http://127.0.0.1:7860>. Di sana ada kolom URL, tombol unggah file, slider
jumlah klip, pilihan provider LLM dan model Whisper, pengaturan durasi serta
gaya subtitle, plus bar progres tiap tahap. Klip hasilnya langsung bisa diputar
dan diunduh dari halaman itu.

---

## Dipanggil dari n8n

```bash
pip install fastapi uvicorn
python -m tukang_sayur serve --port 8000
```

Node **HTTP Request** di n8n:

| Isian | Nilai |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8000/clip` |
| Header | `Authorization: Bearer <TUKANG_SAYUR_TOKEN>` |
| Body (JSON) | `{"source": "https://youtu.be/xxxx", "max_clips": 5}` |

Balasannya:

```json
{ "job_id": "9f2c1a7b3d40", "status": "antre", "poll": "/jobs/9f2c1a7b3d40" }
```

Cek berkala ke `GET /jobs/9f2c1a7b3d40` sampai `status` jadi `selesai`; isi
`result` sama persis dengan `metadata.json`. Kalau mau menunggu dalam satu
permintaan, kirim `"wait": true` (pastikan timeout n8n dinaikkan).

Endpoint lain: `GET /health`, `GET /jobs`, dan dokumen otomatis di `/docs`.

> Kalau `TUKANG_SAYUR_TOKEN` kosong, endpoint terbuka tanpa autentikasi.
> Pakai hanya di jaringan lokal.

---

## Contoh config.yaml

File `config.yaml` di root sudah berisi semua pengaturan beserta komentarnya.
Yang paling sering diubah:

```yaml
transcription:
  model: small        # tiny | base | small | medium  (small = pas untuk RAM 12 GB)
  compute_type: int8  # paling ringan untuk CPU
  language: id        # id (Indonesia) | su (Sunda) | auto

llm:
  provider: anthropic # anthropic | openai | gemini | ollama | mock
  providers:
    anthropic:
      model: claude-sonnet-5
      api_key_env: ANTHROPIC_API_KEY
    ollama:
      model: llama3.1:8b
      base_url: http://localhost:11434

candidates:
  min_clips: 5
  max_clips: 15
  min_duration: 20    # detik
  max_duration: 90
  snap_to_sentence: true

scoring:
  weights: { hook: 0.35, emotion: 0.25, clarity: 0.20, standalone: 0.20 }
  min_score: 55       # klip di bawah ini tidak dirender

subtitle:
  font_name: Montserrat
  font_size: 64
  highlight_color: "&H0000E5FF"  # boleh juga "#FFCC00" atau "kuning"
  max_words_per_line: 4
  margin_bottom: 420
  karaoke: true       # highlight kata yang sedang diucapkan

hook:
  enabled: true
  duration: 3.0       # detik, tampil di atas video yang tetap berjalan
  font_size: 78
  margin_top: 240

audio:
  remove_silence: false
  loudnorm: true
  target_lufs: -14

reframe:
  enabled: true
  detector: auto      # auto | mediapipe | opencv | none
  fallback: blur      # kalau tidak ada wajah: blur | center

render:
  width: 1080
  height: 1920
  crf: 20
  preset: veryfast
```

`config.yaml` boleh ditulis sebagian saja — nilai yang tidak ditulis memakai
default bawaan kode (`tukang_sayur/config.py`).

**Warna subtitle** menerima tiga bentuk: `&HAABBGGRR` (format asli ASS),
`#RRGGBB` (seperti CSS), atau nama (`kuning`, `putih`, `merah`, …).

---

## Hasil keluaran

```
output/podcast-episode-12-20260927-193514/
├── 01-kenapa-harga-sayur-naik.mp4      ← klip 1080x1920 siap unggah
├── 01-kenapa-harga-sayur-naik.jpg      ← thumbnail
├── 02-rahasia-pedagang-pasar.mp4
├── 02-rahasia-pedagang-pasar.jpg
├── transcript.json                     ← transkrip lengkap per kata
├── metadata.json                       ← semua data job (termasuk best_score)
└── metadata.csv                        ← satu baris per klip, enak dibuka di Excel
```

Kolom `metadata.csv`: `index, file, title, caption, hashtags, score,
score_hook, score_emotion, score_clarity, score_standalone, hook_text,
duration, source_start, source_end, source_start_tc, source_end_tc,
thumbnail, resolution, reframe_mode`.

---

## Menjalankan test

```bash
pip install pytest
pytest                      # semua test
pytest -m "not slow"        # lewati test yang merender video (butuh ffmpeg)
pytest tests/test_scoring.py -v
```

Seluruh test jalan **tanpa internet dan tanpa kunci API**: LLM diganti
`MockProvider`, transkripsi diganti `DummyTranscriber`, dan video ujinya dibuat
sendiri oleh ffmpeg. Test end-to-end ikut memastikan klip benar-benar 9:16,
audionya ada sejak detik 0, dan gambarnya **tidak membeku** selama hook tampil.

---

## Troubleshooting

**`ffmpeg tidak ditemukan di PATH`**
Belum terpasang atau PATH belum ter-refresh. Tutup lalu buka lagi terminalnya.
Cek dengan `ffmpeg -version`. Kalau ffmpeg ada di lokasi tidak standar, arahkan
manual: `set TUKANG_SAYUR_FFMPEG=C:\ffmpeg\bin\ffmpeg.exe` (Windows) atau
`export TUKANG_SAYUR_FFMPEG=/opt/ffmpeg/ffmpeg` (Linux).

**Transkripsi lama sekali**
Wajar di CPU. Percepat dengan `--whisper-model base`, atau pakai
`--set transcription.vad_filter=true` supaya bagian senyap dilewati. Hasil
transkrip di-cache, jadi menjalankan ulang video yang sama jauh lebih cepat.

**`Gagal memuat model Whisper`**
Model diunduh otomatis saat pertama dipakai (`small` ±500 MB). Pastikan ada
koneksi dan ruang disk. Di jaringan berproxy, set `HTTPS_PROXY` dulu.

**Hasil transkrip bahasa Sunda berantakan**
Model kecil memang lemah untuk Sunda. Coba `--whisper-model medium`, atau
`--lang auto` supaya Whisper menebak sendiri.

**`ANTHROPIC_API_KEY belum diisi`**
Salin `.env.example` jadi `.env` lalu isi kuncinya, atau pakai LLM lokal:
`--provider ollama`. Untuk sekadar mencoba pipeline: `--provider mock`.

**`Tidak bisa menyambung ke Ollama`**
Jalankan `ollama serve` di terminal lain, lalu `ollama pull llama3.1:8b`.

**LLM membalas tapi kandidatnya kosong / JSON-nya aneh**
Parser sudah memaafkan banyak hal (blok ```json, koma berlebih, komentar, nama
kunci berbeda). Kalau tetap gagal, biasanya modelnya terlalu kecil — pakai model
yang lebih pintar, atau turunkan `candidates.min_duration`.

**Semua klip skornya 0**
Seharusnya tidak mungkin: skor dihitung ulang dari rincian dimensi, dan kalau
penilaian LLM gagal, sistem jatuh ke skor heuristik yang selalu di atas 0.
Kalau tetap terjadi, jalankan dengan `--log-level DEBUG` dan lihat baris
peringatan dari modul `scoring` — baris itu menyebut penyebabnya.

**Subtitle tidak muncul / fontnya beda**
Font di `config.yaml` belum terpasang di sistem. Install fontnya, atau ganti
`subtitle.font_name` ke font yang pasti ada (`DejaVu Sans` di Linux, `Arial` di
Windows).

**Wajah tidak diikuti (selalu latar blur)**
Pasang OpenCV: `pip install opencv-python-headless` (dan `pip install mediapipe`
kalau platformmu didukung — hasilnya lebih akurat). Kalau wajah memang jarang
terdeteksi (<15% sampel), sistem sengaja memilih mode blur supaya gambar tidak
goyang.

**Video 4K bikin RAM penuh**
Turunkan dulu di tahap ingest: `--set ingest.normalize.max_height=720`.

**yt-dlp gagal: "Sign in to confirm you're not a bot"**
Isi `ingest.cookies_from_browser: chrome` (atau `firefox`) di `config.yaml`
supaya yt-dlp memakai cookie browser yang sudah login.

**Klip terpotong di tengah kalimat**
Naikkan toleransi dengan `--set candidates.snap_to_sentence=true` (default) dan
pastikan transkripnya bagus — pemotongan mengikuti batas kalimat dari transkrip,
jadi kalau transkripsinya kacau, batasnya ikut kacau.

---

## Struktur proyek

```
tukang_sayur/
├── cli.py           # perintah terminal (clip, batch, demo, ui, serve, doctor)
├── config.py        # default + config.yaml + .env + override CLI
├── models.py        # Word, Segment, Transcript, Candidate, ScoredCandidate, ClipResult
├── ingest.py        # yt-dlp + normalisasi ffmpeg
├── transcribe.py    # faster-whisper, fallback timestamp kata, cache
├── llm/             # provider (anthropic/openai/gemini/ollama/mock) + parser JSON
├── prompts.py       # semua prompt LLM
├── candidates.py    # pencari kandidat + penyelarasan ke batas kalimat
├── scoring.py       # juri viral + best_score + heuristik cadangan
├── reframe.py       # deteksi wajah, pemulusan gerak, rencana crop 9:16
├── ffexpr.py        # keyframe -> ekspresi ffmpeg (dan evaluasinya di Python)
├── subtitles.py     # file ASS: subtitle per kata + hook
├── audio.py         # potong hening, loudnorm, pemetaan ulang timestamp
├── render.py        # perintah ffmpeg filter_complex
├── metadata.py      # judul/caption/hashtag + metadata.json & .csv
├── pipeline.py      # perakit seluruh tahap
├── batch.py         # mode batch + resume
├── webapp.py        # antarmuka Gradio
├── server.py        # API/webhook FastAPI
└── demo.py          # video contoh untuk uji coba offline

tests/               # 259 test pytest, semuanya offline
config.yaml          # pengaturan
.env.example         # contoh isian kunci API
```

---

## Lisensi

MIT.
