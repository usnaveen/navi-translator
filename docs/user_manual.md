# Na'vi Translator — User Manual

## Getting Started

The Na'vi Translator converts Na'vi language audio or text into English. No technical knowledge is required to use it.

### Accessing the Application

Open your web browser and go to: **http://localhost:3000**

---

## Features

### 1. Translate Audio

**How to use:**
1. Click the **Translate Audio** tab
2. Click **Start Recording** and speak a Na'vi word or phrase into your microphone
3. Click **Stop Recording** when finished
4. Wait for the translation to appear

**Alternatively**, click **Upload Audio File** to select a pre-recorded file.

**Supported formats:** WAV, MP3, OGG (max 30 seconds)

**What you'll see:**
- The Na'vi text (what the system heard)
- The English translation
- A confidence score (how sure the system is)
- Processing time in milliseconds

### 2. Translate Text

**How to use:**
1. Click the **Translate Text** tab
2. Type Na'vi text in the input box (e.g., `kaltxì`)
3. Click **Translate** or press Enter
4. View the English translation and word-by-word breakdown

**Word breakdown:** Each Na'vi word is shown with its English meaning. Words highlighted in orange were not found in the dictionary.

### 3. Contribute a Word

**How to use:**
1. Click the **Contribute** tab
2. Enter the Na'vi word
3. Enter the English meaning
4. Optionally click **Record Pronunciation** to attach audio
5. Click **Submit Word**

Your submission will be reviewed and added to the training data. When enough new words accumulate, the system automatically retrains to learn them.

### 4. Pipeline Status

This tab shows:
- **API Health**: Whether the backend server is running
- **Model Readiness**: Whether the speech and translation models are loaded
- **Management Consoles**: Links to MLflow, Airflow, Prometheus, and Grafana

### 5. Help

Click the **?** button in the top-right corner at any time to open this user manual.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Microphone access denied" | Click the lock icon in the browser address bar and allow microphone access |
| "Backend not reachable" | Make sure `docker compose up` is running |
| "Models not loaded" | The server is still starting up — wait 30 seconds and refresh |
| Translation seems wrong | Try shorter phrases; the model works best with single words or short sentences |
| Audio upload rejected | Ensure the file is WAV, MP3, or OGG and under 30 seconds |

---

## Keyboard Shortcuts

- **Enter**: Submit text translation (in the text input box)
- **Shift+Enter**: New line in text input
- **Escape**: Close help modal
