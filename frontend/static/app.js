/* =============================================================================
   Na'vi Translator — Frontend JavaScript
   =============================================================================
   All API calls go through /api/* which nginx proxies to the backend.
   This keeps the frontend and backend strictly decoupled.
   ============================================================================= */

const API_BASE = '/api';

// --- Tab Switching ---
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Refresh health status when switching to pipeline tab
    if (tabName === 'pipeline') {
        checkHealth();
        checkReady();
    }
}

// --- Help Modal ---
function toggleHelp() {
    document.getElementById('help-modal').classList.toggle('hidden');
}

// Close modal on outside click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('help-modal');
    if (e.target === modal) modal.classList.add('hidden');
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') document.getElementById('help-modal').classList.add('hidden');
});


// =============================================================================
// Audio Recording (uses MediaRecorder API)
// =============================================================================
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });
            stream.getTracks().forEach(t => t.stop());
            translateAudio(blob);
        };

        mediaRecorder.start();
        isRecording = true;
        updateRecordingUI(true);
    } catch (err) {
        showError('audio-error', 'Microphone access denied. Please allow microphone access.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    updateRecordingUI(false);
}

function updateRecordingUI(recording) {
    const btn = document.getElementById('record-btn');
    const label = document.getElementById('record-label');
    const indicator = document.getElementById('recording-indicator');

    if (recording) {
        btn.classList.add('recording');
        label.textContent = 'Stop Recording';
        indicator.classList.remove('hidden');
    } else {
        btn.classList.remove('recording');
        label.textContent = 'Start Recording';
        indicator.classList.add('hidden');
    }
}

function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    translateAudio(file);
}


// =============================================================================
// Translation API Calls
// =============================================================================

async function translateAudio(audioBlob) {
    hideError('audio-error');
    hideResult('audio-result');

    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    try {
        const response = await fetch(`${API_BASE}/translate/audio`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail?.detail || err.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('audio-navi').textContent = data.navi_text;
        document.getElementById('audio-english').textContent = data.english;
        document.getElementById('audio-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('audio-latency').textContent = data.latency_ms;
        showResult('audio-result');
    } catch (err) {
        showError('audio-error', `Translation failed: ${err.message}`);
    }
}

async function translateText() {
    const text = document.getElementById('navi-input').value.trim();
    if (!text) return;

    hideError('text-error');
    hideResult('text-result');

    try {
        const response = await fetch(`${API_BASE}/translate/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail?.detail || err.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('text-english').textContent = data.english;
        document.getElementById('text-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('text-latency').textContent = data.latency_ms;

        // Render word breakdown
        const grid = document.getElementById('word-breakdown');
        grid.innerHTML = '';
        if (data.word_breakdown && data.word_breakdown.length > 0) {
            data.word_breakdown.forEach(w => {
                const card = document.createElement('div');
                card.className = `word-card ${w.found ? '' : 'not-found'}`;
                card.innerHTML = `
                    <div class="navi">${escapeHtml(w.navi)}</div>
                    <div class="en">${escapeHtml(w.en)}</div>
                `;
                grid.appendChild(card);
            });
        }

        showResult('text-result');
    } catch (err) {
        showError('text-error', `Translation failed: ${err.message}`);
    }
}

// Allow Enter key to submit (Shift+Enter for newline)
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('navi-input');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                translateText();
            }
        });
    }
    // Initial health check
    checkHealth();
    checkReady();
});


// =============================================================================
// Vocabulary Contribution
// =============================================================================

let contribAudioBlob = null;
let contribRecorder = null;
let contribRecording = false;

async function toggleContribRecording() {
    if (contribRecording) {
        contribRecorder.stop();
        contribRecording = false;
        document.getElementById('contrib-record-btn').textContent = 'Record Pronunciation';
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            contribRecorder = new MediaRecorder(stream);
            const chunks = [];

            contribRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunks.push(e.data);
            };

            contribRecorder.onstop = () => {
                contribAudioBlob = new Blob(chunks, { type: 'audio/wav' });
                stream.getTracks().forEach(t => t.stop());
                document.getElementById('contrib-audio-status').textContent = 'Audio recorded';
            };

            contribRecorder.start();
            contribRecording = true;
            document.getElementById('contrib-record-btn').textContent = 'Stop Recording';
            document.getElementById('contrib-audio-status').textContent = 'Recording...';
        } catch (err) {
            showError('contrib-error', 'Microphone access denied.');
        }
    }
}

async function submitVocab(event) {
    event.preventDefault();
    hideError('contrib-error');
    hideResult('contrib-result');

    const naviWord = document.getElementById('contrib-navi').value.trim();
    const enMeaning = document.getElementById('contrib-en').value.trim();

    const body = {
        navi_word: naviWord,
        english_meaning: enMeaning,
        audio_b64: null,
    };

    // Convert audio to base64 if recorded
    if (contribAudioBlob) {
        const buffer = await contribAudioBlob.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        let binary = '';
        bytes.forEach(b => binary += String.fromCharCode(b));
        body.audio_b64 = btoa(binary);
    }

    try {
        const response = await fetch(`${API_BASE}/vocab/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await response.json();
        const resultEl = document.getElementById('contrib-result');
        resultEl.textContent = data.message;
        resultEl.style.color = data.accepted ? 'var(--success)' : 'var(--warning)';
        showResult('contrib-result');

        if (data.accepted) {
            document.getElementById('contribute-form').reset();
            contribAudioBlob = null;
            document.getElementById('contrib-audio-status').textContent = '';
        }
    } catch (err) {
        showError('contrib-error', `Submission failed: ${err.message}`);
    }
}


// =============================================================================
// Pipeline Status
// =============================================================================

async function checkHealth() {
    const statusEl = document.getElementById('health-status');
    const detailsEl = document.getElementById('health-details');
    statusEl.textContent = 'Checking...';
    statusEl.className = 'status-value loading';

    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        statusEl.textContent = data.status === 'ok' ? 'Healthy' : data.status;
        statusEl.className = `status-value ${data.status === 'ok' ? 'ok' : 'error'}`;
        detailsEl.textContent = `Model: ${data.model_version} | Uptime: ${Math.round(data.uptime_s)}s`;
    } catch {
        statusEl.textContent = 'Offline';
        statusEl.className = 'status-value error';
        detailsEl.textContent = 'Backend not reachable';
    }
}

async function checkReady() {
    const statusEl = document.getElementById('ready-status');
    const detailsEl = document.getElementById('ready-details');
    statusEl.textContent = 'Checking...';
    statusEl.className = 'status-value loading';

    try {
        const response = await fetch(`${API_BASE}/ready`);
        const data = await response.json();
        statusEl.textContent = data.ready ? 'Ready' : 'Not Ready';
        statusEl.className = `status-value ${data.ready ? 'ok' : 'error'}`;
        detailsEl.textContent = `Whisper: ${data.whisper_loaded ? 'loaded' : 'not loaded'} | MarianMT: ${data.marian_loaded ? 'loaded' : 'not loaded'}`;
    } catch {
        statusEl.textContent = 'Offline';
        statusEl.className = 'status-value error';
        detailsEl.textContent = '';
    }
}


// =============================================================================
// Utilities
// =============================================================================

function showResult(id) {
    document.getElementById(id).classList.remove('hidden');
}

function hideResult(id) {
    document.getElementById(id).classList.add('hidden');
}

function showError(id, message) {
    const el = document.getElementById(id);
    el.textContent = message;
    el.classList.remove('hidden');
}

function hideError(id) {
    document.getElementById(id).classList.add('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
