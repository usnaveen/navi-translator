const API_BASE = '/api';

const TAB_META = {
    audio: {
        chip: 'AUDIO MODE',
        title: 'AUDIO TRANSLATION',
        copy: 'Capture spoken Na\'vi and decode it into clear English.',
        footer: 'FIELD CONSOLE NOMINAL'
    },
    text: {
        chip: 'TEXT MODE',
        title: 'TEXT TRANSLATION',
        copy: 'Inspect written Na\'vi with a clean English render and lexical breakdown.',
        footer: 'TEXT CHANNEL LOCKED'
    },
    contribute: {
        chip: 'INTAKE MODE',
        title: 'LEXICON INTAKE',
        copy: 'Feed new vocabulary into the review queue and grow the translator corpus.',
        footer: 'REVIEW QUEUE READY'
    },
    pipeline: {
        chip: 'OPS MODE',
        title: 'PIPELINE STATUS',
        copy: 'Monitor runtime health, model readiness, and the surrounding MLOps stack.',
        footer: 'OPERATIONS SURFACE ACTIVE'
    }
};

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    syncTabMeta(tabName);

    if (tabName === 'pipeline') {
        checkHealth();
        checkReady();
    }
}

function syncTabMeta(tabName) {
    const meta = TAB_META[tabName];
    if (!meta) return;

    setText('active-mode-chip', meta.chip);
    setText('hero-mode-display', meta.title);
    setText('hero-mode-copy', meta.copy);
    document.body.dataset.activeTab = tabName;
}

function toggleHelp() {
    document.getElementById('help-modal').classList.toggle('hidden');
}

document.addEventListener('click', (event) => {
    const modal = document.getElementById('help-modal');
    if (event.target === modal) modal.classList.add('hidden');
});

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        document.getElementById('help-modal').classList.add('hidden');
    }
});

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// --- Waveform visualizer ---
let audioContext = null;
let analyser = null;
let waveformAnimationId = null;
let activeStream = null;
let staticWaveformData = null;  // Float32Array of decoded audio samples for uploaded files

function getCanvasCtx() {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas) return null;
    // Match canvas pixel size to its CSS size for crispness
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.floor(rect.width) || canvas.height !== Math.floor(rect.height)) {
        canvas.width = Math.floor(rect.width);
        canvas.height = Math.floor(rect.height);
    }
    return { canvas, ctx: canvas.getContext('2d') };
}

function drawIdleWaveform() {
    const c = getCanvasCtx();
    if (!c) return;
    const { canvas, ctx } = c;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(78, 220, 200, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

function drawLiveWaveform() {
    const c = getCanvasCtx();
    if (!c || !analyser) return;
    const { canvas, ctx } = c;
    const bufferLength = analyser.fftSize;
    const data = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(data);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, '#62f4d2');
    grad.addColorStop(1, '#3a8ce0');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
        const v = data[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    waveformAnimationId = requestAnimationFrame(drawLiveWaveform);
}

function drawStaticWaveform(samples) {
    const c = getCanvasCtx();
    if (!c) return;
    const { canvas, ctx } = c;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const step = Math.max(1, Math.floor(samples.length / canvas.width));
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, '#62f4d2');
    grad.addColorStop(1, '#3a8ce0');
    ctx.fillStyle = grad;
    const mid = canvas.height / 2;
    for (let x = 0; x < canvas.width; x++) {
        let min = 1.0, max = -1.0;
        for (let i = 0; i < step; i++) {
            const v = samples[x * step + i] || 0;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const y1 = mid + min * mid;
        const y2 = mid + max * mid;
        ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
}

function startVisualizer(stream) {
    audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    drawLiveWaveform();
}

function stopVisualizer() {
    if (waveformAnimationId) {
        cancelAnimationFrame(waveformAnimationId);
        waveformAnimationId = null;
    }
    analyser = null;
}

async function decodeBlobToSamples(blob) {
    audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuf = await blob.arrayBuffer();
    try {
        const decoded = await audioContext.decodeAudioData(arrayBuf.slice(0));
        return decoded.getChannelData(0);
    } catch {
        return null;
    }
}

function setPlaybackSource(blob) {
    const audio = document.getElementById('audio-playback');
    if (!audio) return;
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(blob);
    audio.classList.remove('hidden');
}

window.addEventListener('DOMContentLoaded', () => {
    drawIdleWaveform();
    window.addEventListener('resize', () => {
        if (staticWaveformData) drawStaticWaveform(staticWaveformData);
        else if (!analyser) drawIdleWaveform();
    });
});

// --- Recording flow ---
function pickRecorderMimeType() {
    const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/mp4',
        'audio/ogg;codecs=opus',
    ];
    for (const t of candidates) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
}

async function convertToWav(blob, targetSampleRate = 16000) {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const tmpCtx = new AudioCtx();
    const decoded = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
    tmpCtx.close();

    const length = Math.ceil(decoded.duration * targetSampleRate);
    const offline = new OfflineAudioContext(1, length, targetSampleRate);
    const monoBuffer = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    const monoData = monoBuffer.getChannelData(0);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        const chData = decoded.getChannelData(ch);
        for (let i = 0; i < chData.length; i++) {
            monoData[i] += chData[i] / decoded.numberOfChannels;
        }
    }
    const src = offline.createBufferSource();
    src.buffer = monoBuffer;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return encodeWav(rendered.getChannelData(0), targetSampleRate);
}

function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    hideError('audio-error');
    staticWaveformData = null;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        activeStream = stream;
        const mimeType = pickRecorderMimeType();
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            stream.getTracks().forEach((track) => track.stop());
            activeStream = null;
            stopVisualizer();
            setText('audio-action-status', 'Signal captured. Converting to WAV...');
            try {
                const wavBlob = await convertToWav(recordedBlob, 16000);
                setPlaybackSource(wavBlob);
                const samples = await decodeBlobToSamples(wavBlob);
                if (samples) { staticWaveformData = samples; drawStaticWaveform(samples); }
                else drawIdleWaveform();
                translateAudio(wavBlob);
            } catch (e) {
                showError('audio-error', 'Audio conversion failed: ' + (e.message || e));
                setText('audio-action-status', 'Audio conversion failed.');
            }
        };

        mediaRecorder.start();
        startVisualizer(stream);
        isRecording = true;
        updateRecordingUI(true);
    } catch {
        showError('audio-error', 'Microphone access denied. Please allow microphone access.');
        setText('audio-action-status', 'Microphone unavailable.');
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
    const button = document.getElementById('record-btn');
    const indicator = document.getElementById('recording-indicator');
    if (button) {
        button.classList.toggle('recording', recording);
        button.title = recording ? 'Stop Capture' : 'Arm Recorder';
    }
    if (indicator) indicator.classList.toggle('hidden', !recording);
    setText('audio-action-status', recording ? "Listening for Na'vi speech..." : 'Microphone standing by.');
}

async function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    setText('upload-status', `Loaded sample: ${file.name}`);
    setText('audio-action-status', 'Audio file loaded. Decoding waveform...');
    setPlaybackSource(file);
    const samples = await decodeBlobToSamples(file);
    if (samples) { staticWaveformData = samples; drawStaticWaveform(samples); }
    translateAudio(file);
}

async function translateAudio(audioBlob) {
    hideError('audio-error');
    hideResult('audio-result');
    setButtonDisabled('record-btn', true);

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
        setText('audio-navi', data.navi_text);
        const corrected = data.navi_corrected || data.navi_text;
        setText('audio-corrected', corrected === data.navi_text ? '(no correction needed)' : corrected);
        setText('audio-english', data.english);
        setText('audio-confidence', `${(data.confidence * 100).toFixed(1)}%`);
        setText('audio-asr-conf', `${((data.asr_confidence ?? 0) * 100).toFixed(1)}%`);
        setText('audio-nmt-conf', `${((data.nmt_confidence ?? 0) * 100).toFixed(1)}%`);
        setText('audio-fuzzy-count', `${data.fuzzy_corrections ?? 0}`);
        setText('audio-latency', data.latency_ms);
        setText('audio-action-status', 'Translation locked. Readout updated.');
        showResult('audio-result');
    } catch (err) {
        showError('audio-error', `Translation failed: ${getErrorMessage(err)}`);
        setText('audio-action-status', 'Capture failed. Check the input and try again.');
    } finally {
        setButtonDisabled('record-btn', false);
    }
}

async function translateText() {
    const text = document.getElementById('navi-input').value.trim();
    if (!text) return;

    hideError('text-error');
    hideResult('text-result');
    setBusyLabel('text-translate-btn', 'Working...');

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
        setText('text-english', data.english);
        setText('text-confidence', `${(data.confidence * 100).toFixed(1)}%`);
        setText('text-latency', data.latency_ms);

        const grid = document.getElementById('word-breakdown');
        grid.innerHTML = '';
        if (data.word_breakdown && data.word_breakdown.length > 0) {
            data.word_breakdown.forEach((word) => {
                const card = document.createElement('div');
                card.className = `word-card ${word.found ? '' : 'not-found'}`;
                card.innerHTML = `
                    <div class="navi">${escapeHtml(word.navi)}</div>
                    <div class="en">${escapeHtml(word.en)}</div>
                `;
                grid.appendChild(card);
            });
        }

        showResult('text-result');
    } catch (err) {
        showError('text-error', `Translation failed: ${getErrorMessage(err)}`);
    } finally {
        resetBusyLabel('text-translate-btn', '&gt;', 'Translate Signal');
    }
}

function setTextExample(text) {
    const input = document.getElementById('navi-input');
    input.value = text;
    input.focus();
}

let contribAudioBlob = null;
let contribRecorder = null;
let contribRecording = false;

async function toggleContribRecording() {
    hideError('contrib-error');

    if (contribRecording) {
        contribRecorder.stop();
        contribRecording = false;
        setInlineButton('contrib-record-btn', 'REC', 'Record Sample');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = pickRecorderMimeType();
        contribRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        const chunks = [];

        contribRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) chunks.push(event.data);
        };

        contribRecorder.onstop = async () => {
            const raw = new Blob(chunks, { type: contribRecorder.mimeType || 'audio/webm' });
            stream.getTracks().forEach((track) => track.stop());
            try {
                contribAudioBlob = await convertToWav(raw, 16000);
                setText('contrib-audio-status', 'Pronunciation sample captured.');
            } catch (e) {
                showError('contrib-error', 'Audio conversion failed: ' + (e.message || e));
                setText('contrib-audio-status', 'Audio conversion failed.');
            }
        };

        contribRecorder.start();
        contribRecording = true;
        setInlineButton('contrib-record-btn', 'STOP', 'Stop Capture');
        setText('contrib-audio-status', 'Recording pronunciation...');
    } catch {
        showError('contrib-error', 'Microphone access denied.');
        setText('contrib-audio-status', 'Microphone unavailable.');
    }
}

async function submitVocab(event) {
    event.preventDefault();
    hideError('contrib-error');
    hideResult('contrib-result');
    setBusyLabel('contrib-submit-btn', 'Submitting...');

    const naviWord = document.getElementById('contrib-navi').value.trim();
    const enMeaning = document.getElementById('contrib-en').value.trim();

    const body = {
        navi_word: naviWord,
        english_meaning: enMeaning,
        audio_b64: null,
    };

    if (contribAudioBlob) {
        const buffer = await contribAudioBlob.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        let binary = '';
        bytes.forEach((byte) => {
            binary += String.fromCharCode(byte);
        });
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
        resultEl.style.color = data.accepted ? '#60ff94' : '#ffd07a';
        showResult('contrib-result');

        if (data.accepted) {
            document.getElementById('contribute-form').reset();
            contribAudioBlob = null;
            setText('contrib-audio-status', 'Optional field recording.');
        }
    } catch (err) {
        showError('contrib-error', `Submission failed: ${getErrorMessage(err)}`);
    } finally {
        resetBusyLabel('contrib-submit-btn', '+', 'Submit to Review');
    }
}

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

function showResult(id) {
    const result = document.getElementById(id);
    const placeholderId = result.dataset.placeholder;
    result.classList.remove('hidden');
    if (placeholderId) {
        document.getElementById(placeholderId).classList.add('hidden');
    }
}

function hideResult(id) {
    const result = document.getElementById(id);
    const placeholderId = result.dataset.placeholder;
    result.classList.add('hidden');
    if (placeholderId) {
        document.getElementById(placeholderId).classList.remove('hidden');
    }
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

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setButtonDisabled(id, disabled) {
    document.getElementById(id).disabled = disabled;
}

function setBusyLabel(id, text) {
    const button = document.getElementById(id);
    button.disabled = true;
    button.innerHTML = `<span>${text}</span>`;
}

function resetBusyLabel(id, code, text) {
    const button = document.getElementById(id);
    button.disabled = false;
    button.innerHTML = `<span class="btn-code">${code}</span><span>${text}</span>`;
}

function setInlineButton(id, code, text) {
    document.getElementById(id).innerHTML = `<span class="btn-code">${code}</span><span>${text}</span>`;
}

function getErrorMessage(err) {
    return err instanceof Error ? err.message : 'Unexpected error';
}

document.addEventListener('DOMContentLoaded', () => {
    syncTabMeta('audio');

    const input = document.getElementById('navi-input');
    if (input) {
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                translateText();
            }
        });
    }

    checkHealth();
    checkReady();
});
