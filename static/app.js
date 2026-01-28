const transcriptFeed = document.getElementById('transcript-feed');
const statusBadge = document.getElementById('status-badge');
const audioToggle = document.getElementById('audio-toggle');
const volLevel = document.getElementById('vol-level');
const volStatus = document.getElementById('vol-status');
const canvas = document.getElementById('visualizer');
const ctx = canvas.getContext('2d');
const latencyStat = document.getElementById('latency-stat');
const bufferStat = document.getElementById('buffer-stat');

let ws;
let audioCtx;
let isAudioEnabled = false;
let startTime = 0;
let speakers = new Set();
let msgCount = 0;

// On-demand playback state
let rawAudioHistory = []; // {timestamp: float, chunk: Float32Array}
let transcriptionHistory = []; // {id: string, text: string, speaker: string, audio: Float32Array}
let isPlaybackMuted = false;
let liveAudioEnabledBeforePlayback = false;
let activePlaybackSource = null;
let activePlaybackId = null;
let watchwords = [];

console.log("App initializing...");

// Initialize Audio Context on user gesture
audioToggle.addEventListener('click', () => {
    console.log("Audio toggle clicked");
    if (!audioCtx) {
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            console.log("AudioContext created at 16000Hz");
        } catch (e) {
            console.error("Failed to create AudioContext:", e);
        }
    }

    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }

    isAudioEnabled = !isAudioEnabled;
    audioToggle.classList.toggle('active', isAudioEnabled);
    audioToggle.innerHTML = isAudioEnabled ? '<span>🔊</span> Mute Audio' : '<span>🔇</span> Enable Audio';
});

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    console.log("Connecting to WebSocket:", wsUrl);

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log("WebSocket connected");
        statusBadge.textContent = 'Connected';
        statusBadge.classList.add('connected');
    };

    ws.onclose = (e) => {
        console.warn("WebSocket closed:", e.code, e.reason);
        statusBadge.textContent = 'Disconnected';
        statusBadge.classList.remove('connected');
        setTimeout(connect, 2000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };

    ws.onmessage = (event) => {
        msgCount++;
        if (msgCount % 100 === 0) console.log(`Received ${msgCount} messages`);

        if (typeof event.data === 'string') {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'transcript') {
                    console.log("Received transcript:", data.text);
                    addTranscriptItem(data);
                }
            } catch (e) {
                console.error("Error parsing JSON message:", e);
            }
        } else {
            // Binary audio data
            handleAudioData(event.data);
        }
    };
}

function handleAudioData(data) {
    const floatData = new Float32Array(data);

    // Update visualizer peak
    let peak = 0;
    for (let i = 0; i < floatData.length; i++) {
        const abs = Math.abs(floatData[i]);
        if (abs > peak) peak = abs;
    }
    drawVisualizer(peak);
    volLevel.textContent = peak.toFixed(4);
    volStatus.textContent = peak > 0.004 ? 'Active' : 'Quiet';

    // Store in history buffer (keep approx 30 seconds)
    const now = Date.now() / 1000;
    rawAudioHistory.push({ timestamp: now, chunk: floatData });

    // Prune raw buffer (120s window to handle long transcription latency)
    const windowSec = 120;
    while (rawAudioHistory.length > 0 && rawAudioHistory[0].timestamp < now - windowSec) {
        rawAudioHistory.shift();
    }

    if (msgCount % 20 === 0) {
        // Latency: Show the last known processing latency
        // Buffer: Show how many ms of audio are currently queued in the AudioContext
        if (audioCtx) {
            const bufferDepth = Math.max(0, startTime - audioCtx.currentTime);
            document.getElementById('buffer-stat').textContent = `${(bufferDepth * 1000).toFixed(0)}ms`;
        }
    }

    // Play if enabled and not currently playing back history
    if (isAudioEnabled && audioCtx && !isPlaybackMuted) {
        const buffer = audioCtx.createBuffer(1, floatData.length, 16000);
        buffer.getChannelData(0).set(floatData);

        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);

        // Schedule playback to avoid gaps
        const scheduleTime = Math.max(audioCtx.currentTime, startTime);
        source.start(scheduleTime);
        startTime = scheduleTime + buffer.duration;
    }
}

function checkWatchwords(text) {
    if (watchwords.length === 0) return false;
    const lowerText = text.toLowerCase();
    return watchwords.some(word => lowerText.includes(word.toLowerCase()));
}

function triggerNotification(text) {
    if (Notification.permission === "granted") {
        try {
            const notification = new Notification("Watchword Detected!", {
                body: text,
                icon: "/favicon.ico",
                tag: 'watchword-alert', // Prevents flooding
                renotify: true
            });

            notification.onclick = () => {
                window.focus();
                notification.close();
            };
        } catch (e) {
            console.error("Failed to show notification:", e);
        }
    }
}

function addTranscriptItem(data) {
    // Remove placeholder
    const placeholder = transcriptFeed.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'transcript-item';

    // Check for watchwords
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
        triggerNotification(data.text);
    }

    // Calculate true latency and extract segment audio
    if (data.origin_time) {
        const latency = (Date.now() / 1000) - data.origin_time;
        latencyStat.textContent = `${(latency * 1000).toFixed(0)}ms`;

        // Extract related audio from history
        // We look for chunks that happened AFTER origin_time - 1.0s (extra lead-in)
        const segmentChunks = rawAudioHistory
            .filter(item => item.timestamp >= data.origin_time - 1.0)
            .map(item => item.chunk);

        if (segmentChunks.length > 0) {
            // Concatenate
            const totalLength = segmentChunks.reduce((acc, chunk) => acc + chunk.length, 0);
            const combinedAudio = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of segmentChunks) {
                combinedAudio.set(chunk, offset);
                offset += chunk.length;
            }

            const itemId = `segment-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
            const historyItem = {
                id: itemId,
                speaker: data.speaker,
                text: data.text,
                audio: combinedAudio,
                timestamp: data.timestamp
            };

            transcriptionHistory.push(historyItem);

            // Add ID for playback lookup
            item.dataset.id = itemId;

            saveHistoryToLocal();
            pruneHistory();
        } else {
            console.warn(`No audio chunks found for segment starting at ${data.origin_time}. Buffer range: ${rawAudioHistory.length > 0 ? rawAudioHistory[0].timestamp : 'empty'} to ${rawAudioHistory.length > 0 ? rawAudioHistory[rawAudioHistory.length - 1].timestamp : 'empty'}`);
        }
    }

    speakers.add(data.speaker);
    document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot') ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="timestamp">${data.timestamp}</span>
                <div class="action-buttons">
                    ${item.dataset.id ? `<button class="play-btn" onclick="playSegment('${item.dataset.id}')" title="Play audio">▶️</button>` : ''}
                    ${item.dataset.id ? `<button class="download-btn" onclick="downloadSegment('${item.dataset.id}')" title="Download clip">📥</button>` : ''}
                </div>
            </div>
        </div>
        <div class="transcript-text">${data.text}</div>
    `;

    transcriptFeed.appendChild(item);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

function drawVisualizer(peak) {
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const barHeight = peak * height * 5;
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, '#38bdf8');
    gradient.addColorStop(1, '#818cf8');

    ctx.fillStyle = gradient;
    ctx.fillRect(10, height - barHeight, width - 20, barHeight);
}

function downloadSegment(id) {
    const item = transcriptionHistory.find(h => h.id === id);
    if (!item) return;

    const buffer = item.audio;
    const wavData = encodeWAV(buffer, 16000);
    const blob = new Blob([wavData], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `clip_${id.replace('segment-', '')}.wav`;
    a.click();
    URL.revokeObjectURL(url);
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);

    return view;
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function playSegment(id) {
    const item = transcriptionHistory.find(h => h.id === id);
    if (!item || !audioCtx) return;

    // 1. If we are clicking the SAME button that is already playing, STOP it.
    if (activePlaybackId === id) {
        if (activePlaybackSource) {
            activePlaybackSource.stop();
        }
        return; // The onended handler below will clean up UI and state
    }

    // 2. If something else is playing, stop it first
    if (activePlaybackSource) {
        try { activePlaybackSource.stop(); } catch (e) { }
    }

    console.log("Playing back segment:", id);

    // 3. Mute live audio
    if (!isPlaybackMuted) {
        liveAudioEnabledBeforePlayback = isAudioEnabled;
        isPlaybackMuted = true;
    }

    const btn = document.querySelector(`[data-id="${id}"] .play-btn`);
    if (btn) {
        btn.innerHTML = '⏸️';
        btn.classList.add('playing');
    }

    const buffer = audioCtx.createBuffer(1, item.audio.length, 16000);
    buffer.getChannelData(0).set(item.audio);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);

    activePlaybackSource = source;
    activePlaybackId = id;

    source.onended = () => {
        // Only clear global state if this was the active playback
        if (activePlaybackId === id) {
            activePlaybackSource = null;
            activePlaybackId = null;
            isPlaybackMuted = false;

            if (btn) {
                btn.innerHTML = '▶️';
                btn.classList.remove('playing');
            }
        }
    };

    source.start(0);
}

function pruneHistory() {
    const limitInput = document.getElementById('history-limit');
    const limit = parseInt(limitInput.value) || 10;

    // Save to localStorage
    localStorage.setItem('history-limit', limit);

    while (transcriptionHistory.length > limit) {
        const removed = transcriptionHistory.shift();
        // UI Clean up: Remove action buttons from the DOM for this segment
        const segmentEl = document.querySelector(`[data-id="${removed.id}"]`);
        if (segmentEl) {
            const actionBtns = segmentEl.querySelector('.action-buttons');
            if (actionBtns) actionBtns.remove();
        }
    }
    saveHistoryToLocal();
}

function saveHistoryToLocal() {
    // Only save the text/metadata, audio is too large for localStorage
    const historyToSave = transcriptionHistory.map(h => ({
        id: h.id,
        speaker: h.speaker,
        text: h.text,
        timestamp: h.timestamp
    }));
    localStorage.setItem('transcription-history', JSON.stringify(historyToSave));
}

function loadHistoryFromLocal() {
    const saved = localStorage.getItem('transcription-history');
    if (saved) {
        try {
            const items = JSON.parse(saved);
            items.forEach(item => {
                // Add to feed but without audio capability (since we can't save audio)
                renderHistoryItemIndividually(item);
            });
        } catch (e) {
            console.error("Error loading history:", e);
        }
    }
}

function renderHistoryItemIndividually(data) {
    const placeholder = transcriptFeed.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'transcript-item';

    // Check for watchwords (mostly for color)
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
    }

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot') ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="timestamp">${data.timestamp}</span>
                <span class="text-muted" style="font-size: 0.7rem; margin-left: 8px;">(Archive)</span>
            </div>
        </div>
        <div class="transcript-text">${data.text}</div>
    `;

    transcriptFeed.appendChild(item);
}

// Initial setup for history limit
const limitInput = document.getElementById('history-limit');
if (limitInput) {
    const savedLimit = localStorage.getItem('history-limit');
    if (savedLimit) {
        limitInput.value = savedLimit;
    } else {
        limitInput.value = 100; // New default
    }

    limitInput.addEventListener('input', pruneHistory);
}

// Watchword Management
function renderWatchwords() {
    const list = document.getElementById('watchwords-list');
    list.innerHTML = '';
    watchwords.forEach((word, index) => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `
            ${word}
            <span class="remove" onclick="removeWatchword(${index})">×</span>
        `;
        list.appendChild(tag);
    });
}

function addWatchword() {
    const input = document.getElementById('watchword-input');
    const word = input.value.trim();
    if (word && !watchwords.includes(word)) {
        watchwords.push(word);
        localStorage.setItem('watchwords', JSON.stringify(watchwords));
        renderWatchwords();
        input.value = '';
    }
}

function removeWatchword(index) {
    watchwords.splice(index, 1);
    localStorage.setItem('watchwords', JSON.stringify(watchwords));
    renderWatchwords();
}

function clearWatchwords() {
    watchwords = [];
    localStorage.removeItem('watchwords');
    renderWatchwords();
}

// Event Listeners for Watchwords
document.getElementById('add-watchword').addEventListener('click', addWatchword);
document.getElementById('watchword-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addWatchword();
});
document.getElementById('clear-watchwords').addEventListener('click', clearWatchwords);

// Notification Management
const notificationBtn = document.getElementById('enable-notifications');

function updateNotificationButton() {
    if (!notificationBtn) return;

    if (Notification.permission === "granted") {
        notificationBtn.style.display = 'none';
    } else {
        notificationBtn.style.display = 'flex';
        if (Notification.permission === "denied") {
            notificationBtn.innerHTML = '⚠️ Notifications Blocked';
            notificationBtn.style.opacity = '0.6';
            notificationBtn.style.cursor = 'not-allowed';
            notificationBtn.title = 'Please enable notifications in your browser settings to receive alerts.';
        }
    }
}

if (notificationBtn) {
    notificationBtn.addEventListener('click', async () => {
        if (Notification.permission === 'denied') {
            alert("Notifications are blocked by your browser. Please enable them in your browser settings (usually in the address bar).");
            return;
        }

        const permission = await Notification.requestPermission();
        updateNotificationButton();

        if (permission === 'granted') {
            new Notification("Notifications Enabled!", {
                body: "You will now receive alerts for watchwords.",
                icon: "/favicon.ico"
            });
        }
    });
}

// Initialization
const savedWatchwords = localStorage.getItem('watchwords');
if (savedWatchwords) {
    watchwords = JSON.parse(savedWatchwords);
    renderWatchwords();
}

loadHistoryFromLocal();
updateNotificationButton();

connect();
