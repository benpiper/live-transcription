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

function ensureAudioContext() {
    if (!audioCtx) {
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            console.log("AudioContext created at 16000Hz");

            // Create AnalyserNode for spectrum
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 64; // Small size for 32 clean bars
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);

            // Start visualization loop
            requestAnimationFrame(drawVisualizer);
        } catch (e) {
            console.error("Failed to create AudioContext:", e);
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

// On-demand playback state
let rawAudioHistory = []; // {timestamp: float, chunk: Float32Array}
let transcriptionHistory = []; // {id: string, text: string, speaker: string, audio: Float32Array}
let isPlaybackMuted = false;
let liveAudioEnabledBeforePlayback = false;
let activePlaybackSource = null;
let activePlaybackId = null;
let activeSpeakerFilter = null;
let watchwords = [];
let theme = 'dark';
let isScrollLocked = false;  // When true, don't auto-scroll on new transcripts

// Analysis node for spectrum visualization
let analyser;
let dataArray;

console.log("App initializing...");

// IndexedDB Setup for persistent audio
let db;
const dbRequest = indexedDB.open("TranscriptionDB", 2);

dbRequest.onupgradeneeded = (event) => {
    const db = event.target.result;
    if (!db.objectStoreNames.contains("audioStore")) {
        db.createObjectStore("audioStore");
    }
};

dbRequest.onsuccess = (event) => {
    db = event.target.result;
    console.log("IndexedDB initialized");
    // Load history after DB is ready
    loadHistoryFromLocal();
};

dbRequest.onerror = (event) => {
    console.error("IndexedDB error:", event.target.error);
    // Fallback: Load text-only history if DB fails
    loadHistoryFromLocal();
};

// Initialize Audio Context on user gesture
audioToggle.addEventListener('click', () => {
    console.log("Audio toggle clicked");
    ensureAudioContext();

    isAudioEnabled = !isAudioEnabled;
    audioToggle.classList.toggle('active', isAudioEnabled);
    audioToggle.innerHTML = isAudioEnabled ? '<span>🔊</span> Mute Audio' : '<span>🔇</span> Enable Audio';
});

// Load current session transcripts from server
async function loadCurrentSession() {
    try {
        const response = await fetch('/api/session/current');
        const data = await response.json();

        if (data.active && data.transcripts && data.transcripts.length > 0) {
            console.log(`Loading ${data.transcripts.length} transcripts from session: ${data.name}`);

            // Clear existing transcripts to avoid duplicates
            transcriptFeed.innerHTML = '';
            transcriptionHistory = [];

            // Batch DOM updates using DocumentFragment
            const fragment = document.createDocumentFragment();

            for (const t of data.transcripts) {
                const element = createTranscriptElement(t, true);  // true = from session
                fragment.appendChild(element);

                // Track speakers (but don't render filters yet)
                const itemSpeaker = t.speaker || 'Unknown';
                speakers.add(itemSpeaker);
            }

            // Single DOM update
            transcriptFeed.appendChild(fragment);

            // Render speaker filters once after all items loaded
            renderSpeakerFilters();
            document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;

            // Scroll to bottom
            transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

            console.log(`Loaded ${data.transcripts.length} transcripts from session`);
        }
    } catch (err) {
        console.log("No active session or error loading:", err);
    }
}

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

        // Load current session transcripts
        loadCurrentSession();
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
                } else if (data.type === 'volume') {
                    // Update visualizer peak even during streaming silence
                    drawVisualizer(data.peak);
                    volLevel.textContent = data.peak.toFixed(4);
                    volStatus.textContent = data.peak > 0.004 ? 'Active' : 'Quiet';
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

    // Update peak for volume stat (legacy use)
    let peak = 0;
    for (let i = 0; i < floatData.length; i++) {
        const abs = Math.abs(floatData[i]);
        if (abs > peak) peak = abs;
    }
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

        // Connect to both output and analyser
        source.connect(audioCtx.destination);
        if (analyser) source.connect(analyser);

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

function highlightWatchwords(text) {
    if (watchwords.length === 0) return text;
    let result = text;
    for (const word of watchwords) {
        const regex = new RegExp(`(${word})`, 'gi');
        result = result.replace(regex, '<mark class="watchword-highlight">$1</mark>');
    }
    return result;
}

function triggerNotification(text) {
    if (!("Notification" in window)) return;

    if (Notification.permission === "granted") {
        try {
            const notification = new Notification("Watchword Detected!", {
                body: text,
                tag: 'watchword-alert',
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

// Creates a transcript DOM element without appending it (for batch operations)
function createTranscriptElement(data, fromSession = false) {
    const item = document.createElement('div');
    item.className = 'transcript-item';

    // Check for watchwords
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
        if (!fromSession) triggerNotification(data.text);
    }

    const itemSpeaker = data.speaker || 'Unknown';
    item.dataset.speaker = itemSpeaker;

    // Apply active filter if necessary
    if (activeSpeakerFilter && itemSpeaker !== activeSpeakerFilter) {
        item.classList.add('filtered-out');
    }

    // Confidence styling
    const confidence = data.confidence || 0;
    let confClass = 'conf-high';
    if (confidence < -0.7) confClass = 'conf-low';
    else if (confidence < -0.6) confClass = 'conf-med';

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker && (data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot')) ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="confidence ${confClass}" title="Whisper Log Probability (closer to 0 is better)">${confidence.toFixed(2)}</span>
                <span class="timestamp">${data.timestamp}${data.duration ? ` (${data.duration.toFixed(1)}s)` : ''}</span>
                <div class="action-buttons">
                    ${item.dataset.id ? `<button class="play-btn" onclick="playSegment('${item.dataset.id}')" title="Play audio">▶️</button>` : ''}
                    ${item.dataset.id ? `<button class="download-btn" onclick="downloadSegment('${item.dataset.id}')" title="Download clip">📥</button>` : ''}
                </div>
            </div>
        </div>
        <div class="transcript-text">${highlightWatchwords(data.text)}</div>
    `;

    return item;
}

function addTranscriptItem(data, fromSession = false) {
    // Remove placeholder
    const placeholder = transcriptFeed.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'transcript-item';

    // Check for watchwords
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
        if (!fromSession) triggerNotification(data.text);  // Don't notify for old items
    }

    // Calculate true latency and extract segment audio (skip for session-loaded items)
    if (data.origin_time && !fromSession) {
        const latency = (Date.now() / 1000) - data.origin_time;
        latencyStat.textContent = `${(latency * 1000).toFixed(0)}ms`;

        // Extract related audio from history with proper boundaries
        // Start: origin_time - 0.5s (small lead-in)
        // End: current time (when transcript arrived)
        const segmentStart = data.origin_time - 0.5;
        const segmentEnd = Date.now() / 1000;  // Current time as end boundary

        const segmentChunks = rawAudioHistory
            .filter(item => item.timestamp >= segmentStart && item.timestamp <= segmentEnd)
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
                timestamp: data.timestamp,
                origin_time: data.origin_time,
                duration: data.duration,
                confidence: data.confidence
            };

            transcriptionHistory.push(historyItem);

            // Add ID for playback lookup
            item.dataset.id = itemId;

            saveAudioToDB(itemId, combinedAudio);
            saveHistoryToLocal();
            pruneHistory();
        } else {
            console.warn(`No audio chunks found for segment ${segmentStart.toFixed(2)} - ${segmentEnd.toFixed(2)}. Buffer range: ${rawAudioHistory.length > 0 ? rawAudioHistory[0].timestamp.toFixed(2) : 'empty'} to ${rawAudioHistory.length > 0 ? rawAudioHistory[rawAudioHistory.length - 1].timestamp.toFixed(2) : 'empty'}`);
        }
    }

    const itemSpeaker = data.speaker || 'Unknown';
    item.dataset.speaker = itemSpeaker;
    speakers.add(itemSpeaker);
    renderSpeakerFilters();
    document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;

    // Apply active filter if necessary
    if (activeSpeakerFilter && itemSpeaker !== activeSpeakerFilter) {
        item.classList.add('filtered-out');
    }

    // Confidence styling
    const confidence = data.confidence || 0;
    let confClass = 'conf-high';
    if (confidence < -0.7) confClass = 'conf-low';
    else if (confidence < -0.6) confClass = 'conf-med';

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker && (data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot')) ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="confidence ${confClass}" title="Whisper Log Probability (closer to 0 is better)">${confidence.toFixed(2)}</span>
                <span class="timestamp">${data.timestamp}${data.duration ? ` (${data.duration.toFixed(1)}s)` : ''}</span>
                <div class="action-buttons">
                    ${item.dataset.id ? `<button class="play-btn" onclick="playSegment('${item.dataset.id}')" title="Play audio">▶️</button>` : ''}
                    ${item.dataset.id ? `<button class="download-btn" onclick="downloadSegment('${item.dataset.id}')" title="Download clip">📥</button>` : ''}
                </div>
            </div>
        </div>
        <div class="transcript-text">${highlightWatchwords(data.text)}</div>
    `;

    transcriptFeed.appendChild(item);
    if (!isScrollLocked) {
        transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
    }
}

function drawVisualizer() {
    if (!analyser || !ctx) return;

    requestAnimationFrame(drawVisualizer);

    const width = canvas.width;
    const height = canvas.height;

    analyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, width, height);

    const barWidth = (width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#38bdf8';
    const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#818cf8';

    for (let i = 0; i < dataArray.length; i++) {
        barHeight = (dataArray[i] / 255) * height;

        const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
        gradient.addColorStop(0, primaryColor);
        gradient.addColorStop(1, accentColor);

        ctx.fillStyle = gradient;
        ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);

        x += barWidth + 1;
    }
}

// Speaker Filtering
function renderSpeakerFilters() {
    const filterContainer = document.getElementById('speaker-filters');
    if (!filterContainer) return;

    // Preserve existing tags, only add new ones
    const currentTags = Array.from(filterContainer.querySelectorAll('.tag')).map(t => t.dataset.label);

    speakers.forEach(speaker => {
        if (!currentTags.includes(speaker)) {
            const tag = document.createElement('div');
            tag.className = 'tag';
            tag.dataset.label = speaker;
            tag.textContent = speaker;
            tag.onclick = () => applySpeakerFilter(speaker);
            filterContainer.appendChild(tag);
        }
    });
}

function applySpeakerFilter(label) {
    activeSpeakerFilter = label;

    // Update UI highlights
    document.querySelectorAll('#speaker-filters .tag').forEach(tag => {
        tag.classList.toggle('active', tag.dataset.label === label);
    });

    // Filter feed
    document.querySelectorAll('.transcript-item').forEach(item => {
        const itemSpeaker = item.dataset.speaker || 'Unknown';

        if (!label || itemSpeaker === label) {
            item.classList.remove('filtered-out');
        } else {
            item.classList.add('filtered-out');
        }
    });

    console.log(`Speaker filter applied: ${label || 'Show All'}`);

    // Smooth scroll to bottom after layout change (respect scroll lock)
    if (!isScrollLocked) {
        transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
    }
}

document.getElementById('reset-filters').addEventListener('click', () => applySpeakerFilter(null));

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
    if (!item || !item.audio) {
        console.warn("No audio data for segment:", id);
        return;
    }

    // Ensure audio context is ready
    const context = ensureAudioContext();
    if (!context) return;

    // 1. If we are clicking the SAME button that is already playing, STOP it.
    if (activePlaybackId === id) {
        if (activePlaybackSource) {
            try { activePlaybackSource.stop(); } catch (e) { }
        }
        return; // The onended handler below will clean up UI and state
    }

    // 2. If something else is playing, stop it first
    if (activePlaybackSource) {
        try {
            // Manually reset the UI for the previously playing clip
            const prevId = activePlaybackId;
            const prevBtn = document.querySelector(`[data-id="${prevId}"] .play-btn`);
            if (prevBtn) {
                prevBtn.innerHTML = '▶️';
                prevBtn.classList.remove('playing');
            }
            activePlaybackSource.stop();
        } catch (e) { }
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

    const buffer = context.createBuffer(1, item.audio.length, 16000);
    buffer.getChannelData(0).set(item.audio);

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);

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

function saveAudioToDB(id, audioData) {
    if (!db) return;
    const transaction = db.transaction(["audioStore"], "readwrite");
    const store = transaction.objectStore("audioStore");
    store.put(audioData, id);
}

function getAudioFromDB(id) {
    return new Promise((resolve) => {
        if (!db) return resolve(null);
        const transaction = db.transaction(["audioStore"], "readonly");
        const store = transaction.objectStore("audioStore");
        const request = store.get(id);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });
}

function deleteAudioFromDB(id) {
    if (!db) return;
    const transaction = db.transaction(["audioStore"], "readwrite");
    const store = transaction.objectStore("audioStore");
    store.delete(id);
}

function pruneHistory() {
    // Use ONLY the saved limit from localStorage, not the input field
    const savedLimit = localStorage.getItem('history-limit');
    const limit = savedLimit ? parseInt(savedLimit) : 60;

    while (transcriptionHistory.length > limit) {
        const removed = transcriptionHistory.shift();
        // Delete audio from IndexedDB
        deleteAudioFromDB(removed.id);

        // UI Clean up
        const segmentEl = document.querySelector(`[data-id="${removed.id}"]`);
        if (segmentEl) {
            segmentEl.remove();
        }
    }
    saveHistoryToLocal();
}

function applyHistoryLimit() {
    // Read from input, save to localStorage, then prune
    const limitInput = document.getElementById('history-limit');
    if (!limitInput) return;
    const limit = parseInt(limitInput.value) || 60;

    // Save to localStorage
    localStorage.setItem('history-limit', limit);

    // Now prune using the newly saved value
    pruneHistory();
}


function saveHistoryToLocal() {
    // Only save the text/metadata, audio is in IndexedDB
    const historyToSave = transcriptionHistory.map(h => ({
        id: h.id,
        speaker: h.speaker,
        text: h.text,
        timestamp: h.timestamp,
        origin_time: h.origin_time,
        duration: h.duration,
        confidence: h.confidence
    }));
    localStorage.setItem('transcription-history', JSON.stringify(historyToSave));
}

async function loadHistoryFromLocal() {
    const saved = localStorage.getItem('transcription-history');
    if (saved) {
        try {
            const items = JSON.parse(saved);
            for (const item of items) {
                // Try to restore audio from DB
                const audio = await getAudioFromDB(item.id);
                if (audio) {
                    item.audio = audio;
                }

                // Add to feed
                renderHistoryItemIndividually(item);
                // Push to memory history so play/download buttons work
                transcriptionHistory.push(item);
                // Track speakers
                const itemSpeaker = item.speaker || 'Unknown';
                speakers.add(itemSpeaker);
            }
            renderSpeakerFilters();
            document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;
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
    item.dataset.id = data.id;
    item.dataset.speaker = data.speaker || 'Unknown';

    // Check for watchwords (mostly for color)
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
    }

    // Confidence styling
    const confidence = data.confidence || 0;
    let confClass = 'conf-high';
    if (confidence < -0.7) confClass = 'conf-low';
    else if (confidence < -0.5) confClass = 'conf-med';

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot') ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="confidence ${confClass}" title="Whisper Log Probability (closer to 0 is better)">${confidence.toFixed(2)}</span>
                <span class="timestamp">${data.timestamp}${data.duration ? ` (${data.duration.toFixed(1)}s)` : ''}</span>
                ${data.audio ? '' : '<span class="text-muted" style="font-size: 0.7rem; margin-left: 8px;">(Text Only)</span>'}
                <div class="action-buttons">
                    ${data.audio ? `
                        <button class="play-btn" onclick="playSegment('${data.id}')" title="Play audio">▶️</button>
                        <button class="download-btn" onclick="downloadSegment('${data.id}')" title="Download clip">📥</button>
                    ` : ''}
                </div>
            </div>
        </div>
        <div class="transcript-text">${data.text}</div>
    `;

    transcriptFeed.appendChild(item);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
}

// Initial setup for history limit
const limitInput = document.getElementById('history-limit');
const applyHistoryBtn = document.getElementById('apply-history-limit');

if (limitInput) {
    const savedLimit = localStorage.getItem('history-limit');
    if (savedLimit) {
        limitInput.value = savedLimit;
    } else {
        limitInput.value = 60; // New default
    }
}

if (applyHistoryBtn) {
    applyHistoryBtn.addEventListener('click', applyHistoryLimit);
}


// Watchword Management
function renderWatchwords() {
    const list = document.getElementById('watchwords-list');
    list.innerHTML = '';

    // Sort alphabetically for display
    const sortedWatchwords = [...watchwords].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

    sortedWatchwords.forEach((word) => {
        // Find original index for removal
        const originalIndex = watchwords.indexOf(word);
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `
            ${word}
            <span class="remove" onclick="removeWatchword(${originalIndex})">×</span>
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
        reApplyWatchwordHighlights();
        input.value = '';
    }
}

function removeWatchword(index) {
    watchwords.splice(index, 1);
    localStorage.setItem('watchwords', JSON.stringify(watchwords));
    renderWatchwords();
    reApplyWatchwordHighlights();
}

function clearWatchwords() {
    watchwords = [];
    localStorage.removeItem('watchwords');
    renderWatchwords();
    reApplyWatchwordHighlights();
}

function reApplyWatchwordHighlights() {
    // Re-evaluate all displayed transcript items for watchword matches
    const items = document.querySelectorAll('.transcript-item');
    items.forEach(item => {
        const textEl = item.querySelector('.transcript-text');
        if (!textEl) return;

        // Get plain text (strip any existing highlights)
        const text = textEl.textContent || '';

        // Re-apply inline highlights
        textEl.innerHTML = highlightWatchwords(text);

        // Toggle item highlight class
        if (checkWatchwords(text)) {
            item.classList.add('highlight');
        } else {
            item.classList.remove('highlight');
        }
    });
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

    if (!("Notification" in window)) {
        notificationBtn.style.display = 'none';
        return;
    }

    if (Notification.permission === "granted") {
        notificationBtn.innerHTML = '🔔 Send Test Alert';
        notificationBtn.className = 'btn-text full-width'; // Using btn-text for a low-profile look
        notificationBtn.style.opacity = '0.7';
        notificationBtn.title = 'Click to send a test notification to verify alerts are working.';
    } else if (Notification.permission === "denied") {
        notificationBtn.innerHTML = '⚠️ Desktop Alerts Blocked';
        notificationBtn.className = 'btn-secondary full-width';
        notificationBtn.style.opacity = '0.5';
        notificationBtn.style.cursor = 'not-allowed';
        notificationBtn.title = 'Notifications are blocked in your browser settings. Check site permissions to unblock.';
    } else {
        notificationBtn.innerHTML = '🔔 Enable Desktop Alerts';
        notificationBtn.className = 'btn-secondary full-width';
        notificationBtn.style.opacity = '1';
        notificationBtn.style.cursor = 'pointer';
    }
}

if (notificationBtn) {
    notificationBtn.addEventListener('click', async () => {
        if (!("Notification" in window)) {
            alert("This browser does not support desktop notifications.");
            return;
        }

        if (Notification.permission === 'granted') {
            triggerNotification("This is a test notification! Alerts are working correctly.");
            return;
        }

        if (Notification.permission === 'denied') {
            const browser = getBrowserName();
            const msg = `Notifications are blocked by your ${browser} settings.\n\n` +
                `How to fix:\n` +
                `1. Click the 'Locks/Settings' icon next to the URL in the address bar.\n` +
                `2. Change the 'Notifications' setting to 'Allow'.\n` +
                `3. Refresh this page.`;
            alert(msg);
            return;
        }

        // Default state: Request permission
        try {
            const permission = await Notification.requestPermission();
            updateNotificationButton();

            if (permission === 'granted') {
                triggerNotification("Notifications Enabled! You will now receive alerts for watchwords.");
            }
        } catch (e) {
            console.error("Error requesting permission:", e);
        }
    });
}

function getBrowserName() {
    const userAgent = navigator.userAgent;
    if (userAgent.match(/chrome|chromium|crios/i)) return "Chrome";
    if (userAgent.match(/firefox|fxios/i)) return "Firefox";
    if (userAgent.match(/safari/i)) return "Safari";
    if (userAgent.match(/edge/i)) return "Edge";
    return "browser";
}

// Theme Management
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.querySelector('.sidebar');

function setTheme(newTheme) {
    theme = newTheme;
    document.documentElement.setAttribute('data-theme', theme);
    themeIcon.textContent = theme === 'dark' ? '🌙' : '☀️';
    localStorage.setItem('theme', theme);
}

themeToggle.addEventListener('click', () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
});

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('active');
    sidebarToggle.classList.toggle('active');
    sidebarToggle.innerHTML = sidebar.classList.contains('active') ? '<span>×</span>' : '<span>⚙️</span>';
});

// Close sidebar on small screens when clicking outside (on the feed)
document.querySelector('.feed-section').addEventListener('click', () => {
    if (window.innerWidth <= 1024 && sidebar.classList.contains('active')) {
        sidebar.classList.remove('active');
        sidebarToggle.innerHTML = '<span>⚙️</span>';
    }
});

// History Clearing
function clearFullHistory() {
    if (!confirm("Are you sure you want to clear ALL transcription history and audio clips?")) return;

    // Clear variables
    transcriptionHistory = [];
    speakers.clear();
    document.getElementById('speaker-count').textContent = '0 Speakers Detected';

    // Clear UI
    transcriptFeed.innerHTML = '<div class="placeholder">Waiting for incoming audio...</div>';

    // Clear LocalStorage
    localStorage.removeItem('transcription-history');

    // Clear IndexedDB
    if (db) {
        const transaction = db.transaction(["audioStore"], "readwrite");
        const store = transaction.objectStore("audioStore");
        store.clear();
    }
}

document.getElementById('clear-history').addEventListener('click', clearFullHistory);

// Initialization (Remove old call, moved to DB success)
const savedTheme = localStorage.getItem('theme') || 'dark';
setTheme(savedTheme);

const savedWatchwords = localStorage.getItem('watchwords');
if (savedWatchwords) {
    watchwords = JSON.parse(savedWatchwords);
    renderWatchwords();
}

updateNotificationButton();

connect();

// Scroll Lock Functionality
const scrollLockBtn = document.getElementById('scroll-lock');
const scrollLockIcon = document.getElementById('scroll-lock-icon');

function updateScrollLockUI() {
    if (scrollLockIcon) {
        scrollLockIcon.textContent = isScrollLocked ? '🔒' : '🔓';
    }
    if (scrollLockBtn) {
        scrollLockBtn.classList.toggle('active', isScrollLocked);
        scrollLockBtn.title = isScrollLocked
            ? 'Scroll Locked - Click to unlock and jump to bottom'
            : 'Scroll Lock (auto-engages when scrolling up)';
    }
}

// Auto-engage scroll lock when user scrolls up
transcriptFeed.addEventListener('scroll', () => {
    const isAtBottom = transcriptFeed.scrollHeight - transcriptFeed.scrollTop <= transcriptFeed.clientHeight + 50;

    if (!isAtBottom && !isScrollLocked) {
        // User scrolled up, engage lock
        isScrollLocked = true;
        updateScrollLockUI();
    }
});

// Toggle scroll lock on button click
if (scrollLockBtn) {
    scrollLockBtn.addEventListener('click', () => {
        isScrollLocked = !isScrollLocked;
        updateScrollLockUI();

        // If unlocking, jump to bottom
        if (!isScrollLocked) {
            transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
        }
    });
}
