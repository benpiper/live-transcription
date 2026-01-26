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
let historyLimit = 10;
let isPlaybackMuted = false;
let liveAudioEnabledBeforePlayback = false;
let activePlaybackSource = null;
let activePlaybackId = null;

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

    // Prune raw buffer (30s window)
    const windowSec = 30;
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

function addTranscriptItem(data) {
    // Remove placeholder
    const placeholder = transcriptFeed.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'transcript-item';

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

            const itemId = `segment-${Date.now()}`;
            const historyItem = {
                id: itemId,
                speaker: data.speaker,
                text: data.text,
                audio: combinedAudio,
                timestamp: data.timestamp
            };

            transcriptionHistory.push(historyItem);

            // Limit history
            const limit = parseInt(document.getElementById('history-limit').value) || 10;
            while (transcriptionHistory.length > limit) {
                transcriptionHistory.shift();
            }

            // Add ID for playback lookup
            item.dataset.id = itemId;
        }
    }

    speakers.add(data.speaker);
    document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot') ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="timestamp">${data.timestamp}</span>
                ${item.dataset.id ? `<button class="play-btn" onclick="playSegment('${item.dataset.id}')" title="Play audio">▶️</button>` : ''}
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

connect();
