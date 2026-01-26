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
    volStatus.textContent = peak > 0.002 ? 'Active' : 'Quiet';

    if (msgCount % 20 === 0) {
        // Latency: Show the last known processing latency
        // Buffer: Show how many ms of audio are currently queued in the AudioContext
        if (audioCtx) {
            const bufferDepth = Math.max(0, startTime - audioCtx.currentTime);
            document.getElementById('buffer-stat').textContent = `${(bufferDepth * 1000).toFixed(0)}ms`;
        }
    }

    // Play if enabled
    if (isAudioEnabled && audioCtx) {
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

    // Calculate true latency
    if (data.origin_time) {
        const latency = (Date.now() / 1000) - data.origin_time;
        latencyStat.textContent = `${(latency * 1000).toFixed(0)}ms`;
    }

    speakers.add(data.speaker);
    document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker.includes('Dispatcher') || data.speaker.includes('AI') ? 'robotic' : ''}">${data.speaker || 'Unknown'}</span>
            <span class="timestamp">${data.timestamp}</span>
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

connect();
