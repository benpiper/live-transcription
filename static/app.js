const transcriptFeed = document.getElementById('transcript-feed');
const statusBadge = document.getElementById('status-badge');
const audioToggle = document.getElementById('audio-toggle');
const volLevel = document.getElementById('vol-level');
const volStatus = document.getElementById('vol-status');
const canvas = document.getElementById('visualizer');
const ctx = canvas.getContext('2d');
const latencyStat = document.getElementById('latency-stat');
const processTimeStat = document.getElementById('process-time-stat');
const bufferSizeStat = document.getElementById('buffer-size-stat');
const silentAudio = document.getElementById('silent-audio');

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
            updateVisualizerColors();
            setupCanvasVisibilityObserver();
            if (isCanvasVisible) {
                startVisualizer();
            }
        } catch (e) {
            console.error("Failed to create AudioContext:", e);
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function setupMediaSession() {
    if ('mediaSession' in navigator) {
        navigator.mediaSession.metadata = new MediaMetadata({
            title: 'Live Transcription',
            artist: 'Transcription App',
            album: 'Live Feed'
        });

        navigator.mediaSession.setActionHandler('play', () => {
            if (!isAudioEnabled) {
                audioToggle.click();
            }
        });

        navigator.mediaSession.setActionHandler('pause', () => {
            if (isAudioEnabled) {
                audioToggle.click();
            }
        });
    } else {
        console.warn("Media Session API not available - check if using HTTPS");
    }
}

function setupCanvasVisibilityObserver() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                isCanvasVisible = true;
                startVisualizer();
            } else {
                isCanvasVisible = false;
                stopVisualizer();
            }
        });
    }, { threshold: 0.1 });

    observer.observe(canvas);
}

function startVisualizer() {
    if (!isVisualizerRunning) {
        isVisualizerRunning = true;
        drawVisualizer();
    }
}

function stopVisualizer() {
    isVisualizerRunning = false;
}

// On-demand playback state
let rawAudioHistory = []; // {timestamp: float, chunk: Float32Array}
let transcriptionHistory = []; // {id: string, text: string, speaker: string, audio: Float32Array}
let isPlaybackMuted = false;
let liveAudioEnabledBeforePlayback = false;
let activePlaybackSource = null;
let activePlaybackId = null;
let speakerFilterTimeout = null;
let speakerCountTimeout = null;
// selectedSpeakers is now defined in the Speaker Filtering section below
let selectedSpeakers = new Set();  // Empty = show all speakers
let watchwords = [];
let theme = 'dark';
let isScrollLocked = false;  // When true, don't auto-scroll on new transcripts
let sessionLoaded = false;   // Prevents processing new transcripts until session is loaded
let isCanvasVisible = true;  // Track if canvas is in viewport
let isVisualizerRunning = false;  // Track if visualizer animation is active
let reconnectAttempts = 0;  // Track reconnection attempts

// Analysis node for spectrum visualization
let analyser;
let dataArray;
let cachedPrimaryColor = '#38bdf8';
let cachedAccentColor = '#818cf8';

function updateVisualizerColors() {
    const style = getComputedStyle(document.documentElement);
    cachedPrimaryColor = style.getPropertyValue('--primary').trim() || '#38bdf8';
    cachedAccentColor = style.getPropertyValue('--accent').trim() || '#818cf8';
}

let lastVolumeUpdateTime = 0;
const VOLUME_UPDATE_INTERVAL = 100; // 10fps for volume text updates

console.log("App initializing...");

// IndexedDB Setup (for future use, settings storage)
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
    // Audio is now fetched from backend on-demand via API
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

    // Handle silent audio element for backgrounding
    if (silentAudio) {
        if (isAudioEnabled) {
            silentAudio.play().catch(e => console.error("Failed to play silent audio:", e));
        } else {
            silentAudio.pause();
        }
    }

    // Update Media Session state
    if ('mediaSession' in navigator) {
        navigator.mediaSession.playbackState = isAudioEnabled ? 'playing' : 'paused';
        console.log("Media Session playback state set to:", navigator.mediaSession.playbackState);
    }
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
            speakers.clear();

            // Batch DOM updates using DocumentFragment
            const fragment = document.createDocumentFragment();

            for (const t of data.transcripts) {
                // Add to transcriptionHistory without loading binary audio into RAM
                const historyItem = {
                    id: t.origin_time ? `audio-${t.origin_time}` : `temp-${Date.now()}-${Math.random()}`,
                    speaker: t.speaker,
                    text: t.text,
                    audio: null, // Don't hold binary audio in memory
                    timestamp: t.timestamp,
                    origin_time: t.origin_time,
                    duration: t.duration,
                    confidence: t.confidence,
                    el: null
                };

                const element = createTranscriptElement(t, true); // No need to pass audio data
                historyItem.el = element;
                element.dataset.historyId = historyItem.id;
                transcriptionHistory.push(historyItem);

                fragment.appendChild(element);

                // Track speakers
                const itemSpeaker = t.speaker || 'Unknown';
                speakers.add(itemSpeaker);
            }

            // Single DOM update
            transcriptFeed.appendChild(fragment);

            // Render speaker filters once after all items loaded
            renderSpeakerFilters();
            updateSpeakerCountUI();

            // Scroll to bottom
            transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

            // Update watchword navigation if there are matches
            updateMatchCounter();

            console.log(`Loaded ${data.transcripts.length} transcripts from session`);
        }
    } catch (err) {
        console.log("No active session or error loading:", err);
    }
}

// Fetch and display audio buffer status
let bufferStatusInterval = null;

async function updateBufferStatus() {
    try {
        const response = await fetch('/api/audio/buffer-status');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const stats = await response.json();
        const bufferMb = stats.buffer_size_mb || 0;
        const durationSec = stats.duration_available_sec || 0;

        // Format buffer size: show in MB if < 1GB, otherwise GB
        let bufferText;
        if (bufferMb >= 1024) {
            bufferText = `${(bufferMb / 1024).toFixed(1)}GB`;
        } else {
            bufferText = `${bufferMb.toFixed(1)}MB`;
        }

        // Add duration info in tooltip (duration_available_sec)
        const minutes = Math.floor(durationSec / 60);
        const seconds = Math.floor(durationSec % 60);
        const durationText = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;

        bufferSizeStat.textContent = bufferText;
        bufferSizeStat.title = `Audio buffer: ${bufferText} (${durationText} of audio)`;
    } catch (err) {
        console.warn("Failed to fetch buffer status:", err);
        bufferSizeStat.textContent = 'N/A';
    }
}

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    console.log("Connecting to WebSocket:", wsUrl);

    // Show connecting state
    statusBadge.textContent = 'Connecting...';
    statusBadge.classList.remove('connected');

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = async () => {
        console.log("WebSocket connected");
        reconnectAttempts = 0;  // Reset attempts on successful connection
        statusBadge.textContent = 'Connected';
        statusBadge.classList.add('connected');

        // Load current session transcripts BEFORE processing new messages
        await loadCurrentSession();
        sessionLoaded = true;

        // Initialize Media Session metadata
        setupMediaSession();

        // Start fetching buffer status periodically (every 2 seconds)
        updateBufferStatus();  // Initial fetch
        if (bufferStatusInterval) clearInterval(bufferStatusInterval);
        bufferStatusInterval = setInterval(updateBufferStatus, 2000);
    };

    ws.onclose = (e) => {
        console.warn("WebSocket closed:", e.code, e.reason);

        // Stop buffer status updates
        if (bufferStatusInterval) {
            clearInterval(bufferStatusInterval);
            bufferStatusInterval = null;
        }
        bufferSizeStat.textContent = '---';

        reconnectAttempts++;
        const retryDelay = Math.min(2000 * Math.pow(1.5, reconnectAttempts - 1), 30000);  // Exponential backoff, max 30s

        if (reconnectAttempts === 1) {
            statusBadge.textContent = `Reconnecting...`;
        } else {
            statusBadge.textContent = `Reconnecting (attempt ${reconnectAttempts})...`;
        }
        statusBadge.classList.remove('connected');

        console.log(`Reconnection attempt ${reconnectAttempts} in ${retryDelay}ms`);
        setTimeout(connect, retryDelay);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        statusBadge.textContent = 'Error';
        statusBadge.classList.remove('connected');
    };

    ws.onmessage = (event) => {
        msgCount++;
        if (msgCount % 100 === 0) console.log(`Received ${msgCount} messages`);

        if (typeof event.data === 'string') {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'transcript') {
                    if (!sessionLoaded) {
                        console.log("Skipping transcript - session not loaded yet");
                        return;
                    }
                    console.log("Received transcript:", data.text);
                    addTranscriptItem(data);
                } else if (data.type === 'volume') {
                    const now = Date.now();
                    if (now - lastVolumeUpdateTime > VOLUME_UPDATE_INTERVAL) {
                        volLevel.textContent = data.peak.toFixed(4);
                        volStatus.textContent = data.peak > 0.004 ? 'Active' : 'Quiet';
                        lastVolumeUpdateTime = now;
                    }
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

    // Update peak for volume stat (throttled)
    const now = Date.now();
    if (now - lastVolumeUpdateTime > VOLUME_UPDATE_INTERVAL) {
        let peak = 0;
        for (let i = 0; i < floatData.length; i++) {
            const abs = Math.abs(floatData[i]);
            if (abs > peak) peak = abs;
        }
        volLevel.textContent = peak.toFixed(4);
        volStatus.textContent = peak > 0.004 ? 'Active' : 'Quiet';
        lastVolumeUpdateTime = now;
    }

    // Store in history buffer (keep approx 30 seconds)
    const timestampNow = Date.now() / 1000;
    rawAudioHistory.push({ timestamp: timestampNow, chunk: floatData });

    // Prune raw buffer (120s window to handle long transcription latency)
    const windowSec = 120;
    while (rawAudioHistory.length > 0 && rawAudioHistory[0].timestamp < timestampNow - windowSec) {
        rawAudioHistory.shift();
    }

    // Buffer monitoring removed - using compact connection info display

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

// Creates a transcript DOM element without appending it
function createTranscriptElement(data, fromSession = false) {
    const item = document.createElement('div');
    item.className = 'transcript-item';

    // Set ID based on origin_time (for audio matching)
    if (data.origin_time) {
        item.dataset.id = `audio-${data.origin_time}`;
    }

    // Audio is available if we have an origin_time (stored in IndexedDB)
    const hasAudio = !!data.origin_time;

    // Check for watchwords
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
        if (!fromSession) triggerNotification(data.text);
        // Update nav after DOM append (deferred)
        setTimeout(() => updateMatchCounter(), 0);
    }

    const itemSpeaker = data.speaker || 'Unknown';
    item.dataset.speaker = itemSpeaker;

    // Apply active filter if necessary
    if (selectedSpeakers.size > 0 && !selectedSpeakers.has(itemSpeaker)) {
        item.classList.add('filtered-out');
    }

    // Confidence styling
    const confidence = data.confidence || 0;
    let confClass = 'conf-high';
    if (confidence < -0.7) confClass = 'conf-low';
    else if (confidence < -0.6) confClass = 'conf-med';

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker && (data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot')) ? 'robotic' : ''}"
                  onclick="filterBySpeaker('${itemSpeaker}')"
                  onkeydown="if(event.key==='Enter'||event.key===' ')filterBySpeaker('${itemSpeaker}')"
                  tabindex="0"
                  role="button"
                  aria-label="Filter by speaker ${itemSpeaker}"
                  title="Click to filter by this speaker">${data.speaker || 'Unknown'}</span>
            <div class="timestamp-wrapper">
                <span class="confidence ${confClass}" title="Whisper Log Probability (closer to 0 is better)">${confidence.toFixed(2)}</span>
                <span class="timestamp">${data.timestamp}${data.duration ? ` (${data.duration.toFixed(1)}s)` : ''}</span>
                <div class="action-buttons">
                    ${hasAudio && item.dataset.id ? `<button class="play-btn" onclick="playSegment('${item.dataset.id}')" title="Play audio">▶️</button>` : ''}
                    ${hasAudio && item.dataset.id ? `<button class="download-btn" onclick="downloadSegment('${item.dataset.id}')" title="Download clip">📥</button>` : ''}
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

    // Global history item
    const historyItem = {
        id: data.origin_time ? `audio-${data.origin_time}` : `live-${Date.now()}`,
        speaker: data.speaker,
        text: data.text,
        audio: null,
        timestamp: data.timestamp,
        origin_time: data.origin_time,
        duration: data.duration,
        confidence: data.confidence,
        el: item
    };
    item.dataset.historyId = historyItem.id;

    // Check for watchwords
    if (checkWatchwords(data.text)) {
        item.classList.add('highlight');
        if (!fromSession) triggerNotification(data.text);
        setTimeout(() => updateMatchCounter(), 0);
    }

    // Process audio if present
    if (data.origin_time && !fromSession) {
        const latency = (Date.now() / 1000) - data.origin_time;
        latencyStat.textContent = `${(latency * 1000).toFixed(0)}ms`;

        // Update processing time if available
        if (data.processing_time !== undefined && data.processing_time !== null) {
            const procTimeMs = (data.processing_time * 1000).toFixed(0);
            processTimeStat.textContent = `${procTimeMs}ms`;

            // Warn if processing is slower than real-time
            if (data.processing_time > data.duration) {
                processTimeStat.classList.add('slow-processing');
            } else {
                processTimeStat.classList.remove('slow-processing');
            }
        } else {
            processTimeStat.textContent = '---';
            processTimeStat.classList.remove('slow-processing');
        }

        const segmentStart = data.origin_time - 0.5;
        const segmentEnd = Date.now() / 1000;
        const segmentChunks = rawAudioHistory
            .filter(ah => ah.timestamp >= segmentStart && ah.timestamp <= segmentEnd)
            .map(ah => ah.chunk);

        if (segmentChunks.length > 0) {
            const totalLength = segmentChunks.reduce((acc, chunk) => acc + chunk.length, 0);
            const combinedAudio = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of segmentChunks) {
                combinedAudio.set(chunk, offset);
                offset += chunk.length;
            }

            // Don't keep audio in historyItem to save RAM
            // Audio is stored in backend buffer, fetched on demand via API
            // historyItem.audio remains null
            item.dataset.id = historyItem.id;

            pruneHistory();
        }
    }

    const itemSpeaker = data.speaker || 'Unknown';
    item.dataset.speaker = itemSpeaker;

    if (!speakers.has(itemSpeaker)) {
        speakers.add(itemSpeaker);
        throttledRenderSpeakerFilters();
        throttledUpdateSpeakerCount();
    }

    // Apply active filter if necessary
    if (selectedSpeakers.size > 0 && !selectedSpeakers.has(itemSpeaker)) {
        item.classList.add('filtered-out');
    }

    // Confidence styling
    const confidence = data.confidence || 0;
    let confClass = 'conf-high';
    if (confidence < -0.7) confClass = 'conf-low';
    else if (confidence < -0.6) confClass = 'conf-med';

    item.innerHTML = `
        <div class="transcript-header">
            <span class="speaker ${data.speaker && (data.speaker.includes('Dispatcher') || data.speaker.includes('AI') || data.speaker.includes('Bot')) ? 'robotic' : ''}"
                  onclick="filterBySpeaker('${itemSpeaker}')"
                  onkeydown="if(event.key==='Enter'||event.key===' ')filterBySpeaker('${itemSpeaker}')"
                  tabindex="0"
                  role="button"
                  aria-label="Filter by speaker ${itemSpeaker}"
                  title="Click to filter by this speaker">${data.speaker || 'Unknown'}</span>
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

    transcriptionHistory.push(historyItem);
    transcriptFeed.appendChild(item);

    if (!isScrollLocked) {
        transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
    }
}

function updateSpeakerCountUI() {
    document.getElementById('speaker-count').textContent = `${speakers.size} Speakers Detected`;
}

function throttledRenderSpeakerFilters() {
    if (speakerFilterTimeout) return;
    speakerFilterTimeout = setTimeout(() => {
        renderSpeakerFilters();
        speakerFilterTimeout = null;
    }, 1000);
}

function throttledUpdateSpeakerCount() {
    if (speakerCountTimeout) return;
    speakerCountTimeout = setTimeout(() => {
        updateSpeakerCountUI();
        speakerCountTimeout = null;
    }, 2000);
}

function drawVisualizer() {
    if (!analyser || !ctx || !isVisualizerRunning) return;

    requestAnimationFrame(drawVisualizer);

    const width = canvas.width;
    const height = canvas.height;

    analyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, width, height);

    const barWidth = (width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        barHeight = (dataArray[i] / 255) * height;

        const xPos = x;
        const yPos = height - barHeight;

        ctx.fillStyle = cachedPrimaryColor;
        ctx.fillRect(xPos, yPos, barWidth - 2, barHeight);

        x += barWidth + 1;
    }
}

// Pre-create gradient once when colors change? Actually, solid color is much faster for animation.
// If user really wants gradient, it should be created once outside the loop.

// Speaker Filtering (Multi-select dropdown)

function renderSpeakerFilters() {
    const filterContainer = document.getElementById('speaker-filters');
    if (!filterContainer) return;

    // Get current speakers in the list
    const currentItems = Array.from(filterContainer.querySelectorAll('.dropdown-item')).map(
        item => item.dataset.speaker
    );

    speakers.forEach(speaker => {
        if (!currentItems.includes(speaker)) {
            const item = document.createElement('label');
            item.className = 'dropdown-item';
            item.dataset.speaker = speaker;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = selectedSpeakers.size === 0 || selectedSpeakers.has(speaker);
            checkbox.addEventListener('change', () => toggleSpeaker(speaker, checkbox.checked));

            const label = document.createElement('span');
            label.textContent = speaker;

            item.appendChild(checkbox);
            item.appendChild(label);
            filterContainer.appendChild(item);
        }
    });
}

function toggleSpeaker(speaker, checked) {
    if (checked) {
        // If we had "show all" mode (empty set), switching to selective mode
        if (selectedSpeakers.size === 0) {
            // Add all speakers except this one will be unchecked scenario
            // Actually, if checking one while in "show all", we stay in show all
        }
        selectedSpeakers.add(speaker);
    } else {
        // If currently showing all, need to switch to selective mode
        if (selectedSpeakers.size === 0) {
            // Populate with all except the unchecked one
            speakers.forEach(s => {
                if (s !== speaker) selectedSpeakers.add(s);
            });
        } else {
            selectedSpeakers.delete(speaker);
        }
    }
    applyMultiSpeakerFilter();
}

function applyMultiSpeakerFilter() {
    // Update dropdown label
    const label = document.getElementById('speaker-filter-label');
    if (selectedSpeakers.size === 0 || selectedSpeakers.size === speakers.length) {
        label.textContent = 'All Speakers';
    } else if (selectedSpeakers.size === 1) {
        label.textContent = [...selectedSpeakers][0];
    } else {
        label.textContent = `${selectedSpeakers.size} speakers selected`;
    }

    // Filter feed using cached references
    transcriptionHistory.forEach(item => {
        if (!item.el) return;
        const itemSpeaker = item.speaker || 'Unknown';

        // Show all if no filter, or if speaker is selected
        if (selectedSpeakers.size === 0 || selectedSpeakers.has(itemSpeaker)) {
            item.el.classList.remove('filtered-out');
        } else {
            item.el.classList.add('filtered-out');
        }
    });

    // Scroll if not locked
    if (!isScrollLocked) {
        transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
    }
}

function filterBySpeaker(speaker) {
    // If clicking a speaker, we want to show ONLY that speaker.
    // Reset selection and add this speaker.
    selectedSpeakers.clear();
    selectedSpeakers.add(speaker);

    // Update the checkbox UI
    document.querySelectorAll('#speaker-filters input[type="checkbox"]').forEach(cb => {
        const item = cb.closest('.dropdown-item');
        cb.checked = (item && item.dataset.speaker === speaker);
    });

    applyMultiSpeakerFilter();
}

// Dropdown toggle
document.getElementById('speaker-dropdown-toggle').addEventListener('click', () => {
    const menu = document.getElementById('speaker-dropdown-menu');
    const toggle = document.getElementById('speaker-dropdown-toggle');
    menu.classList.toggle('hidden');
    toggle.classList.toggle('open');
});

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.speaker-dropdown');
    if (!dropdown.contains(e.target)) {
        document.getElementById('speaker-dropdown-menu').classList.add('hidden');
        document.getElementById('speaker-dropdown-toggle').classList.remove('open');
    }
});

// Search functionality
document.getElementById('speaker-search').addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    document.querySelectorAll('#speaker-filters .dropdown-item').forEach(item => {
        const speaker = item.dataset.speaker.toLowerCase();
        item.classList.toggle('hidden', !speaker.includes(query));
    });
});

// Select All
document.getElementById('select-all-speakers').addEventListener('click', () => {
    selectedSpeakers.clear();  // Empty = show all
    document.querySelectorAll('#speaker-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    applyMultiSpeakerFilter();
});

// Deselect All
document.getElementById('deselect-all-speakers').addEventListener('click', () => {
    selectedSpeakers.clear();
    speakers.forEach(s => selectedSpeakers.add(s));  // Will then remove all
    selectedSpeakers.clear();
    // Actually just hide everything
    document.querySelectorAll('#speaker-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    // Set to explicit empty selection (hide all)
    speakers.forEach(s => { }); // noop - keep selectedSpeakers as we want
    // Force hide all by creating a dummy set that matches nothing
    selectedSpeakers = new Set(['__NONE__']);
    applyMultiSpeakerFilter();
});

// Reset/Clear button
document.getElementById('reset-filters').addEventListener('click', () => {
    selectedSpeakers.clear();
    document.querySelectorAll('#speaker-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    document.getElementById('speaker-search').value = '';
    document.querySelectorAll('#speaker-filters .dropdown-item').forEach(item => {
        item.classList.remove('hidden');
    });
    applyMultiSpeakerFilter();
});

async function downloadSegment(id) {
    const item = transcriptionHistory.find(h => h.id === id);
    if (!item) {
        console.error("Item not found for download:", id);
        console.log("Available items:", transcriptionHistory.map(h => h.id).slice(-5));
        alert("Transcript item not found");
        return;
    }

    console.log("Download item found:", {id: item.id, duration: item.duration, origin_time: item.origin_time, text: item.text.substring(0, 50)});

    // Extract timestamp from segment ID
    // Format: "audio-{origin_time}" where origin_time is Unix float
    const originTime = parseFloat(id.replace("audio-", ""));
    if (isNaN(originTime)) {
        console.error("Invalid segment ID format:", id);
        return;
    }

    // Use transcript's exact duration for download
    const duration = item.duration || 5.0;  // Fallback to 5s if duration missing
    const startTime = originTime;
    const endTime = originTime + duration;

    console.log(`Downloading: id='${id}', origin=${originTime.toFixed(3)}, duration=${duration.toFixed(3)}, range=[${startTime.toFixed(3)}, ${endTime.toFixed(3)}]`);

    try {
        const response = await fetch(
            `/api/audio/range?start_time=${startTime}&end_time=${endTime}&format=wav`
        );

        if (!response.ok) {
            if (response.status === 400) {
                alert("Audio not available. This segment is outside the audio buffer window (2 hours).");
            } else {
                alert("Failed to download audio segment");
            }
            return;
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `clip_${originTime.toFixed(0)}.wav`;
        link.click();
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error("Error downloading audio:", error);
        alert("Failed to download audio segment");
    }
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

async function playSegment(id) {
    const item = transcriptionHistory.find(h => h.id === id);
    if (!item) return;

    // Extract timestamp from segment ID
    const originTime = parseFloat(id.replace("audio-", ""));
    if (isNaN(originTime)) {
        console.error("Invalid segment ID format:", id);
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
        return;
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

    const btn = document.querySelector(`[data-id="${id}"] .play-btn`);
    if (btn) {
        btn.innerHTML = '⏸️';
        btn.classList.add('playing');
    }

    try {
        // Use transcript's exact duration for playback
        const duration = item.duration || 5.0;  // Fallback to 5s if duration missing
        const startTime = originTime;
        const endTime = originTime + duration;

        const response = await fetch(
            `/api/audio/range?start_time=${startTime}&end_time=${endTime}&format=raw`
        );

        if (!response.ok) {
            if (response.status === 400) {
                console.warn("Audio not available (outside buffer window)");
                // Try fallback to rawAudioHistory if available
                const historyAudio = getAudioFromHistory(originTime);
                if (historyAudio && historyAudio.length > 0) {
                    console.log("Using fallback audio from history");
                    playAudioBuffer(historyAudio, id, btn, context);
                } else {
                    alert("Audio not available for playback. This segment is outside the 2-hour buffer window.");
                }
            } else {
                alert("Failed to load audio segment");
            }
            return;
        }

        const arrayBuffer = await response.arrayBuffer();
        const floatData = new Float32Array(arrayBuffer);
        playAudioBuffer(floatData, id, btn, context);

    } catch (error) {
        console.error("Error loading audio:", error);
        alert("Failed to load audio segment");
    }
}

// Helper function to play audio from Float32 data
function playAudioBuffer(audioData, id, btn, context) {
    // 3. Mute live audio
    if (!isPlaybackMuted) {
        liveAudioEnabledBeforePlayback = isAudioEnabled;
        isPlaybackMuted = true;
    }

    const buffer = context.createBuffer(1, audioData.length, 16000);
    buffer.getChannelData(0).set(audioData);

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

// Fallback: get audio from rawAudioHistory (for recent transcripts within browser buffer)
function getAudioFromHistory(originTime) {
    const tolerance = 0.5;  // 500ms tolerance
    const matches = rawAudioHistory.filter(item => Math.abs(item.timestamp - originTime) < tolerance);
    if (matches.length === 0) return null;

    // Concatenate matching audio chunks
    const totalLength = matches.reduce((sum, item) => sum + item.chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const item of matches) {
        combined.set(item.chunk, offset);
        offset += item.chunk.length;
    }
    return combined;
}

// History Pruning
function pruneHistory() {
    // Use saved limit from localStorage or default to 1000 (keeps watchword matches visible longer)
    const savedLimit = localStorage.getItem('history-limit');
    const limit = savedLimit ? parseInt(savedLimit) : 1000;

    while (transcriptionHistory.length > limit) {
        const removed = transcriptionHistory.shift();

        // Audio is stored in backend buffer, no local cleanup needed
        // Just remove UI element if it exists
        if (removed.el) {
            removed.el.remove();
        }
    }
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
    // Re-evaluate all displayed transcript items using cached references
    transcriptionHistory.forEach(item => {
        if (!item.el) return;
        const textEl = item.el.querySelector('.transcript-text');
        if (!textEl) return;

        // Get plain text
        const text = item.text || textEl.textContent || '';

        // Re-apply inline highlights
        textEl.innerHTML = highlightWatchwords(text);

        // Toggle item highlight class
        if (checkWatchwords(text)) {
            item.el.classList.add('highlight');
        } else {
            item.el.classList.remove('highlight');
        }
    });

    // Update navigation controls
    updateMatchCounter();
}


// Watchword Navigation State
let currentMatchIndex = 0;
let showOnlyMatches = false;

function getMatchedItems() {
    return transcriptionHistory.filter(item => item.el && item.el.classList.contains('highlight')).map(item => item.el);
}

function updateMatchCounter() {
    const matches = getMatchedItems();
    const counter = document.getElementById('match-counter');
    const nav = document.getElementById('watchword-nav');
    const timeline = document.getElementById('match-timeline');

    if (matches.length === 0) {
        nav.classList.add('hidden');
        timeline.classList.add('hidden');
        return;
    }

    nav.classList.remove('hidden');
    timeline.classList.remove('hidden');

    // Clamp index
    if (currentMatchIndex >= matches.length) currentMatchIndex = matches.length - 1;
    if (currentMatchIndex < 0) currentMatchIndex = 0;

    counter.textContent = `${currentMatchIndex + 1}/${matches.length}`;

    // Update button states
    document.getElementById('prev-match').disabled = currentMatchIndex === 0;
    document.getElementById('next-match').disabled = currentMatchIndex >= matches.length - 1;

    // Update timeline
    updateMatchTimeline(matches);
}

function updateMatchTimeline(matches) {
    const timeline = document.getElementById('match-timeline');
    const feed = document.getElementById('transcript-feed');

    timeline.innerHTML = '';

    if (matches.length === 0 || feed.scrollHeight === 0) return;

    const feedScrollHeight = feed.scrollHeight;

    matches.forEach((item, index) => {
        const itemTop = item.offsetTop;
        const position = (itemTop / feedScrollHeight) * 100;

        // Get timestamp from the transcript item (already in element)
        const timestampEl = item.querySelector('.timestamp');
        const timestamp = timestampEl ? timestampEl.textContent : `Match ${index + 1}`;

        const marker = document.createElement('div');
        marker.className = 'timeline-marker' + (index === currentMatchIndex ? ' current' : '');
        marker.style.left = `${Math.min(position, 98)}%`;
        marker.title = timestamp;
        marker.addEventListener('click', () => jumpToMatch(index));
        timeline.appendChild(marker);
    });
}

function jumpToMatch(index) {
    const matches = getMatchedItems();
    if (index < 0 || index >= matches.length) return;

    currentMatchIndex = index;
    const target = matches[index];

    // Scroll into view with offset
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Brief highlight pulse
    target.style.transition = 'box-shadow 0.2s';
    target.style.boxShadow = '0 0 0 3px var(--primary)';
    setTimeout(() => {
        target.style.boxShadow = '';
    }, 1000);

    updateMatchCounter();
}

function prevMatch() {
    if (currentMatchIndex > 0) {
        jumpToMatch(currentMatchIndex - 1);
    }
}

function nextMatch() {
    const matches = getMatchedItems();
    if (currentMatchIndex < matches.length - 1) {
        jumpToMatch(currentMatchIndex + 1);
    }
}

function toggleMatchFilter() {
    showOnlyMatches = !showOnlyMatches;
    const btn = document.getElementById('filter-matches');
    const icon = document.getElementById('filter-icon');

    if (showOnlyMatches) {
        btn.classList.add('active');
        btn.innerHTML = '<span id="filter-icon">👁</span> Matches';
        // Hide non-matching items using cached references
        transcriptionHistory.forEach(item => {
            if (item.el && !item.el.classList.contains('highlight')) {
                item.el.classList.add('match-filtered');
            }
        });
    } else {
        btn.classList.remove('active');
        btn.innerHTML = '<span id="filter-icon">👁</span> All';
        // Show all items using cached references
        transcriptionHistory.forEach(item => {
            if (item.el && item.el.classList.contains('match-filtered')) {
                item.el.classList.remove('match-filtered');
            }
        });
    }

    updateMatchTimeline(getMatchedItems());
}

// Event Listeners for Watchwords
document.getElementById('add-watchword').addEventListener('click', addWatchword);
document.getElementById('watchword-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addWatchword();
});
document.getElementById('clear-watchwords').addEventListener('click', clearWatchwords);

// Event Listeners for Navigation
document.getElementById('prev-match').addEventListener('click', prevMatch);
document.getElementById('next-match').addEventListener('click', nextMatch);
document.getElementById('filter-matches').addEventListener('click', toggleMatchFilter);

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
    updateVisualizerColors(); // Refresh colors for visualizer
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

// Watchword Collapse Toggle
const watchwordCollapseToggle = document.getElementById('watchword-collapse-toggle');
const watchwordCollapseIcon = document.getElementById('watchword-collapse-icon');
const watchwordContent = document.getElementById('watchword-content');

let isWatchwordsCollapsed = localStorage.getItem('watchwords-collapsed') === 'true';

function updateWatchwordCollapseUI() {
    if (isWatchwordsCollapsed) {
        watchwordContent.classList.add('collapsed');
        watchwordCollapseIcon.textContent = '+';
    } else {
        watchwordContent.classList.remove('collapsed');
        watchwordCollapseIcon.textContent = '−';
    }
}

if (watchwordCollapseToggle) {
    watchwordCollapseToggle.addEventListener('click', () => {
        isWatchwordsCollapsed = !isWatchwordsCollapsed;
        localStorage.setItem('watchwords-collapsed', isWatchwordsCollapsed);
        updateWatchwordCollapseUI();
    });
}

// Initial state
updateWatchwordCollapseUI();

// History Clearing
function clearFullHistory() {
    if (!confirm("Are you sure you want to clear ALL transcription history and audio clips?")) return;

    // Clear variables
    transcriptionHistory = [];
    speakers.clear();
    if (speakerFilterTimeout) clearTimeout(speakerFilterTimeout);
    if (speakerCountTimeout) clearTimeout(speakerCountTimeout);
    speakerFilterTimeout = null;
    speakerCountTimeout = null;

    document.getElementById('speaker-count').textContent = '0 Speakers Detected';

    // Clear UI
    transcriptFeed.innerHTML = '<div class="placeholder">Waiting for incoming audio...</div>';

    // Clear speaker filters UI
    const filterContainer = document.getElementById('speaker-filters');
    if (filterContainer) filterContainer.innerHTML = '';

    // Clear LocalStorage
    localStorage.removeItem('transcription-history');

    // Clear IndexedDB
    if (db) {
        const transaction = db.transaction(["audioStore"], "readwrite");
        const store = transaction.objectStore("audioStore");
        store.clear();
    }
}


// Initialization (Remove old call, moved to DB success)
const savedTheme = localStorage.getItem('theme') || 'dark';
setTheme(savedTheme);

const savedWatchwords = localStorage.getItem('watchwords');
if (savedWatchwords) {
    watchwords = JSON.parse(savedWatchwords);
    renderWatchwords();
}

updateNotificationButton();

setupMediaSession();

connect();

// Scroll Lock Functionality
const scrollLockBtn = document.getElementById('scroll-lock');
const scrollLockIcon = document.getElementById('scroll-lock-icon');

function updateScrollLockUI() {
    if (scrollLockIcon) {
        scrollLockIcon.textContent = isScrollLocked ? '⏸ PAUSED' : '▼ LIVE';
    }
    if (scrollLockBtn) {
        scrollLockBtn.classList.toggle('active', isScrollLocked);
        scrollLockBtn.classList.toggle('locked', isScrollLocked);
        scrollLockBtn.title = isScrollLocked
            ? 'Scroll paused - Click to resume auto-scroll'
            : 'Auto-scrolling - Scroll up to pause';
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

