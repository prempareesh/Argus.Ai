/**
 * Object Detection Pro - High Performance AI Neural Engine
 * Powered by TensorFlow.js, COCO-SSD & NVIDIA NIM Multimodal Vision AI
 * 100% Live Computer Vision Stream with IoU Temporal Stabilization
 */

// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const modelStatus = document.getElementById('model-status');
const loadingScreen = document.getElementById('loading-screen');
const objectList = document.getElementById('object-list');
const voiceToggle = document.getElementById('voice-toggle');
const threshRange = document.getElementById('threshold-range');
const threshVal = document.getElementById('threshold-val');
const detectionInfo = document.getElementById('detection-info');

// Multimodal AI Scene Intelligence DOM Elements
const inspectSceneBtn = document.getElementById('inspect-scene-btn');
const aiScenePanel = document.getElementById('ai-scene-panel');
const aiSceneContent = document.getElementById('ai-scene-content');
const closeSceneBtn = document.getElementById('close-scene-btn');
const speakSceneBtn = document.getElementById('speak-scene-btn');

// Global State
let model = null;
let currentStream = null;
let isPageVisible = true;
let isCameraRunning = false;
let isDetecting = false;
let isInspectingScene = false;
let minConfidence = 0.50; // 50% default base threshold
let lastSpokenSummary = "";
let lastSceneExplanation = "";
let speechCooldown = 0;

// Temporal IoU Tracking State
let activeTracks = [];
let nextTrackId = 1;
const IOU_MATCH_THRESHOLD = 0.32;
const MAX_MISSES = 4; // Purge tracks after 4 consecutive missed inferences (~200ms)

// High False-Positive Ambiguous Classes (require higher confidence & persistence)
const AMBIGUOUS_SMALL_CLASSES = new Set([
    'cup', 'cell phone', 'remote', 'mouse', 'fork', 'spoon', 'knife', 'toothbrush'
]);

// High-Contrast Assistive Colors for Detection Classes
const classColors = {
    'person': '#00E676',       // Vibrant Green
    'car': '#00E5FF',          // Cyan
    'dog': '#FFD600',          // Yellow
    'cat': '#FFD600',
    'traffic light': '#FF1744',// Red
    'stop sign': '#FF1744',
    'bottle': '#FF9100',       // Orange
    'cup': '#FF9100',
    'chair': '#E040FB',        // Purple
    'couch': '#E040FB',
    'bed': '#E040FB',
    'laptop': '#2979FF',       // Blue
    'cell phone': '#2979FF',
    'book': '#00B0FF',
    'backpack': '#AA00FF',
    'default': '#00F2FE'       // Electric Cyan
};

function getColor(className, isReliable = true) {
    if (!isReliable) return '#FFB300'; // Amber/Yellow for unverified / low confidence
    return classColors[className] || classColors['default'];
}

// Text-to-Speech
const synth = window.speechSynthesis;

function speak(text, priority = false) {
    if (!synth || !voiceToggle || !voiceToggle.checked) return;
    const cleanText = text.replace(/[*#_~`]/g, '');

    const doSpeak = () => {
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        utterance.onend = () => { window._objDetectUtterance = null; };
        utterance.onerror = () => { window._objDetectUtterance = null; };
        window._objDetectUtterance = utterance;

        try {
            if (synth.paused) {
                synth.resume();
            }
        } catch (e) {}

        synth.speak(utterance);
    };

    if (synth.speaking) {
        if (priority) {
            synth.cancel();
            setTimeout(doSpeak, 50);
        }
        return;
    }

    doSpeak();
}

// 1. Initialize Neural AI & Camera
async function init() {
    try {
        console.log("Initializing TensorFlow COCO-SSD Model...");
        if (modelStatus) {
            modelStatus.innerText = "LOADING AI MODEL...";
            modelStatus.style.color = "#FFD600";
        }

        // Load COCO-SSD
        if (typeof cocoSsd !== 'undefined') {
            model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
            console.log("✓ COCO-SSD Neural Model Ready");
            if (modelStatus) {
                modelStatus.innerText = "MODEL READY";
                modelStatus.style.color = "#00E676";
            }
        } else {
            throw new Error("COCO-SSD library not loaded");
        }

        // Setup Live Camera
        const camOk = await setupCamera();

        // Dismiss loading screen
        if (loadingScreen) {
            loadingScreen.classList.add('hide');
        }

        if (camOk) {
            speak("Object detection online. Scanning your surroundings.");
            requestAnimationFrame(renderLoop);
        } else {
            if (modelStatus) {
                modelStatus.innerText = "CAMERA OFF";
                modelStatus.style.color = "#FF1744";
            }
        }

    } catch (err) {
        console.error("Initialization error:", err);
        if (loadingScreen) loadingScreen.classList.add('hide');
        if (modelStatus) {
            modelStatus.innerText = "LOAD ERROR";
            modelStatus.style.color = "#FF1744";
        }
    }
}

// 2. Camera Setup with Resolution Fallbacks
async function setupCamera() {
    try {
        if (isCameraRunning && currentStream) return true;

        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "environment"
            },
            audio: false
        };

        const fallback = { video: true, audio: false };

        try {
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (e) {
            console.warn("High-res camera unavailable, using fallback:", e);
            currentStream = await navigator.mediaDevices.getUserMedia(fallback);
        }

        video.srcObject = currentStream;

        await new Promise((resolve) => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                return resolve();
            }
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
            setTimeout(() => {
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                resolve();
            }, 800);
        });

        await video.play();
        isCameraRunning = true;
        return true;

    } catch (err) {
        console.error("Camera setup error:", err);
        return false;
    }
}

// 3. Continuous 60 FPS Render & Inference Loop
let frameCount = 0;
async function renderLoop() {
    if (!isPageVisible) {
        requestAnimationFrame(renderLoop);
        return;
    }

    // Keep canvas dimensions synced with live stream
    if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    // Run neural inference every 3 frames (~20 FPS)
    frameCount++;
    if (model && video.readyState >= 2 && !isDetecting && frameCount % 3 === 0) {
        runLiveInference();
    }

    // Draw stabilized bounding boxes and spatial tags
    drawOverlays();

    requestAnimationFrame(renderLoop);
}

// 4. Compute Intersection over Union (IoU)
function computeIoU(boxA, boxB) {
    const [ax, ay, aw, ah] = boxA;
    const [bx, by, bw, bh] = boxB;

    const xLeft = Math.max(ax, bx);
    const yTop = Math.max(ay, by);
    const xRight = Math.min(ax + aw, bx + bw);
    const yBottom = Math.min(ay + ah, by + bh);

    if (xRight <= xLeft || yBottom <= yTop) return 0.0;

    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    const areaA = aw * ah;
    const areaB = bw * bh;
    const unionArea = areaA + areaB - intersectionArea;

    return unionArea > 0 ? (intersectionArea / unionArea) : 0.0;
}

// 5. Execute Neural Classification & Temporal IoU Stabilization
async function runLiveInference() {
    isDetecting = true;
    try {
        const rawDetections = await model.detect(video);
        updateTemporalTracker(rawDetections || []);

        // Filter active confirmed/visible tracks
        const visibleTracks = activeTracks.filter(t => t.hits >= 2 && t.misses === 0);

        // Update UI Tags and Spoken Announcements
        updateUIBadges(visibleTracks);
        generateSpatialVoiceGuidance(visibleTracks);

    } catch (err) {
        console.warn("Inference frame skip:", err);
    } finally {
        isDetecting = false;
    }
}

// 6. Temporal Multi-Frame IoU Tracker Algorithm
function updateTemporalTracker(rawDetections) {
    const matchedTrackIndices = new Set();
    const matchedDetectionIndices = new Set();

    // Step A: Find best IoU matches between incoming detections and existing tracks
    for (let dIdx = 0; dIdx < rawDetections.length; dIdx++) {
        const det = rawDetections[dIdx];
        const detClass = det.class.toLowerCase();
        const detScore = det.score;
        const detBbox = det.bbox;

        // Class-specific threshold guard: require higher baseline for ambiguous small items
        const isAmbiguous = AMBIGUOUS_SMALL_CLASSES.has(detClass);
        const threshold = isAmbiguous ? Math.max(minConfidence, 0.60) : minConfidence;

        if (detScore < threshold * 0.85) {
            // Discard severely under-threshold predictions immediately
            continue;
        }

        let bestIoU = 0;
        let bestTrackIdx = -1;

        for (let tIdx = 0; tIdx < activeTracks.length; tIdx++) {
            if (matchedTrackIndices.has(tIdx)) continue;

            const track = activeTracks[tIdx];
            const iou = computeIoU(detBbox, track.bbox);

            if (iou > bestIoU) {
                bestIoU = iou;
                bestTrackIdx = tIdx;
            }
        }

        if (bestIoU >= IOU_MATCH_THRESHOLD && bestTrackIdx !== -1) {
            // Matched with existing track: Smooth coordinates with Exponential Moving Average (EMA)
            const track = activeTracks[bestTrackIdx];
            const alpha = 0.65; // Smoothing factor

            track.bbox = [
                alpha * detBbox[0] + (1 - alpha) * track.bbox[0],
                alpha * detBbox[1] + (1 - alpha) * track.bbox[1],
                alpha * detBbox[2] + (1 - alpha) * track.bbox[2],
                alpha * detBbox[3] + (1 - alpha) * track.bbox[3]
            ];

            // Update smoothed confidence score
            track.score = alpha * detScore + (1 - alpha) * track.score;
            track.class = detClass;
            track.hits++;
            track.misses = 0;

            // Determine reliability
            const hitsNeeded = isAmbiguous ? 3 : 2;
            const confNeeded = isAmbiguous ? 0.62 : minConfidence;
            track.isReliable = (track.hits >= hitsNeeded) && (track.score >= confNeeded);
            track.confirmed = track.hits >= hitsNeeded;

            matchedTrackIndices.add(bestTrackIdx);
            matchedDetectionIndices.add(dIdx);
        }
    }

    // Step B: Spawn new tracks for unmatched confident detections
    for (let dIdx = 0; dIdx < rawDetections.length; dIdx++) {
        if (matchedDetectionIndices.has(dIdx)) continue;

        const det = rawDetections[dIdx];
        const detClass = det.class.toLowerCase();
        const detScore = det.score;
        const isAmbiguous = AMBIGUOUS_SMALL_CLASSES.has(detClass);
        const threshold = isAmbiguous ? Math.max(minConfidence, 0.60) : minConfidence;

        if (detScore >= threshold) {
            activeTracks.push({
                id: nextTrackId++,
                class: detClass,
                bbox: [...det.bbox],
                score: detScore,
                hits: 1,
                misses: 0,
                confirmed: false,
                isReliable: false,
                isAmbiguous
            });
        }
    }

    // Step C: Age unmatched existing tracks and purge expired ones
    for (let tIdx = 0; tIdx < activeTracks.length; tIdx++) {
        if (!matchedTrackIndices.has(tIdx)) {
            activeTracks[tIdx].misses++;
        }
    }

    activeTracks = activeTracks.filter(track => track.misses <= MAX_MISSES);
}

// 7. Render Stabilized Overlays & Uncertainty Indicators
function drawOverlays() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const visibleTracks = activeTracks.filter(t => t.hits >= 2 && t.misses === 0);
    if (visibleTracks.length === 0) return;

    visibleTracks.forEach(track => {
        const className = track.class;
        const score = Math.round(track.score * 100);
        const [x, y, width, height] = track.bbox;
        const isReliable = track.isReliable;
        const color = getColor(className, isReliable);

        // Spatial position calculation
        const centerX = x + width / 2;
        const normX = centerX / (canvas.width || 640);
        let posTag = "CENTER";
        if (normX < 0.35) posTag = "LEFT";
        else if (normX > 0.65) posTag = "RIGHT";

        // Draw Bounding Box: Dashed for unverified/uncertain, solid for verified
        ctx.strokeStyle = color;
        ctx.lineWidth = isReliable ? 3 : 2;
        if (!isReliable) {
            ctx.setLineDash([6, 4]); // Dashed line for uncertain detections
        } else {
            ctx.setLineDash([]);
        }

        ctx.strokeRect(x, y, width, height);

        // Reset line dash for solid corner brackets
        ctx.setLineDash([]);
        const cornerLen = Math.min(18, width / 4, height / 4);
        ctx.lineWidth = 4;
        ctx.beginPath();
        // Top-left
        ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
        // Top-right
        ctx.moveTo(x + width - cornerLen, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + cornerLen);
        // Bottom-left
        ctx.moveTo(x, y + height - cornerLen); ctx.lineTo(x, y + height); ctx.lineTo(x + cornerLen, y + height);
        // Bottom-right
        ctx.moveTo(x + width - cornerLen, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - cornerLen);
        ctx.stroke();

        // Label Tag with Uncertainty Indicator
        const statusBadge = isReliable ? "[VERIFIED]" : "[UNVERIFIED]";
        const labelText = `${className.toUpperCase()} ${score}% [${posTag}] ${statusBadge}`;

        ctx.font = 'bold 13px "Inter", Arial, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const labelHeight = 24;
        const labelY = (y - labelHeight > 0) ? y - labelHeight : y;

        ctx.fillStyle = isReliable ? "rgba(10, 15, 25, 0.88)" : "rgba(25, 20, 10, 0.90)";
        ctx.fillRect(x, labelY, textWidth + 14, labelHeight);

        ctx.fillStyle = color;
        ctx.fillText(labelText, x + 7, labelY + 16);
    });
}

// 8. Update UI Badges with Stabilized Live Data
function updateUIBadges(tracks) {
    if (!objectList) return;

    if (tracks.length === 0) {
        objectList.innerHTML = '<span class="object-tag" style="background:transparent; border:1px dashed rgba(255,255,255,0.3);">Scanning scene...</span>';
        if (detectionInfo) {
            detectionInfo.innerHTML = '<div class="label">SCANNING FOR OBJECTS...</div>';
        }
        return;
    }

    objectList.innerHTML = '';
    tracks.forEach(t => {
        const span = document.createElement('span');
        const pct = Math.round(t.score * 100);
        if (!t.isReliable) {
            span.className = 'object-tag unverified';
            span.innerText = `${t.class.toUpperCase()}? (${pct}%) [UNVERIFIED]`;
        } else {
            span.className = 'object-tag';
            span.innerText = `${t.class.toUpperCase()} (${pct}%)`;
        }
        objectList.appendChild(span);
    });

    if (detectionInfo) {
        const verifiedCount = tracks.filter(t => t.isReliable).length;
        const topTrack = tracks[0];
        detectionInfo.innerHTML = `<div class="label">${tracks.length} OBJECT${tracks.length > 1 ? 'S' : ''} DETECTED (${verifiedCount} VERIFIED, TOP: ${topTrack.class.toUpperCase()})</div>`;
    }
}

// 9. Intelligent Spatial Voice Guidance (Only Speaks Verified Stable Objects)
function generateSpatialVoiceGuidance(tracks) {
    if (!voiceToggle || !voiceToggle.checked) return;

    const now = Date.now();
    if (now < speechCooldown) return;

    // Only speak verified objects to avoid misinforming blind users
    const verifiedTracks = tracks.filter(t => t.isReliable);
    if (verifiedTracks.length === 0) {
        return;
    }

    // Group items by spatial sector
    const leftItems = [];
    const centerItems = [];
    const rightItems = [];

    verifiedTracks.forEach(t => {
        const centerX = t.bbox[0] + t.bbox[2] / 2;
        const normX = centerX / (canvas.width || 640);
        if (normX < 0.35) leftItems.push(t.class);
        else if (normX > 0.65) rightItems.push(t.class);
        else centerItems.push(t.class);
    });

    const parts = [];
    if (centerItems.length > 0) {
        parts.push(`${[...new Set(centerItems)].join(' and ')} ahead in center`);
    }
    if (leftItems.length > 0) {
        parts.push(`${[...new Set(leftItems)].join(' and ')} on your left`);
    }
    if (rightItems.length > 0) {
        parts.push(`${[...new Set(rightItems)].join(' and ')} on your right`);
    }

    const summaryKey = parts.join('; ');
    if (summaryKey && summaryKey !== lastSpokenSummary) {
        const phrase = "I see " + parts.join(', and ') + ".";
        speak(phrase);
        lastSpokenSummary = summaryKey;
        speechCooldown = now + 4500; // 4.5 second cooldown between spatial announcements
    }
}

// 10. Capture Snapshot for Multimodal Vision AI
function captureCameraSnapshot() {
    if (!video || video.videoWidth === 0) {
        throw new Error("Camera feed is not ready.");
    }

    const offscreen = document.createElement('canvas');
    const maxDim = 800; // Optimized size for vision tokenization & fast network transfer
    let w = video.videoWidth;
    let h = video.videoHeight;

    if (w > maxDim || h > maxDim) {
        if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
        } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
        }
    }

    offscreen.width = w;
    offscreen.height = h;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(video, 0, 0, w, h);

    return offscreen.toDataURL('image/jpeg', 0.85);
}

// 11. Multimodal Scene Intelligence via Secure /api/vision Proxy
async function inspectSceneWithVisionAI() {
    if (isInspectingScene) return;

    try {
        isInspectingScene = true;
        if (inspectSceneBtn) {
            inspectSceneBtn.disabled = true;
            inspectSceneBtn.classList.add('loading');
            inspectSceneBtn.querySelector('.btn-text').innerText = "ANALYZING SCENE...";
        }

        // Open AI scene panel in loading state
        if (aiScenePanel) {
            aiScenePanel.classList.remove('hidden');
        }
        if (aiSceneContent) {
            aiSceneContent.innerHTML = `<span style="color: #00F2FE;">🔍 Capturing camera snapshot and analyzing scene with NVIDIA Vision AI...</span>`;
        }

        speak("Analyzing scene with Vision AI.", true);

        // 1. Capture current video frame
        const snapshotBase64 = captureCameraSnapshot();
        const userLanguage = localStorage.getItem('argus_language_preference') || 'en';

        // 2. Query secure backend endpoint with 35s timeout
        const abortCtrl = new AbortController();
        const clientTimeout = setTimeout(() => abortCtrl.abort(), 35000);

        const response = await fetch('/api/vision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: snapshotBase64,
                mode: 'scene',
                language: userLanguage
            }),
            signal: abortCtrl.signal
        });

        clearTimeout(clientTimeout);

        const data = await response.json();

        if (!response.ok || !data.success) {
            const errorMsg = data.error || `Server returned error (${response.status})`;
            throw new Error(errorMsg);
        }

        // 3. Display Result
        lastSceneExplanation = data.description;
        if (aiSceneContent) {
            aiSceneContent.innerText = data.description;
        }

        // 4. Voice announcement
        speak(data.description, true);

    } catch (err) {
        console.error("Multimodal scene inspection error:", err);
        const isTimeout = err.name === 'AbortError';
        const displayErr = isTimeout ? 'Vision AI request timed out. Please try again.' : err.message;
        const userNotice = `⚠️ Could not complete scene analysis: ${displayErr}`;
        if (aiSceneContent) {
            aiSceneContent.innerText = userNotice;
        }
        speak("Scene analysis unavailable. Please try again.");
    } finally {
        isInspectingScene = false;
        if (inspectSceneBtn) {
            inspectSceneBtn.disabled = false;
            inspectSceneBtn.classList.remove('loading');
            inspectSceneBtn.querySelector('.btn-text').innerText = "INSPECT SCENE";
        }
    }
}

// 12. Wire UI Event Listeners
if (inspectSceneBtn) {
    inspectSceneBtn.addEventListener('click', inspectSceneWithVisionAI);
}

if (closeSceneBtn) {
    closeSceneBtn.addEventListener('click', () => {
        if (aiScenePanel) aiScenePanel.classList.add('hidden');
        if (synth && synth.speaking) synth.cancel();
    });
}

if (speakSceneBtn) {
    speakSceneBtn.addEventListener('click', () => {
        if (lastSceneExplanation) {
            speak(lastSceneExplanation, true);
        }
    });
}

// Confidence Slider
if (threshRange) {
    threshRange.value = 50;
    if (threshVal) threshVal.innerText = "50%";
    threshRange.addEventListener('input', (e) => {
        minConfidence = e.target.value / 100;
        if (threshVal) threshVal.innerText = e.target.value + "%";
    });
}

// Handle Visibility & Lifecycle
document.addEventListener('visibilitychange', () => {
    isPageVisible = !document.hidden;
});

// Start Engine
window.addEventListener('DOMContentLoaded', init);
