/* ==========================================================================
   POSE MEME APP - PREMIUM CLIENT ENGINE
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // DOM Elements
    const btnCamera = document.getElementById('btn-camera');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    const loadingOverlay = document.getElementById('loading-overlay');
    const promptMsg = document.getElementById('prompt-msg');

    // Sidebar Metrics Elements
    const metricLatency = document.getElementById('metric-latency');
    const metricFps = document.getElementById('metric-fps');

    // Meme Matching Elements
    const primaryMeme = document.getElementById('primary-meme');
    const primaryMemeImg = document.getElementById('primary-meme-img');
    const primaryScore = document.getElementById('primary-score');
    const primaryTitle = document.getElementById('primary-title');
    const primarySource = document.getElementById('primary-source');
    const mainMemeContainer = primaryMeme.querySelector('.meme-image-container');
    const mainMemePlaceholder = primaryMeme.querySelector('.meme-placeholder');

    const secMatch1 = document.getElementById('sec-match-1');
    const secMatch2 = document.getElementById('sec-match-2');

    // State Variables
    let stream = null;
    let ws = null;
    let active = false;
    let isProcessing = false;
    let lastFrameTime = Date.now();
    let frameSentTime = 0;

    // FPS Tracking variables
    let fpsInterval = 1000; // 1 second
    let fpsTicks = 0;
    let lastFpsTime = Date.now();

    // Landmark tracking cache (for smoothing skeleton drawing between responses)
    let currentLandmarks = null;

    // --- Start / Stop Webcam Stream ---
    btnCamera.addEventListener('click', async () => {
        if (!active) {
            await startWebcam();
        } else {
            stopWebcam();
        }
    });

    async function startWebcam() {
        promptMsg.textContent = "Requesting camera access...";

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                },
                audio: false
            });

            const onVideoReady = () => {
                video.play().then(() => {
                    console.log("[Camera] Playback started successfully");
                }).catch(err => {
                    console.error("[Camera] Playback failed:", err);
                });
                
                canvas.width = video.videoWidth > 0 ? video.videoWidth : 640;
                canvas.height = video.videoHeight > 0 ? video.videoHeight : 480;
                loadingOverlay.classList.remove('active');
                
                // Initialize WebSocket connection
                connectWebSocket();
            };

            // Set up handler first to prevent race conditions
            video.onloadedmetadata = onVideoReady;
            
            // Assign stream
            video.srcObject = stream;
            
            // Double check if already loaded
            if (video.readyState >= 1) {
                onVideoReady();
            }

            active = true;
            btnCamera.classList.add('recording');
            btnCamera.querySelector('span').textContent = "Stop Webcam";
            const cameraIcon = btnCamera.querySelector('i') || btnCamera.querySelector('svg');
            if (cameraIcon) {
                cameraIcon.setAttribute('data-lucide', 'square');
            }
            lucide.createIcons();

            statusDot.className = "status-dot online";
            statusText.textContent = "Camera active";

            // Start local rendering loop
            requestAnimationFrame(renderLoop);

        } catch (err) {
            console.error("Camera access failed:", err);
            promptMsg.textContent = "Camera access denied. Please allow permissions and retry.";
            loadingOverlay.classList.add('active');
            statusDot.className = "status-dot";
            statusText.textContent = "Camera offline";
        }
    }

    function stopWebcam() {
        active = false;
        isProcessing = false;

        // Stop video tracks
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        // Close WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }

        btnCamera.classList.remove('recording');
        btnCamera.querySelector('span').textContent = "Start Webcam";
        const cameraIcon = btnCamera.querySelector('i') || btnCamera.querySelector('svg');
        if (cameraIcon) {
            cameraIcon.setAttribute('data-lucide', 'video');
        }
        lucide.createIcons();

        statusDot.className = "status-dot";
        statusText.textContent = "Camera offline";

        promptMsg.textContent = 'Click "Start Webcam" to begin matching';
        loadingOverlay.classList.add('active');

        // Reset matches UI
        resetMatchesUI();
        currentLandmarks = null;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // --- WebSocket Pose Processor ---
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/stream`;

        ws = new WebSocket(wsUrl);
        ws.binaryType = "blob";

        ws.onopen = () => {
            console.log("[WebSocket] Connection established");
            statusDot.className = "status-dot processing";
            statusText.textContent = "Tracking ready";
        };

        ws.onmessage = (event) => {
            // Calculate round trip latency
            const latency = Date.now() - frameSentTime;
            metricLatency.textContent = `${latency} ms`;

            try {
                const data = JSON.parse(event.data);

                if (data.matched) {
                    currentLandmarks = data.landmarks;
                    updateMemeMatches(data.matches);
                    statusDot.className = "status-dot online";
                    statusText.textContent = "Pose Matched";
                } else {
                    // Update status prompts on video frame
                    statusText.textContent = data.reason || "Analyzing...";
                    statusDot.className = "status-dot processing";
                }
            } catch (err) {
                console.error("Error parsing WebSocket message:", err);
            }

            // Release the lock to allow sending the next frame
            isProcessing = false;
        };

        ws.onerror = (err) => {
            console.error("[WebSocket] Connection error:", err);
        };

        ws.onclose = () => {
            console.log("[WebSocket] Connection closed");
            if (active) {
                statusDot.className = "status-dot";
                statusText.textContent = "Stream disconnected";
            }
        };
    }

    // Send a frame to the backend via WebSocket
    function sendFrame() {
        if (!ws || ws.readyState !== WebSocket.OPEN || isProcessing) return;

        // Draw webcam image to capture it
        // We use a temporary capture canvas at 480x360 for high performance MediaPipe processing
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 480;
        captureCanvas.height = 360;
        const captureCtx = captureCanvas.getContext('2d');
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        // Convert to JPEG blob and transmit
        isProcessing = true;
        frameSentTime = Date.now();

        captureCanvas.toBlob((blob) => {
            if (blob && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(blob);
            } else {
                isProcessing = false;
            }
        }, 'image/jpeg', 0.7); // 70% quality compresses frames heavily for rapid streaming
    }

    // --- Render Loop (Webcam + Glowing Skeleton Overlay) ---
    function renderLoop() {
        if (!active) return;

        try {
            // Dynamically correct canvas resolution if video size was initialized late
            if (video.videoWidth > 0 && canvas.width !== video.videoWidth) {
                console.log(`[Canvas] Correcting resolution to match video: ${video.videoWidth}x${video.videoHeight}`);
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }

            // Draw skeleton overlay (only if dimensions are valid)
            if (canvas.width > 0 && canvas.height > 0) {
                // Clear the canvas overlay completely to keep it transparent
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw cached pose landmarks on top of mirrored webcam
                if (currentLandmarks) {
                    drawSkeleton(currentLandmarks);
                }

                // Calculate and display FPS metrics
                fpsTicks++;
                const now = Date.now();
                if (now - lastFpsTime >= fpsInterval) {
                    metricFps.textContent = Math.round((fpsTicks * 1000) / (now - lastFpsTime));
                    fpsTicks = 0;
                    lastFpsTime = now;
                }

                // Check if we should transmit a new frame
                const timeSinceLastFrame = now - lastFrameTime;
                if (timeSinceLastFrame > 100) { // Limit to 10 FPS transmission to maximize local performance
                    sendFrame();
                    lastFrameTime = now;
                }
            }
        } catch (err) {
            console.error("[RenderLoop] Error in draw/render loop:", err);
        }

        requestAnimationFrame(renderLoop);
    }

    // Helper to scale landmark X coordinate from 480px server space to canvas width
    function getCanvasX(val) {
        return val * (canvas.width / 480);
    }

    // Helper to scale landmark Y coordinate from 360px server space to canvas height
    function getCanvasY(val) {
        return val * (canvas.height / 360);
    }

    // --- Draw Skeleton on Canvas ---
    function drawSkeleton(landmarks) {
        if (!landmarks || !landmarks.body || landmarks.body.length === 0) return;

        const body = landmarks.body;
        const leftHand = landmarks.left_hand;
        const rightHand = landmarks.right_hand;

        // Body Upper-Body Skeleton joints (MediaPipe Indices)
        const leftShoulder = body[11];
        const rightShoulder = body[12];
        const leftElbow = body[13];
        const rightElbow = body[14];
        const leftWrist = body[15];
        const rightWrist = body[16];

        // Draw Body Skeleton Lines
        ctx.lineWidth = 4;
        ctx.shadowBlur = 10;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Glowing Purple for Body Joints
        ctx.strokeStyle = '#8B5CF6';
        ctx.shadowColor = '#8B5CF6';

        // Connect Shoulders
        ctx.beginPath();
        ctx.moveTo(getCanvasX(leftShoulder[0]), getCanvasY(leftShoulder[1]));
        ctx.lineTo(getCanvasX(rightShoulder[0]), getCanvasY(rightShoulder[1]));
        ctx.stroke();

        // Connect Left Arm (Shoulder -> Elbow -> Wrist)
        ctx.beginPath();
        ctx.moveTo(getCanvasX(leftShoulder[0]), getCanvasY(leftShoulder[1]));
        ctx.lineTo(getCanvasX(leftElbow[0]), getCanvasY(leftElbow[1]));
        ctx.lineTo(getCanvasX(leftWrist[0]), getCanvasY(leftWrist[1]));
        ctx.stroke();

        // Connect Right Arm (Shoulder -> Elbow -> Wrist)
        ctx.beginPath();
        ctx.moveTo(getCanvasX(rightShoulder[0]), getCanvasY(rightShoulder[1]));
        ctx.lineTo(getCanvasX(rightElbow[0]), getCanvasY(rightElbow[1]));
        ctx.lineTo(getCanvasX(rightWrist[0]), getCanvasY(rightWrist[1]));
        ctx.stroke();

        // Draw Body Joint Circles
        ctx.fillStyle = '#FFFFFF';
        ctx.shadowBlur = 5;
        const joints = [leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist];
        joints.forEach(j => {
            ctx.beginPath();
            ctx.arc(getCanvasX(j[0]), getCanvasY(j[1]), 6, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Draw Hands if detected
        if (landmarks.left_hand_detected && leftHand.length > 0) {
            drawHand(leftHand);
        }
        if (landmarks.right_hand_detected && rightHand.length > 0) {
            drawHand(rightHand);
        }

        // Reset shadow blur
        ctx.shadowBlur = 0;
    }

    // --- Draw Hand Skeleton ---
    function drawHand(handLandmarks) {
        const fingers = [
            [0, 1, 2, 3, 4],     // Thumb
            [0, 5, 6, 7, 8],     // Index
            [0, 9, 10, 11, 12],  // Middle
            [0, 13, 14, 15, 16], // Ring
            [0, 17, 18, 19, 20]  // Pinky
        ];

        // Glowing Cyan for hands
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#06B6D4';
        ctx.shadowColor = '#06B6D4';
        ctx.shadowBlur = 8;

        fingers.forEach(finger => {
            ctx.beginPath();
            ctx.moveTo(getCanvasX(handLandmarks[finger[0]][0]), getCanvasY(handLandmarks[finger[0]][1]));
            for (let i = 1; i < finger.length; i++) {
                ctx.lineTo(getCanvasX(handLandmarks[finger[i]][0]), getCanvasY(handLandmarks[finger[i]][1]));
            }
            ctx.stroke();
        });

        // Draw knuckles joints
        ctx.fillStyle = '#FFFFFF';
        ctx.shadowBlur = 3;
        handLandmarks.forEach(pt => {
            ctx.beginPath();
            ctx.arc(getCanvasX(pt[0]), getCanvasY(pt[1]), 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // --- Update Matching Memes UI ---
    function updateMemeMatches(matches) {
        if (!matches || matches.length === 0) return;

        // 1. Update Primary Match (Top Meme Card)
        const topMatch = matches[0];
        primaryTitle.textContent = topMatch.name;
        primarySource.textContent = `${topMatch.source.toUpperCase()} template`;
        primaryScore.textContent = `${Math.round(topMatch.similarity * 100)}%`;

        primaryMemeImg.src = topMatch.url;
        mainMemePlaceholder.style.display = 'none';
        mainMemeContainer.style.display = 'flex';

        // 2. Update Secondary Matches
        if (matches.length > 1) {
            updateSecondaryCard(secMatch1, matches[1]);
        } else {
            resetSecondaryCard(secMatch1, "No Match #2");
        }

        if (matches.length > 2) {
            updateSecondaryCard(secMatch2, matches[2]);
        } else {
            resetSecondaryCard(secMatch2, "No Match #3");
        }
    }

    function updateSecondaryCard(cardEl, match) {
        cardEl.classList.remove('disabled');

        const imgHolder = cardEl.querySelector('.sec-img-holder');
        imgHolder.innerHTML = `<img src="${match.url}" alt="${match.name}">`;

        const title = cardEl.querySelector('.sec-title');
        title.textContent = match.name;

        const fill = cardEl.querySelector('.sec-score-fill');
        const pct = Math.round(match.similarity * 100);
        fill.style.width = `${pct}%`;

        const text = cardEl.querySelector('.sec-score-text');
        text.textContent = `${pct}% Match`;
    }

    function resetSecondaryCard(cardEl, placeholderTitle) {
        cardEl.classList.add('disabled');

        const imgHolder = cardEl.querySelector('.sec-img-holder');
        imgHolder.innerHTML = `<i data-lucide="image"></i>`;

        const title = cardEl.querySelector('.sec-title');
        title.textContent = placeholderTitle;

        const fill = cardEl.querySelector('.sec-score-fill');
        fill.style.width = `0%`;

        const text = cardEl.querySelector('.sec-score-text');
        text.textContent = `0% Match`;

        lucide.createIcons();
    }

    function resetMatchesUI() {
        primaryTitle.textContent = "Awaiting Pose...";
        primarySource.textContent = "-";
        primaryScore.textContent = "0%";
        mainMemeContainer.style.display = 'none';
        mainMemePlaceholder.style.display = 'flex';

        resetSecondaryCard(secMatch1, "No Match #2");
        resetSecondaryCard(secMatch2, "No Match #3");
    }
});
