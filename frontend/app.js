/**
 * DeepFake Forensics - Web Interface Script
 * Handles image upload, backend API prediction, Grad-CAM visualization, and verdict output.
 */

document.addEventListener("DOMContentLoaded", () => {
    const API_BASE_URL = window.location.origin.startsWith("http") &&
        !window.location.origin.includes(":5500") &&
        !window.location.origin.includes(":3000")
        ? window.location.origin
        : "http://127.0.0.1:8000";

    // DOM Elements - Upload & Controls
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropzoneEmpty = document.getElementById("dropzoneEmpty");
    const previewBox = document.getElementById("previewBox");
    const previewImage = document.getElementById("previewImage");
    const fileNameEl = document.getElementById("fileName");
    const fileSizeEl = document.getElementById("fileSize");
    const replaceBtn = document.getElementById("replaceBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const sampleRealBtn = document.getElementById("sampleRealBtn");
    const sampleFakeBtn = document.getElementById("sampleFakeBtn");

    // Loading & Status
    const loadingCard = document.getElementById("loadingCard");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    // Results - Grad-CAM & Verdict
    const analysisResultsWrapper = document.getElementById("analysisResultsWrapper");
    const resultStepCard = document.getElementById("resultStepCard");
    const gradcamFacesContainer = document.getElementById("gradcamFacesContainer");
    const outcomeBadge = document.getElementById("outcomeBadge");
    const scoreValue = document.getElementById("scoreValue");
    const explanationIcon = document.getElementById("explanationIcon");
    const explanationTitle = document.getElementById("explanationTitle");
    const explanationText = document.getElementById("explanationText");
    const resetBtn = document.getElementById("resetBtn");

    // Alert Banner
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");
    const closeAlertBtn = document.getElementById("closeAlertBtn");

    let currentFile = null;

    const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
    const ALLOWED_MIME_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB

    // Health Check
    async function checkApiHealth() {
        try {
            const resp = await fetch(`${API_BASE_URL}/health`);
            if (resp.ok) {
                const data = await resp.json();
                statusDot.className = "status-dot online";
                statusText.textContent = `API Ready (${data.device.toUpperCase()})`;
            } else {
                statusDot.className = "status-dot offline";
                statusText.textContent = "API Degraded";
            }
        } catch (_) {
            statusDot.className = "status-dot offline";
            statusText.textContent = "API Offline (Start Backend on :8000)";
        }
    }

    checkApiHealth();

    // Alerts
    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove("hidden");
        errorAlert.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function hideError() {
        errorAlert.classList.add("hidden");
    }

    closeAlertBtn.addEventListener("click", hideError);

    // File Validation & Preview
    function validateFile(file) {
        if (!file) return "No file selected.";

        const nameLower = file.name.toLowerCase();
        const hasValidExt = ALLOWED_EXTENSIONS.some(ext => nameLower.endsWith(ext));
        const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type);

        if (!hasValidExt && !hasValidMime) {
            return "Unsupported file format. Please upload a JPG, PNG, or WEBP image.";
        }

        if (file.size === 0) {
            return "The selected file is empty (0 bytes).";
        }

        if (file.size > MAX_FILE_SIZE_BYTES) {
            return `File size exceeds the 20MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB).`;
        }

        return null;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        const kb = bytes / 1024;
        if (kb < 1024) return kb.toFixed(1) + " KB";
        return (kb / 1024).toFixed(1) + " MB";
    }

    function handleFileSelection(file, autoAnalyze = false) {
        hideError();
        const err = validateFile(file);
        if (err) {
            showError(err);
            resetUploadState();
            return;
        }

        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            fileNameEl.textContent = file.name;
            fileSizeEl.textContent = formatFileSize(file.size);

            dropzoneEmpty.classList.add("hidden");
            previewBox.classList.remove("hidden");
            analysisResultsWrapper.classList.add("hidden");

            if (autoAnalyze) {
                runInference();
            }
        };
        reader.onerror = () => {
            showError("Failed to read the image file.");
            resetUploadState();
        };
        reader.readAsDataURL(file);
    }

    function resetUploadState() {
        currentFile = null;
        fileInput.value = "";
        previewImage.src = "";
        dropzoneEmpty.classList.remove("hidden");
        previewBox.classList.add("hidden");
        loadingCard.classList.add("hidden");
        analysisResultsWrapper.classList.add("hidden");
        gradcamFacesContainer.innerHTML = "";
    }

    // Event Listeners: Drag & Drop & Browse
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    replaceBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    dropzone.addEventListener("click", (e) => {
        if (!previewBox.classList.contains("hidden")) return;
        if (e.target.closest(".sample-btn")) return;
        fileInput.click();
    });

    // Sample Image Buttons
    async function loadSample(samplePath, sampleName) {
        try {
            hideError();
            loadingCard.classList.remove("hidden");
            const resp = await fetch(samplePath);
            if (!resp.ok) {
                throw new Error(`Sample file not found (${resp.status})`);
            }
            const blob = await resp.blob();
            const file = new File([blob], sampleName, { type: "image/jpeg" });
            loadingCard.classList.add("hidden");
            handleFileSelection(file, true);
        } catch (err) {
            loadingCard.classList.add("hidden");
            showError(`Could not load sample: ${err.message}. Please upload an image directly.`);
        }
    }

    sampleRealBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        loadSample("/static/samples/real_sample.jpg", "sample_real_face.jpg");
    });

    sampleFakeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        loadSample("/static/samples/fake_sample.jpg", "sample_fake_face.jpg");
    });

    analyzeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        runInference();
    });

    resetBtn.addEventListener("click", () => {
        resetUploadState();
        document.getElementById("uploadStepCard").scrollIntoView({ behavior: "smooth" });
    });

    // Inference Execution
    async function runInference() {
        if (!currentFile) {
            showError("Please select or drop an image first.");
            return;
        }

        hideError();
        analysisResultsWrapper.classList.add("hidden");
        loadingCard.classList.remove("hidden");
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const resp = await fetch(`${API_BASE_URL}/predict/image?include_gradcam=true`, {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) {
                let detail = `Server responded with status ${resp.status}`;
                try {
                    const errData = await resp.json();
                    if (errData.detail) detail = errData.detail;
                } catch (_) {}
                throw new Error(detail);
            }

            const data = await resp.json();
            renderResults(data);
        } catch (err) {
            showError(`Analysis failed: ${err.message}`);
        } finally {
            loadingCard.classList.add("hidden");
            analyzeBtn.disabled = false;
        }
    }

    // Render Results: Grad-CAM first, then Verdict Result
    function renderResults(data) {
        const predictions = data.predictions || [];
        const finalPred = data.final_prediction || predictions[0];
        gradcamFacesContainer.innerHTML = "";

        if (predictions.length === 0) {
            showError("No face could be analyzed in this image.");
            return;
        }

        // 1. Render Grad-CAM Section (Step 2)
        [finalPred].forEach((pred, index) => {
            const block = document.createElement("div");
            block.className = "face-gradcam-block";

            const faceLabel = pred.label || "INCONCLUSIVE";
            const faceHeader = `
                <div class="face-gradcam-header">
                    <span class="face-index-tag">Fused face analysis</span>
                    <span class="face-mini-badge ${faceLabel}">${faceLabel}</span>
                </div>
            `;

            let visualContent = "";
            if (pred.face_crop_base64 && pred.gradcam_heatmap_base64) {
                visualContent = `
                    <div class="gradcam-visual-grid">
                        <div class="visual-item">
                            <span class="visual-item-title">1. Detected Face (SCRFD)</span>
                            <div class="visual-frame">
                                <img src="${pred.face_crop_base64}" alt="Face Crop">
                            </div>
                            <span class="visual-item-caption">Isolated Face Region (224x224)</span>
                        </div>

                        <div class="visual-item">
                            <span class="visual-item-title">2. Grad-CAM Heatmap</span>
                            <div class="visual-frame">
                                <img src="${pred.gradcam_heatmap_base64}" alt="Grad-CAM Heatmap">
                            </div>
                            <span class="visual-item-caption">Feature Attribution (Turbo)</span>
                        </div>

                        <div class="visual-item">
                            <span class="visual-item-title">3. Attention Overlay</span>
                            <div class="visual-frame">
                                <img src="${pred.gradcam_overlay_base64 || pred.face_crop_base64}" alt="Grad-CAM Overlay">
                            </div>
                            <span class="visual-item-caption">Active Regions Blended (Transparent Base)</span>
                        </div>
                    </div>

                    <div class="gradcam-legend-bar">
                        <span><strong>Model Activation:</strong></span>
                        <div class="legend-spectrum">
                            <span>Low (Transparent)</span>
                            <div class="spectrum-track"></div>
                            <span>High Activation</span>
                        </div>
                    </div>
                `;
            } else if (pred.face_crop_base64) {
                visualContent = `
                    <div style="text-align: center; padding: 20px;">
                        <img src="${pred.face_crop_base64}" alt="Face Crop" style="max-width: 180px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <p style="margin-top: 10px; font-size: 0.88rem; color: #64748b;">
                            Grad-CAM is not available for this face because it did not meet quality thresholds (${pred.reason ? pred.reason.replace(/_/g, " ") : "low quality"}).
                        </p>
                    </div>
                `;
            } else {
                visualContent = `
                    <div class="alert alert-error" style="background-color: #fffbeb; border-color: #fde68a; color: #92400e;">
                        <span>⚠️ No usable face crop could be isolated. Reason: <strong>${pred.reason ? pred.reason.replace(/_/g, " ") : "no face detected"}</strong>.</span>
                    </div>
                `;
            }

            block.innerHTML = faceHeader + visualContent;
            gradcamFacesContainer.appendChild(block);
        });

        // 2. Render Final Verdict (Step 3: Fake or Real)
        const primaryPred = finalPred;
        const overallLabel = primaryPred.label || "INCONCLUSIVE";

        outcomeBadge.textContent = overallLabel === "FAKE" ? "FAKE (MANIPULATED)" :
                                   overallLabel === "REAL" ? "REAL (AUTHENTIC)" :
                                   "INCONCLUSIVE";
        outcomeBadge.className = `outcome-badge ${overallLabel}`;

        resultStepCard.className = `card step-card result-card verdict-${overallLabel.toLowerCase()}`;

        if (primaryPred.score !== null && primaryPred.score !== undefined) {
            const scorePct = (primaryPred.score * 100).toFixed(1);
            scoreValue.textContent = `${scorePct}%`;
        } else {
            scoreValue.textContent = "--%";
        }

        if (overallLabel === "FAKE") {
            explanationIcon.textContent = "🚨";
            explanationTitle.textContent = "Synthetic / AI-Generated Face Detected";
            explanationText.textContent = "Two independent classifiers agreed this face is manipulated. The Grad-CAM heatmap shows the EfficientNet-B0 regions supporting its vote.";
        } else if (overallLabel === "REAL") {
            explanationIcon.textContent = "✅";
            explanationTitle.textContent = "Authentic Portrait Verified";
            explanationText.textContent = "Two independent classifiers agreed this face is authentic. The Grad-CAM heatmap shows the EfficientNet-B0 regions supporting its vote.";
        } else {
            explanationIcon.textContent = "⚠️";
            explanationTitle.textContent = "Inconclusive Analysis";
            explanationText.textContent = `The system could not make a confident classification (${primaryPred.reason ? primaryPred.reason.replace(/_/g, " ") : "confidence below threshold"}). Please ensure the uploaded image has a clearly visible, well-lit human face.`;
        }

        analysisResultsWrapper.classList.remove("hidden");
        document.getElementById("gradcamStepCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
});
