const state = {
    cameraActive: false,
    collectionActive: false,
    trainingActive: false,
    dashboardTimer: null,
    predictionTimer: null,
};

const elements = {
    connectionStatus: document.getElementById("connectionStatus"),
    cameraShell: document.getElementById("cameraShell"),
    cameraFeed: document.getElementById("cameraFeed"),
    cameraOverlay: document.getElementById("cameraOverlay"),
    cameraStateValue: document.getElementById("cameraStateValue"),
    modelStatusValue: document.getElementById("modelStatusValue"),
    samplesValue: document.getElementById("samplesValue"),
    labelsValue: document.getElementById("labelsValue"),
    predictionLabel: document.getElementById("predictionLabel"),
    predictionConfidence: document.getElementById("predictionConfidence"),
    predictionTimestamp: document.getElementById("predictionTimestamp"),
    trainingStatusPill: document.getElementById("trainingStatusPill"),
    trainingMessage: document.getElementById("trainingMessage"),
    trainingSamplesValue: document.getElementById("trainingSamplesValue"),
    trainingLabelsValue: document.getElementById("trainingLabelsValue"),
    gestureNameInput: document.getElementById("gestureNameInput"),
    collectionStatusValue: document.getElementById("collectionStatusValue"),
    currentGestureValue: document.getElementById("currentGestureValue"),
    currentGestureSamplesValue: document.getElementById("currentGestureSamplesValue"),
    recognitionLog: document.getElementById("recognitionLog"),
    startCameraBtn: document.getElementById("startCameraBtn"),
    stopCameraBtn: document.getElementById("stopCameraBtn"),
    trainModelBtn: document.getElementById("trainModelBtn"),
    trainSpinner: document.getElementById("trainSpinner"),
    trainButtonText: document.getElementById("trainButtonText"),
    startCollectBtn: document.getElementById("startCollectBtn"),
    stopCollectBtn: document.getElementById("stopCollectBtn"),
    toastRegion: document.getElementById("toastRegion"),
};

function formatNumber(value) {
    return new Intl.NumberFormat().format(value || 0);
}

function formatConfidence(value) {
    if (value === null || value === undefined) {
        return "--";
    }
    return `${Math.round(value * 100)}%`;
}

function formatTimestamp(timestamp) {
    if (!timestamp) {
        return "Waiting for data";
    }

    return new Date(timestamp * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        cache: "no-store",
        headers: {
            "Accept": "application/json",
            ...(options.headers || {}),
        },
        ...options,
    });

    let data = {};
    try {
        data = await response.json();
    } catch (error) {
        data = {};
    }

    if (!response.ok) {
        const message = data.message || `Request failed for ${url}`;
        throw new Error(message);
    }

    return data;
}

async function requestJsonWithFallback(urls, options = {}) {
    let lastError = null;

    for (const url of urls) {
        try {
            return await requestJson(url, options);
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error(`Request failed for ${urls[0]}`);
}

function setConnectionStatus(text, tone = "default") {
    elements.connectionStatus.textContent = text;
    elements.connectionStatus.dataset.tone = tone;
}

function showToast(message, tone = "success") {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.dataset.tone = tone;
    toast.textContent = message;
    elements.toastRegion.appendChild(toast);

    window.setTimeout(() => {
        toast.remove();
    }, 3200);
}

function syncButtonStates() {
    elements.startCameraBtn.disabled = state.cameraActive || state.trainingActive;
    elements.stopCameraBtn.disabled = !state.cameraActive || state.trainingActive;
    elements.startCollectBtn.disabled = state.collectionActive || state.trainingActive;
    elements.stopCollectBtn.disabled = !state.collectionActive || state.trainingActive;
    elements.trainModelBtn.disabled = state.trainingActive || state.collectionActive;
}

function setButtonBusy(button, isBusy, buttonTextElement, busyLabel, idleLabel, spinnerElement) {
    button.disabled = isBusy;
    if (buttonTextElement) {
        buttonTextElement.textContent = isBusy ? busyLabel : idleLabel;
    }
    if (spinnerElement) {
        spinnerElement.classList.toggle("hidden", !isBusy);
    }
}

function setCameraShell(stateName, title, subtitle) {
    elements.cameraShell.dataset.state = stateName;
    elements.cameraOverlay.innerHTML = `<strong>${title}</strong><span>${subtitle}</span>`;
}

function applyPrediction(data) {
    if (!data || !data.label) {
        elements.predictionLabel.textContent = "No prediction";
        elements.predictionConfidence.textContent = "--";
        elements.predictionTimestamp.textContent = "Waiting for data";
        return;
    }

    elements.predictionLabel.textContent = data.label;
    elements.predictionConfidence.textContent = formatConfidence(data.confidence);
    elements.predictionTimestamp.textContent = formatTimestamp(data.timestamp);
}

function applyStats(data) {
    elements.samplesValue.textContent = formatNumber(data.samples);
    elements.labelsValue.textContent = formatNumber(data.labels);
    elements.trainingSamplesValue.textContent = formatNumber(data.samples);
    elements.trainingLabelsValue.textContent = formatNumber(data.labels);
    elements.modelStatusValue.textContent = data.model_status === "trained" ? "Trained" : "Not trained";

    const trainingStatus = data.training_status || "idle";
    const trainingLabel = trainingStatus.charAt(0).toUpperCase() + trainingStatus.slice(1);
    state.trainingActive = trainingStatus === "training";
    state.collectionActive = Boolean(data.collection_active);
    elements.trainingStatusPill.textContent = trainingLabel;
    elements.trainingStatusPill.dataset.tone = trainingStatus;
    elements.trainingMessage.textContent = data.training_message || "Training idle.";

    elements.collectionStatusValue.textContent = data.collection_active ? "Collecting" : "Idle";
    elements.currentGestureValue.textContent = data.current_gesture || "--";
    elements.currentGestureSamplesValue.textContent = formatNumber(data.current_gesture_samples);
    syncButtonStates();
}

function applyLogs(logs) {
    if (!Array.isArray(logs) || logs.length === 0) {
        elements.recognitionLog.innerHTML = '<div class="empty-state">No recognition events yet.</div>';
        return;
    }

    elements.recognitionLog.innerHTML = logs.map((log) => `
        <div class="log-row">
            <span>${escapeHtml(log.label || "Unknown")}</span>
            <span>${formatConfidence(log.confidence)}</span>
            <span>${formatTimestamp(log.timestamp)}</span>
        </div>
    `).join("");
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

async function fetchPrediction() {
    try {
        const data = await requestJsonWithFallback(["/predict", "/prediction/status"]);
        if (Array.isArray(data.predictions)) {
            const leadPrediction = data.predictions[0];
            applyPrediction(
                leadPrediction
                    ? {
                          label: leadPrediction.label,
                          confidence: leadPrediction.confidence,
                          timestamp: data.history && data.history[0] ? data.history[0].timestamp : null,
                      }
                    : {}
            );
        } else {
            applyPrediction(data);
        }
        setConnectionStatus("Backend online");
    } catch (error) {
        applyPrediction({});
        setConnectionStatus("Prediction unavailable", "error");
    }
}

async function fetchStats() {
    try {
        let data;
        try {
            data = await requestJson("/stats");
        } catch (primaryError) {
            const [modelInfo, collectionInfo] = await Promise.all([
                requestJson("/model-info"),
                requestJson("/collection/status"),
            ]);
            data = {
                samples: modelInfo.training_sample_count || 0,
                labels: modelInfo.label_count || 0,
                model_status: modelInfo.model_available ? "trained" : "not_trained",
                training_status: "idle",
                training_message: "Training idle.",
                current_gesture: collectionInfo.label || "",
                current_gesture_samples: collectionInfo.count || 0,
                collection_active: Boolean(collectionInfo.active),
            };
        }
        applyStats(data);
        setConnectionStatus("Backend online");
    } catch (error) {
        setConnectionStatus("Stats unavailable", "error");
    }
}

async function fetchLogs() {
    try {
        const data = await requestJsonWithFallback(["/logs", "/prediction/status"]);
        applyLogs(data.logs || data.history || []);
        setConnectionStatus("Backend online");
    } catch (error) {
        elements.recognitionLog.innerHTML = '<div class="empty-state">Unable to load recognition history.</div>';
        setConnectionStatus("Logs unavailable", "error");
    }
}

async function syncDashboard() {
    await Promise.allSettled([
        fetchStats(),
        fetchLogs(),
        fetchPrediction(),
    ]);
}

function startPredictionPolling() {
    if (state.predictionTimer) {
        window.clearInterval(state.predictionTimer);
    }

    state.predictionTimer = window.setInterval(() => {
        fetchPrediction();
        fetchLogs();
        fetchStats();
    }, 1200);
}

function stopPredictionPolling() {
    if (state.predictionTimer) {
        window.clearInterval(state.predictionTimer);
        state.predictionTimer = null;
    }
}

function startCamera() {
    if (state.cameraActive) {
        return;
    }

    state.cameraActive = true;
    elements.cameraFeed.src = `/video_feed?ts=${Date.now()}`;
    elements.cameraStateValue.textContent = "Starting";
    setCameraShell("loading", "Starting camera", "Connecting to the live Flask stream.");
    syncButtonStates();
    startPredictionPolling();
}

async function stopCamera() {
    if (state.collectionActive) {
        await stopCollection({ silent: true, preserveGesture: true });
    }

    state.cameraActive = false;
    stopPredictionPolling();
    elements.cameraFeed.removeAttribute("src");
    elements.cameraFeed.src = "";
    elements.cameraStateValue.textContent = "Stopped";
    applyPrediction({});
    setCameraShell("idle", "Camera is off", "Start the camera to begin recognition.");
    syncButtonStates();
}

async function trainModel() {
    state.trainingActive = true;
    syncButtonStates();
    setButtonBusy(
        elements.trainModelBtn,
        true,
        elements.trainButtonText,
        "Training...",
        "Train Model",
        elements.trainSpinner
    );
    elements.trainingStatusPill.textContent = "Training";
    elements.trainingStatusPill.dataset.tone = "training";
    elements.trainingMessage.textContent = "Training in progress...";
    showToast("Training started...", "success");

    try {
        const data = await requestJson("/train", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({}),
        });
        if (data.stats) {
            applyStats(data.stats);
        } else {
            await fetchStats();
        }
        showToast("Training completed", "success");
        await syncDashboard();
    } catch (error) {
        state.trainingActive = false;
        elements.trainingStatusPill.textContent = "Failed";
        elements.trainingStatusPill.dataset.tone = "failed";
        elements.trainingMessage.textContent = error.message;
        showToast(error.message, "error");
    } finally {
        state.trainingActive = false;
        setButtonBusy(
            elements.trainModelBtn,
            false,
            elements.trainButtonText,
            "Training...",
            "Train Model",
            elements.trainSpinner
        );
        syncButtonStates();
    }
}

async function startCollection() {
    const label = elements.gestureNameInput.value.trim();
    if (!label) {
        showToast("Enter a gesture name first.", "error");
        elements.gestureNameInput.focus();
        return;
    }

    if (!state.cameraActive) {
        startCamera();
        showToast("Camera started for data collection.", "success");
    }

    elements.startCollectBtn.disabled = true;

    try {
        const data = await requestJsonWithFallback(["/collect/start", "/collection/start"], {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ label }),
        });

        state.collectionActive = Boolean(data.active);
        elements.collectionStatusValue.textContent = data.active ? "Collecting" : "Idle";
        elements.currentGestureValue.textContent = data.label || label;
        elements.currentGestureSamplesValue.textContent = formatNumber(data.count);
        showToast(data.message || "Collection started.", "success");
        await fetchStats();
    } catch (error) {
        showToast(error.message, "error");
    } finally {
        syncButtonStates();
    }
}

async function stopCollection(options = {}) {
    const { silent = false, preserveGesture = false } = options;
    elements.stopCollectBtn.disabled = true;

    try {
        const data = await requestJsonWithFallback(["/collect/stop", "/collection/stop"], {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({}),
        });

        state.collectionActive = Boolean(data.active);
        elements.collectionStatusValue.textContent = data.active ? "Collecting" : "Idle";
        elements.currentGestureValue.textContent = preserveGesture ? (elements.gestureNameInput.value.trim() || "--") : (data.label || "--");
        elements.currentGestureSamplesValue.textContent = formatNumber(data.count);
        if (!silent) {
            showToast(data.message || "Collection stopped.", "success");
        }
        await syncDashboard();
    } catch (error) {
        if (!silent) {
            showToast(error.message, "error");
        }
    } finally {
        syncButtonStates();
    }
}

function bindEvents() {
    elements.startCameraBtn.addEventListener("click", startCamera);
    elements.stopCameraBtn.addEventListener("click", stopCamera);
    elements.trainModelBtn.addEventListener("click", trainModel);
    elements.startCollectBtn.addEventListener("click", startCollection);
    elements.stopCollectBtn.addEventListener("click", stopCollection);

    elements.cameraFeed.addEventListener("load", () => {
        if (!state.cameraActive) {
            return;
        }
        elements.cameraStateValue.textContent = "Live";
        setCameraShell("live", "", "");
        syncButtonStates();
    });

    elements.cameraFeed.addEventListener("error", () => {
        if (!state.cameraActive) {
            return;
        }
        elements.cameraStateValue.textContent = "Unavailable";
        setCameraShell("error", "Camera unavailable", "Check webcam permissions or close other apps using the camera.");
        state.cameraActive = false;
        stopPredictionPolling();
        syncButtonStates();
    });
}

async function initializeApp() {
    bindEvents();
    await stopCamera();
    await syncDashboard();
    syncButtonStates();

    state.dashboardTimer = window.setInterval(() => {
        syncDashboard();
    }, 2500);
}

window.addEventListener("DOMContentLoaded", initializeApp);
