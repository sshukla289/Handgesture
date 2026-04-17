const numberFormatter = new Intl.NumberFormat();
const timeFormatter = new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
});

const state = {
    cameraActive: false,
    collectionActive: false,
    trainingActive: false,
    logCollapsed: false,
    latestStats: null,
    modelMetadata: null,
    predictionTimer: null,
    collectionTimer: null,
    trainingTimer: null,
    dashboardTimer: null,
    fetchingPrediction: false,
    fetchingCollection: false,
    fetchingStats: false,
    fetchingTraining: false,
    lastTrainingTerminalStatus: "",
};

const $ = (id) => document.getElementById(id);

const elements = {
    connectionStatus: $("connectionStatus"),
    headerModelStatusChip: $("headerModelStatusChip"),
    cameraShell: $("cameraShell"),
    cameraFeed: $("cameraFeed"),
    cameraOverlay: $("cameraOverlay"),
    cameraAlert: $("cameraAlert"),
    exitFullscreenCameraBtn: $("exitFullscreenCameraBtn"),
    streamHudChip: $("streamHudChip"),
    predictionHudChip: $("predictionHudChip"),
    cameraPredictionOverlay: $("cameraPredictionOverlay"),
    cameraPredictionLabel: $("cameraPredictionLabel"),
    cameraPredictionMeta: $("cameraPredictionMeta"),
    cameraStateValue: $("cameraStateValue"),
    cameraStatusMeta: $("cameraStatusMeta"),
    modelStatusValue: $("modelStatusValue"),
    modelStatusMeta: $("modelStatusMeta"),
    samplesValue: $("samplesValue"),
    samplesMeta: $("samplesMeta"),
    labelsValue: $("labelsValue"),
    labelsMeta: $("labelsMeta"),
    predictionLabel: $("predictionLabel"),
    predictionConfidence: $("predictionConfidence"),
    predictionTimestamp: $("predictionTimestamp"),
    predictionSummary: $("predictionSummary"),
    trainingStatusPill: $("trainingStatusPill"),
    trainingMessage: $("trainingMessage"),
    trainingRefreshState: $("trainingRefreshState"),
    trainingProgressFill: $("trainingProgressFill"),
    trainingStageValue: $("trainingStageValue"),
    trainingElapsedValue: $("trainingElapsedValue"),
    trainingProgressValue: $("trainingProgressValue"),
    trainingSourceSelect: $("trainingSourceSelect"),
    trainingSourceValue: $("trainingSourceValue"),
    trainingStageSummary: $("trainingStageSummary"),
    trainingSamplesValue: $("trainingSamplesValue"),
    trainingLabelsValue: $("trainingLabelsValue"),
    trainingCsvValue: $("trainingCsvValue"),
    trainingAccuracyValue: $("trainingAccuracyValue"),
    clearDatasetBtn: $("clearDatasetBtn"),
    gestureNameInput: $("gestureNameInput"),
    collectionStatusChip: $("collectionStatusChip"),
    collectionSummary: $("collectionSummary"),
    collectionStatusValue: $("collectionStatusValue"),
    currentGestureValue: $("currentGestureValue"),
    currentGestureSamplesValue: $("currentGestureSamplesValue"),
    logPanel: $("logPanel"),
    recognitionLog: $("recognitionLog"),
    logCaption: $("logCaption"),
    logCollapseBtn: $("logCollapseBtn"),
    logCollapseLabel: $("logCollapseLabel"),
    datasetRuleNote: $("datasetRuleNote"),
    startCameraBtn: $("startCameraBtn"),
    fullscreenCameraBtn: $("fullscreenCameraBtn"),
    stopCameraBtn: $("stopCameraBtn"),
    trainModelBtn: $("trainModelBtn"),
    trainSpinner: $("trainSpinner"),
    trainButtonText: $("trainButtonText"),
    startCollectBtn: $("startCollectBtn"),
    stopCollectBtn: $("stopCollectBtn"),
    toastRegion: $("toastRegion"),
};

function patchText(element, value) {
    if (!element) {
        return;
    }

    const nextValue = String(value ?? "");
    if (element.textContent !== nextValue) {
        element.textContent = nextValue;
    }
}

function setTone(element, tone) {
    if (element) {
        element.dataset.tone = tone;
    }
}

function formatNumber(value) {
    return numberFormatter.format(value || 0);
}

function formatConfidence(value) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${Math.round(value * 100)}%`;
}

function formatAccuracy(value) {
    if (value === null || value === undefined) {
        return "--";
    }

    return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(timestamp) {
    if (!timestamp) {
        return "Waiting for data";
    }

    return timeFormatter.format(new Date(timestamp * 1000));
}

function setLogCollapsed(collapsed) {
    state.logCollapsed = Boolean(collapsed);

    if (elements.logPanel) {
        elements.logPanel.dataset.collapsed = String(state.logCollapsed);
    }

    if (elements.logCollapseBtn) {
        elements.logCollapseBtn.setAttribute("aria-expanded", String(!state.logCollapsed));
    }

    patchText(elements.logCollapseLabel, state.logCollapsed ? "Expand" : "Collapse");
}

function toggleLogCollapse() {
    setLogCollapsed(!state.logCollapsed);
}

function formatElapsed(value) {
    return `${Number(value || 0).toFixed(1)}s`;
}

function labelizeStatus(value) {
    const normalized = String(value || "idle").replaceAll("_", " ").trim();
    if (!normalized) {
        return "Idle";
    }

    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        cache: "no-store",
        headers: {
            Accept: "application/json",
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
        throw new Error(data.message || `Request failed for ${url}`);
    }

    return data;
}

function showToast(message, tone = "success") {
    return;
}

function setConnectionStatus(text, tone = "online") {
    patchText(elements.connectionStatus, text);
    setTone(elements.connectionStatus, tone);
}

function setButtonBusy(button, spinner, label, isBusy, busyText, idleText) {
    if (button) {
        button.disabled = isBusy;
    }
    if (spinner) {
        spinner.classList.toggle("hidden", !isBusy);
    }
    patchText(label, isBusy ? busyText : idleText);
}

function setStreamHudLive(isLive) {
    if (elements.streamHudChip) {
        elements.streamHudChip.dataset.live = String(Boolean(isLive));
    }
}

function syncButtonStates() {
    elements.startCameraBtn.disabled = state.cameraActive;
    elements.stopCameraBtn.disabled = !state.cameraActive;
    elements.startCollectBtn.disabled = state.collectionActive || state.trainingActive || hasGestureConflict(elements.gestureNameInput.value);
    elements.stopCollectBtn.disabled = !state.collectionActive;
    elements.trainModelBtn.disabled = state.trainingActive || state.collectionActive;
    if (elements.clearDatasetBtn) {
        elements.clearDatasetBtn.disabled = state.trainingActive || state.collectionActive;
    }
    elements.gestureNameInput.disabled = state.collectionActive || state.trainingActive;
    elements.trainingSourceSelect.disabled = state.trainingActive;
}

function normalizeLabelKey(label) {
    return String(label || "")
        .trim()
        .replace(/[^a-zA-Z0-9_-]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .toLowerCase();
}

function hasGestureConflict(label) {
    const candidate = normalizeLabelKey(label);
    if (!candidate) {
        return false;
    }

    const modelLabels = Array.isArray(state.modelMetadata?.labels) ? state.modelMetadata.labels : [];
    return modelLabels.some((existingLabel) => normalizeLabelKey(existingLabel) === candidate);
}

function applyGestureConflictState() {
    const currentLabel = elements.gestureNameInput.value.trim();
    const hasConflict = hasGestureConflict(currentLabel);

    if (elements.cameraAlert) {
        elements.cameraAlert.classList.toggle("hidden", !hasConflict);
    }

    if (elements.datasetRuleNote) {
        patchText(
            elements.datasetRuleNote,
            hasConflict && currentLabel
                ? `Gesture "${currentLabel}" is already used by the trained model and cannot be collected again as a new sample.`
                : "A gesture that is already used by the trained model cannot be collected again as a new sample."
        );
    }

    patchText(
        elements.collectionSummary,
        hasConflict
            ? "Gesture is already used by the trained model. Enter a new gesture label to collect fresh samples."
            : state.collectionActive
                ? elements.collectionSummary.textContent
                : state.trainingActive
                    ? "Collection is paused while training is running."
                    : "Prepare the camera, enter a label, and start recording landmarks."
    );

    syncButtonStates();
}

function getSelectedTrainingSourceKey() {
    return elements.trainingSourceSelect.value || "auto";
}

function getSelectedTrainingSourceSnapshot() {
    const metadata = state.modelMetadata || {};
    const sources = Array.isArray(metadata.training_sources) ? metadata.training_sources : [];
    const selectedKey = getSelectedTrainingSourceKey();
    const selectedSource = sources.find((source) => source.key === selectedKey) || null;
    const stats = state.latestStats || {};

    let title = "Auto Select";
    let sampleCount = Number(stats.samples || 0);
    let csvCount = 0;
    let labelCount = Number(stats.labels || 0);
    let usesCustomSource = false;

    if (selectedSource) {
        title =
            selectedKey === "auto" && selectedSource.resolved_title
                ? `Auto Select (${selectedSource.resolved_title})`
                : selectedSource.title;
        sampleCount = Number(selectedSource.sample_count || 0);
        csvCount = Number(selectedSource.csv_file_count || 0);
        labelCount = Number(selectedSource.label_count || 0);
        usesCustomSource =
            selectedKey === "custom" ||
            (selectedKey === "auto" && selectedSource.resolved_key === "custom");
    }

    if (state.collectionActive && usesCustomSource) {
        sampleCount += Number(stats.current_gesture_samples || 0);
    }

    return {
        title,
        sampleCount,
        csvCount,
        labelCount,
    };
}

function applySelectedTrainingSourceMetrics() {
    const snapshot = getSelectedTrainingSourceSnapshot();

    patchText(elements.trainingSourceValue, snapshot.title);
    patchText(elements.samplesValue, formatNumber(snapshot.sampleCount));
    patchText(elements.samplesMeta, "Samples in selected source");
    patchText(elements.labelsValue, formatNumber(snapshot.labelCount));
    patchText(elements.labelsMeta, "Labels in selected source");

    if (!state.trainingActive) {
        patchText(elements.trainingSamplesValue, formatNumber(snapshot.sampleCount));
        patchText(elements.trainingLabelsValue, formatNumber(snapshot.labelCount));
        patchText(elements.trainingCsvValue, formatNumber(snapshot.csvCount));
    }
}

function setCameraOverlay(stateName, title, copy) {
    elements.cameraShell.dataset.state = stateName;
    elements.cameraOverlay.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(copy)}</span>`;
}

function setCameraPredictionOverlay(label, meta, visible) {
    if (!elements.cameraPredictionOverlay) {
        return;
    }

    patchText(elements.cameraPredictionLabel, label);
    patchText(elements.cameraPredictionMeta, meta);
    elements.cameraPredictionOverlay.classList.toggle("hidden", !visible);
}

function applyPrediction(prediction = {}) {
    const hasPrediction = Boolean(prediction.label);

    if (!hasPrediction) {
        patchText(elements.predictionLabel, state.cameraActive ? "No hand detected" : "No prediction");
        patchText(elements.predictionConfidence, "--");
        patchText(elements.predictionTimestamp, state.cameraActive ? "Awaiting a stable gesture" : "Waiting for data");
        patchText(
            elements.predictionSummary,
            state.cameraActive
                ? "Keep your hand stable and centered to receive a prediction."
                : "Predictions will appear here once the live stream starts."
        );
        patchText(elements.predictionHudChip, state.cameraActive ? "Listening" : "Prediction idle");
        setCameraPredictionOverlay(
            state.cameraActive ? "Listening..." : "No prediction",
            state.cameraActive ? "Keep your gesture steady in the frame" : "Start the stream to begin predictions",
            state.cameraActive
        );
        return;
    }

    patchText(elements.predictionLabel, prediction.label);
    patchText(elements.predictionConfidence, formatConfidence(prediction.confidence));
    patchText(elements.predictionTimestamp, formatTimestamp(prediction.timestamp));
    patchText(
        elements.predictionSummary,
        `${prediction.label} is the strongest live match right now with ${formatConfidence(prediction.confidence)} confidence.`
    );
    patchText(elements.predictionHudChip, `${prediction.label} ${formatConfidence(prediction.confidence)}`);
    setCameraPredictionOverlay(
        prediction.label,
        `${formatConfidence(prediction.confidence)} confidence`,
        true
    );
}

function applyCollectionState(collection = {}) {
    state.collectionActive = Boolean(collection.active);
    const label = collection.label || "--";
    const count = formatNumber(collection.count);
    const statusText = state.collectionActive ? "Collecting" : "Idle";

    patchText(elements.collectionStatusChip, statusText);
    setTone(elements.collectionStatusChip, state.collectionActive ? "training" : "idle");
    patchText(elements.collectionStatusValue, statusText);
    patchText(elements.currentGestureValue, label);
    patchText(elements.currentGestureSamplesValue, count);
    patchText(
        elements.collectionSummary,
        state.collectionActive
            ? `Recording "${label}" live. Keep the gesture steady until you have enough samples.`
            : state.trainingActive
                ? "Collection is paused while training is running."
                : "Prepare the camera, enter a label, and start recording landmarks."
    );
    applyGestureConflictState();

    if (state.collectionActive) {
        startCollectionPolling();
    } else {
        stopCollectionPolling();
    }
}

function applyTrainingStatus(training = {}) {
    const status = training.status || "idle";
    const progress = Math.max(0, Math.min(1, Number(training.progress || 0)));
    const stageText = labelizeStatus(training.stage || "idle");
    const statusText = labelizeStatus(status);
    const refreshText =
        status === "training"
            ? "Training is running in the background while the rest of the interface stays responsive."
            : status === "completed"
                ? "The refreshed model is ready for live inference."
                : status === "failed"
                    ? "Training failed before the model could refresh."
                    : "Model ready for live inference";

    state.trainingActive = status === "training";

    patchText(elements.trainingStatusPill, statusText);
    setTone(elements.trainingStatusPill, status);
    patchText(elements.trainingMessage, training.message || "Training idle. Start a run when your samples are ready.");
    patchText(elements.trainingRefreshState, refreshText);
    elements.trainingProgressFill.style.width = `${Math.round(progress * 100)}%`;
    patchText(elements.trainingStageValue, stageText);
    patchText(elements.trainingElapsedValue, formatElapsed(training.elapsed_seconds));
    patchText(elements.trainingProgressValue, `${Math.round(progress * 100)}%`);
    patchText(elements.trainingStageSummary, statusText);

    if (training.resolved_source_title || training.source_title) {
        patchText(elements.trainingSourceValue, training.resolved_source_title || training.source_title);
    }

    patchText(elements.trainingSamplesValue, formatNumber(training.samples));
    patchText(elements.trainingLabelsValue, formatNumber(training.classes));
    patchText(elements.trainingCsvValue, formatNumber(training.csv_files));
    patchText(elements.trainingAccuracyValue, formatAccuracy(training.accuracy));

    if (!state.trainingActive) {
        setButtonBusy(elements.trainModelBtn, elements.trainSpinner, elements.trainButtonText, false, "Training...", "Train Model");
    }

    syncButtonStates();
}

function applyStats(stats = {}) {
    state.latestStats = stats;

    patchText(elements.samplesValue, formatNumber(stats.samples));
    patchText(elements.labelsValue, formatNumber(stats.labels));
    patchText(elements.samplesMeta, "Current dataset");
    patchText(elements.labelsMeta, stats.model_status === "trained" ? "Ready for inference" : "No trained model yet");

    const modelReady = stats.model_status === "trained";
    patchText(elements.modelStatusValue, modelReady ? "Ready" : "Not trained");
    patchText(
        elements.modelStatusMeta,
        modelReady
            ? `${formatNumber(stats.labels)} labels across ${formatNumber(stats.samples)} samples`
            : "Collect samples and train the model"
    );

    applyCollectionState({
        active: stats.collection_active,
        label: stats.current_gesture,
        count: stats.current_gesture_samples,
    });

    applyTrainingStatus({
        status: stats.training_status,
        message: stats.training_message,
        stage: stats.training_stage,
        progress: stats.training_progress,
        elapsed_seconds: stats.training_elapsed_seconds,
        resolved_source_title: stats.training_source_title,
        samples: stats.samples,
        classes: stats.labels,
        accuracy: stats.training_accuracy,
        csv_files: stats.training_csv_files,
    });

    applySelectedTrainingSourceMetrics();
}

function applyModelMetadata(model = {}) {
    state.modelMetadata = model;
    const modelReady = Boolean(model.model_available);
    patchText(elements.headerModelStatusChip, modelReady ? "Model ready" : "Model missing");
    setTone(elements.headerModelStatusChip, modelReady ? "online" : "error");

    if (!state.trainingActive) {
        patchText(elements.trainingSourceValue, model.dataset_title || "Auto Select");
    }

    const sources = Array.isArray(model.training_sources) ? model.training_sources : [];
    const previousValue = elements.trainingSourceSelect.value;

    if (sources.length === 0) {
        elements.trainingSourceSelect.innerHTML = '<option value="auto">Auto Select</option>';
        return;
    }

    elements.trainingSourceSelect.innerHTML = sources
        .map((source) => {
            const title =
                source.key === "auto" && source.resolved_title
                    ? `${source.title} (${source.resolved_title})`
                    : source.title;
            return `<option value="${escapeHtml(source.key)}">${escapeHtml(title)}</option>`;
        })
        .join("");

    const canKeepPrevious = sources.some((source) => source.key === previousValue);
    elements.trainingSourceSelect.value = canKeepPrevious ? previousValue : model.default_training_source || "auto";
    applySelectedTrainingSourceMetrics();
    applyGestureConflictState();
}

function applyLogs(logs = []) {
    if (!Array.isArray(logs) || logs.length === 0) {
        elements.recognitionLog.innerHTML = `
            <div class="empty-state">
                <strong>No recognition events yet</strong>
                <span>Start the camera and show a gesture to build the live feed history.</span>
            </div>
        `;
        patchText(elements.logCaption, "Waiting for events");
        return;
    }

    const rows = logs.slice(0, 10).map((log) => `
        <div class="log-row">
            <div>
                <strong>${escapeHtml(log.label || "Unknown")}</strong>
                <span>${escapeHtml(log.summary || "Recognition event")}</span>
            </div>
            <div><span class="confidence-pill">${formatConfidence(log.confidence)}</span></div>
            <div>
                <strong>${formatTimestamp(log.timestamp)}</strong>
                <span>Backend event</span>
            </div>
        </div>
    `).join("");

    elements.recognitionLog.innerHTML = rows;
    patchText(elements.logCaption, `${Math.min(logs.length, 10)} recent events`);
}

async function fetchBootstrap() {
    const payload = await requestJson("/bootstrap");
    applyStats(payload.stats || {});
    applyPrediction(payload.prediction || {});
    applyLogs(payload.logs || []);
    applyCollectionState(payload.collection || {});
    applyTrainingStatus(payload.training || {});
    applyModelMetadata(payload.model || {});
    applySelectedTrainingSourceMetrics();
    setConnectionStatus("Backend online", "online");
}

async function fetchStats() {
    if (state.fetchingStats) {
        return;
    }

    state.fetchingStats = true;
    try {
        const stats = await requestJson("/stats");
        applyStats(stats);
        setConnectionStatus("Backend online", "online");
    } catch (error) {
        setConnectionStatus("Stats unavailable", "error");
    } finally {
        state.fetchingStats = false;
    }
}

async function fetchPredictionStatus() {
    if (state.fetchingPrediction) {
        return;
    }

    state.fetchingPrediction = true;
    try {
        const payload = await requestJson("/prediction/status");
        const lead = Array.isArray(payload.predictions) ? payload.predictions[0] : null;
        applyPrediction(
            lead
                ? {
                      label: lead.label,
                      confidence: lead.confidence,
                      timestamp: Array.isArray(payload.history) && payload.history[0] ? payload.history[0].timestamp : null,
                  }
                : {}
        );
        applyLogs(payload.history || []);
        setConnectionStatus("Backend online", "online");
    } catch (error) {
        applyPrediction({});
        setConnectionStatus("Prediction unavailable", "error");
    } finally {
        state.fetchingPrediction = false;
    }
}

async function fetchCollectionStatus() {
    if (state.fetchingCollection) {
        return;
    }

    state.fetchingCollection = true;
    try {
        const collection = await requestJson("/collection/status");

        state.latestStats = {
            ...(state.latestStats || {}),
            collection_active: Boolean(collection.active),
            current_gesture: collection.label || "",
            current_gesture_samples: Number(collection.count || 0),
        };

        applyCollectionState(collection);
        applySelectedTrainingSourceMetrics();
    } finally {
        state.fetchingCollection = false;
    }
}

async function fetchTrainingStatus() {
    if (state.fetchingTraining) {
        return;
    }

    state.fetchingTraining = true;
    try {
        const training = await requestJson("/training/status");
        applyTrainingStatus(training);
        await maybeFinalizeTraining(training);
        setConnectionStatus("Backend online", "online");
    } catch (error) {
        setConnectionStatus("Training status unavailable", "error");
    } finally {
        state.fetchingTraining = false;
    }
}

async function fetchModelMetadata() {
    try {
        const model = await requestJson("/model-info");
        applyModelMetadata(model);
    } catch (error) {
        patchText(elements.headerModelStatusChip, "Model check failed");
        setTone(elements.headerModelStatusChip, "error");
    }
}

function startPredictionPolling() {
    if (state.predictionTimer) {
        return;
    }

    fetchPredictionStatus();
    state.predictionTimer = window.setInterval(fetchPredictionStatus, 1000);
}

function startCollectionPolling() {
    if (state.collectionTimer) {
        return;
    }

    fetchCollectionStatus();
    state.collectionTimer = window.setInterval(fetchCollectionStatus, 700);
}

function stopCollectionPolling() {
    if (!state.collectionTimer) {
        return;
    }

    window.clearInterval(state.collectionTimer);
    state.collectionTimer = null;
}

function stopPredictionPolling() {
    if (!state.predictionTimer) {
        return;
    }

    window.clearInterval(state.predictionTimer);
    state.predictionTimer = null;
}

function startTrainingPolling() {
    if (state.trainingTimer) {
        return;
    }

    fetchTrainingStatus();
    state.trainingTimer = window.setInterval(fetchTrainingStatus, 700);
}

function stopTrainingPolling() {
    if (!state.trainingTimer) {
        return;
    }

    window.clearInterval(state.trainingTimer);
    state.trainingTimer = null;
}

async function maybeFinalizeTraining(training) {
    if (training.status === "training") {
        state.lastTrainingTerminalStatus = "";
        startTrainingPolling();
        return;
    }

    if (training.status !== "completed" && training.status !== "failed") {
        stopTrainingPolling();
        return;
    }

    if (state.lastTrainingTerminalStatus === training.status) {
        return;
    }

    state.lastTrainingTerminalStatus = training.status;
    state.trainingActive = false;
    stopTrainingPolling();
    syncButtonStates();

    if (training.status === "completed") {
        showToast("Training completed and model refreshed.", "success");
        await Promise.allSettled([fetchStats(), fetchModelMetadata(), fetchPredictionStatus()]);
    } else {
        showToast(training.message || "Training failed.", "error");
    }
}

function setIdleCameraState() {
    state.cameraActive = false;
    setStreamHudLive(false);
    patchText(elements.streamHudChip, "Standby");
    patchText(elements.predictionHudChip, "Prediction idle");
    patchText(elements.cameraStateValue, "Offline");
    patchText(elements.cameraStatusMeta, "Camera standby");
    setCameraOverlay("idle", "Camera offline", "Start the stream to begin hand tracking and live gesture prediction.");
    setCameraPredictionOverlay("No prediction", "Start the stream to begin predictions", false);
}

function isCameraFullscreen() {
    return document.fullscreenElement === elements.cameraShell;
}

async function toggleCameraFullscreen() {
    if (!elements.cameraShell || !document.fullscreenEnabled) {
        showToast("Fullscreen is not supported in this browser.", "error");
        return;
    }

    try {
        if (isCameraFullscreen()) {
            await document.exitFullscreen();
        } else {
            await elements.cameraShell.requestFullscreen();
        }
    } catch (error) {
        showToast("Unable to change fullscreen mode.", "error");
    }
}

function syncFullscreenButton() {
    if (elements.fullscreenCameraBtn) {
        patchText(elements.fullscreenCameraBtn, isCameraFullscreen() ? "Exit Fullscreen" : "Fullscreen");
    }

    if (elements.exitFullscreenCameraBtn) {
        elements.exitFullscreenCameraBtn.classList.toggle("hidden", !isCameraFullscreen());
    }
}

function startCamera() {
    if (state.cameraActive) {
        return;
    }

    state.cameraActive = true;
    setStreamHudLive(false);
    elements.cameraFeed.src = `/video_feed?ts=${Date.now()}`;
    patchText(elements.streamHudChip, "Starting");
    patchText(elements.predictionHudChip, "Awaiting frames");
    patchText(elements.cameraStateValue, "Starting");
    patchText(elements.cameraStatusMeta, "Opening webcam stream");
    setCameraOverlay("loading", "Starting camera", "Connecting to the local webcam feed.");
    setCameraPredictionOverlay("Starting...", "Connecting to the webcam feed", true);
    syncButtonStates();
    startPredictionPolling();
}

async function stopCamera() {
    if (state.collectionActive) {
        await stopCollection({ silent: true });
    }

    stopPredictionPolling();
    stopCollectionPolling();
    elements.cameraFeed.removeAttribute("src");
    elements.cameraFeed.src = "";
    setIdleCameraState();
    applyPrediction({});
    syncButtonStates();
}

async function trainModel() {
    if (state.trainingActive) {
        return;
    }

    state.trainingActive = true;
    state.lastTrainingTerminalStatus = "";
    setButtonBusy(elements.trainModelBtn, elements.trainSpinner, elements.trainButtonText, true, "Training...", "Train Model");
    syncButtonStates();

    try {
        const response = await requestJson("/train", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                source: elements.trainingSourceSelect.value || "auto",
            }),
        });

        applyTrainingStatus(response.training || {});
        applyStats(response.stats || {});
        startTrainingPolling();
        showToast("Training started.", "success");
    } catch (error) {
        state.trainingActive = false;
        setButtonBusy(elements.trainModelBtn, elements.trainSpinner, elements.trainButtonText, false, "Training...", "Train Model");
        applyTrainingStatus({
            status: "failed",
            message: error.message,
            stage: "failed",
            progress: 1,
            elapsed_seconds: 0,
        });
        showToast(error.message, "error");
    }
}

async function clearSelectedDataset() {
    try {
        const response = await requestJson("/dataset/clear", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                source: getSelectedTrainingSourceKey(),
            }),
        });

        applyStats(response.stats || {});
        applyModelMetadata(response.model || {});
        applySelectedTrainingSourceMetrics();
        showToast(response.message || "Dataset cleared.", "success");
    } catch (error) {
        showToast(error.message, "error");
    }
}

async function startCollection() {
    const label = elements.gestureNameInput.value.trim();
    if (!label) {
        showToast("Enter a gesture label first.", "error");
        elements.gestureNameInput.focus();
        return;
    }

    if (!state.cameraActive) {
        startCamera();
    }

    try {
        const payload = await requestJson("/collection/start", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ label }),
        });

        applyCollectionState(payload);
        syncButtonStates();
        startCollectionPolling();
        await fetchStats();
        applySelectedTrainingSourceMetrics();
        showToast(payload.message || "Collection started.", "success");
    } catch (error) {
        if (String(error.message || "").toLowerCase().includes("already used by the trained model")) {
            applyGestureConflictState();
        }
        showToast(error.message, "error");
    }
}

async function stopCollection(options = {}) {
    const { silent = false } = options;

    try {
        const payload = await requestJson("/collection/stop", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({}),
        });

        applyCollectionState(payload);
        syncButtonStates();
        stopCollectionPolling();
        await fetchStats();
        await fetchModelMetadata();
        applySelectedTrainingSourceMetrics();

        if (!silent) {
            showToast(payload.message || "Collection stopped.", "success");
        }
    } catch (error) {
        if (!silent) {
            showToast(error.message, "error");
        }
    }
}

function bindEvents() {
    elements.startCameraBtn.addEventListener("click", startCamera);
    elements.fullscreenCameraBtn.addEventListener("click", toggleCameraFullscreen);
    elements.exitFullscreenCameraBtn.addEventListener("click", toggleCameraFullscreen);
    elements.stopCameraBtn.addEventListener("click", stopCamera);
    elements.trainModelBtn.addEventListener("click", trainModel);
    if (elements.clearDatasetBtn) {
        elements.clearDatasetBtn.addEventListener("click", clearSelectedDataset);
    }
    if (elements.logCollapseBtn) {
        elements.logCollapseBtn.addEventListener("click", toggleLogCollapse);
    }
    elements.startCollectBtn.addEventListener("click", startCollection);
    elements.stopCollectBtn.addEventListener("click", () => stopCollection());
    elements.trainingSourceSelect.addEventListener("change", () => {
        applySelectedTrainingSourceMetrics();
    });
    elements.gestureNameInput.addEventListener("input", applyGestureConflictState);

    elements.cameraFeed.addEventListener("load", () => {
        if (!state.cameraActive) {
            return;
        }

        setStreamHudLive(true);
        patchText(elements.streamHudChip, "Live feed");
        patchText(elements.cameraStateValue, "Live");
        patchText(elements.cameraStatusMeta, "MediaPipe and model inference active");
        elements.cameraShell.dataset.state = "live";
        syncButtonStates();
    });

    elements.cameraFeed.addEventListener("error", () => {
        if (!state.cameraActive) {
            return;
        }

        stopPredictionPolling();
        setStreamHudLive(false);
        patchText(elements.streamHudChip, "Camera error");
        patchText(elements.predictionHudChip, "Prediction paused");
        patchText(elements.cameraStateValue, "Unavailable");
        patchText(elements.cameraStatusMeta, "Check webcam permissions");
        setCameraOverlay("error", "Camera unavailable", "Check webcam permissions or close other apps using the webcam.");
        state.cameraActive = false;
        syncButtonStates();
    });

    document.addEventListener("fullscreenchange", syncFullscreenButton);
}

async function initializeApp() {
    bindEvents();
    setLogCollapsed(false);
    setIdleCameraState();
    syncFullscreenButton();
    syncButtonStates();

    try {
        await fetchBootstrap();
        if (state.trainingActive) {
            startTrainingPolling();
        }
        applyGestureConflictState();
    } catch (error) {
        setConnectionStatus("Backend unavailable", "error");
        showToast("Unable to load dashboard data from Flask.", "error");
    }

    state.dashboardTimer = window.setInterval(fetchStats, 3000);
}

window.addEventListener("DOMContentLoaded", initializeApp);
