let selectedFile = null;
let cameraStream = null;
let mediaRecorder = null;
let recordedChunks = [];

// ==================== БЕЗОПАСНЫЕ ХЕЛПЕРЫ ====================
function safeSetText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
function safeSetHTML(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}
function safeRemoveHidden(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('hidden');
}
function safeAddHidden(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
}
function safeGet(id) {
    return document.getElementById(id);
}

// Красивое имя ткани: 'Cotton_MIxed' / 'Cotton MIxed' -> 'Cotton Mixed'
function prettyName(name) {
    if (!name) return '—';
    return String(name)
        .replace(/_/g, ' ')
        .replace(/MIxed/g, 'Mixed')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// ==================== ЭЛЕМЕНТЫ ====================
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const videoPreview = document.getElementById('videoPreview');
const uploadContent = document.getElementById('uploadContent');
const resultsSection = document.getElementById('resultsSection');
const chartSection = document.getElementById('chartSection');
const mediaInfo = document.getElementById('mediaInfo');

const cameraBtn = document.getElementById('cameraBtn');
const cameraZone = document.getElementById('cameraZone');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const cancelBtn = document.getElementById('cancelBtn');
const recordBtn = document.getElementById('recordBtn');

// ==================== DRAG & DROP ====================
dropZone?.addEventListener('click', (e) => {
    // Игнорируем клик, если нажата кнопка внутри (Select File, Open Camera и т.д.)
    if (e.target.tagName !== 'BUTTON' && !e.target.closest('button')) {
        fileInput?.click();
    }
});

dropZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone?.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput?.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// ==================== КАМЕРА ====================
cameraBtn?.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
        // Запрашиваем и видео, и аудио (для записи)
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: true
        });
        cameraVideo.srcObject = cameraStream;
        safeRemoveHidden('cameraZone');
        uploadContent?.classList.add('hidden');
        preview?.classList.add('hidden');
        videoPreview?.classList.add('hidden');
    } catch (err) {
        alert('Camera/Mic access denied: ' + err.message);
    }
});

// Фото
captureBtn?.addEventListener('click', () => {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);

    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], 'fabric-sample.jpg', { type: 'image/jpeg' });
        stopCamera();
        handleFile(file);
    }, 'image/jpeg', 0.9);
});

// Запись видео
recordBtn?.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordBtn.textContent = '🔴 Start Recording';
        recordBtn.classList.remove('recording');
        return;
    }

    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '');

    mediaRecorder = new MediaRecorder(cameraStream, mimeType ? { mimeType } : {});
    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const file = new File([blob], 'fabric-video.webm', { type: 'video/webm' });
        stopCamera();
        handleFile(file); // Отправляем на бэкенд!
    };

    mediaRecorder.start();
    recordBtn.textContent = '⏹ Stop Recording';
    recordBtn.classList.add('recording');
});

cancelBtn?.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    stopCamera();
    uploadContent?.classList.remove('hidden');
    safeAddHidden('cameraZone');
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    mediaRecorder = null;
    safeAddHidden('cameraZone');
}

// ==================== ОБРАБОТКА ФАЙЛА ====================
function handleFile(file) {
    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isImage) {
        alert('Please upload an image or video file');
        return;
    }

    selectedFile = file;
    uploadContent?.classList.add('hidden');

    if (isVideo) {
        preview?.classList.add('hidden');
        safeRemoveHidden('videoPreview');
        videoPreview.src = URL.createObjectURL(file);
    } else {
        videoPreview?.classList.add('hidden');
        safeRemoveHidden('preview');
        const reader = new FileReader();
        reader.onload = (e) => { preview.src = e.target.result; };
        reader.readAsDataURL(file);
    }

    analyzeMedia();
}

// ==================== АНАЛИЗ ====================
async function analyzeMedia() {
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                const err = await response.json();
                if (err.detail) detail = err.detail;
            } catch (_) { /* not JSON */ }

            if (response.status === 503) {
                alert('Модель ещё не обучена. Запустите: python train.py\n\n' + detail);
            } else if (response.status === 413) {
                alert('Файл слишком большой.\n' + detail);
            } else if (response.status === 415) {
                alert('Неподдерживаемый формат файла.\n' + detail);
            } else {
                alert('Ошибка анализа: ' + detail);
            }
            return;
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        alert('Analysis failed: ' + error.message);
    }
}

// ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====================
function displayResults(data) {
    safeRemoveHidden('resultsSection');
    safeRemoveHidden('chartSection');

    // --- Media info ---
    safeRemoveHidden('mediaInfo');
    if (data.is_video) {
        safeSetText('mediaBadge', '🎥 Video');
        safeSetText('framesCount', `${data.frames_used ?? data.frames_analyzed ?? 0} frames analyzed`);
    } else {
        safeSetText('mediaBadge', '📷 Photo');
        safeSetText('framesCount', '1 frame analyzed');
    }

    const mode = data.recommended_mode || {};
    const fabrics = data.detected_fabrics || [];
    const topFabric = fabrics[0] || { confidence: 0, fabric: '—' };

    // --- Confidence badge ---
    const confValue = (typeof data.confidence === 'number') ? data.confidence : topFabric.confidence;
    safeSetText('confidenceBadge', `${confValue.toFixed(1)}% confidence`);

    // --- Mode ---
    safeSetText('modeIcon', mode.icon || '❓');
    safeSetText('modeName', mode.name || prettyName(topFabric.fabric));
    safeSetText('modeTemp', mode.temp != null ? `${mode.temp}°C` : '--');
    safeSetText('modeSpin', mode.spin != null ? `${mode.spin} RPM` : '--');
    safeSetText('modeTime', mode.time != null ? `${mode.time} min` : '--');

    // --- Basis ---
    if (mode.basis) {
        safeSetText('modeBasis', mode.basis);
        safeRemoveHidden('modeBasis');
    } else {
        safeAddHidden('modeBasis');
    }
    if (mode.color_note && !mode.basis) {
        safeSetText('modeBasis', mode.color_note);
        safeRemoveHidden('modeBasis');
    }

    // --- Low confidence warning ---
    if (data.low_confidence) {
        safeRemoveHidden('lowConfWarning');
    } else {
        safeAddHidden('lowConfWarning');
    }

    // --- Adjustments ---
    const adjustments = data.adjustments || [];
    if (adjustments.length > 0) {
        safeSetHTML('adjustmentsList', adjustments.map(a => `
            <div class="adjustment-item">
                <div class="adjustment-line">
                    <b>${a.param}</b>:
                    <span class="from">${a.from}</span> → <span class="to">${a.to}</span>
                </div>
                <div class="adjustment-reason">${reasonText(a.reason)}</div>
            </div>
        `).join(''));
        safeRemoveHidden('adjustmentsBlock');
    } else {
        safeAddHidden('adjustmentsBlock');
    }

    // --- Mixed load ---
    const mixedFabrics = data.mixed_fabrics || [];
    if (data.mixed_load_warning && mixedFabrics.length > 0) {
        safeSetHTML('mixedFabricsList', mixedFabrics.map(m => `
            <div class="mixed-fabric-item">
                <span class="mixed-name">${prettyName(m.fabric)}</span>
                <span class="mixed-share">${(m.share * 100).toFixed(0)}% of frame</span>
            </div>
        `).join(''));
        safeRemoveHidden('mixedLoadBlock');
    } else {
        safeAddHidden('mixedLoadBlock');
    }

    // --- Color analysis ---
    const colorReport = data.color_check || data.color_analysis || null;
    if (colorReport) {
        safeRemoveHidden('colorAnalysis');
        displayColorAnalysis(colorReport);
    } else {
        safeAddHidden('colorAnalysis');
    }

    // --- Confidence chart ---
    safeSetHTML('confidenceChart', fabrics.map(f => `
        <div class="confidence-bar">
            <div class="confidence-label">${prettyName(f.fabric)}</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: ${(f.confidence ?? 0).toFixed(1)}%"></div>
            </div>
            <div class="confidence-value">${(f.confidence ?? 0).toFixed(1)}%</div>
        </div>
    `).join(''));
}

function reasonText(reason) {
    const map = {
        'mixed_fabrics': 'Mixed load — chosen the most delicate fabric.',
        'low_confidence': 'Low confidence — chosen the safest plausible option.',
        'color_mix': 'Color safety — mode softened to prevent dye bleeding.',
    };
    return map[reason] || reason;
}

// ==================== COLOR ANALYSIS ====================
function displayColorAnalysis(report) {
    const severityDot = safeGet('severityDot');
    const severityText = safeGet('severityText');
    const colorScore = safeGet('colorScore');
    const colorMessage = safeGet('colorMessage');
    const paletteGrid = safeGet('paletteGrid');
    const conflictsSection = safeGet('conflictsSection');
    const conflictsList = safeGet('conflictsList');

    if (!severityDot || !paletteGrid) {
        console.warn('Color Analysis DOM elements missing');
        return;
    }

    severityDot.className = 'severity-dot ' + report.severity;
    severityText.textContent = report.severity === 'ok' ? 'Safe Load' :
                               report.severity === 'medium' ? 'Caution' : 'High Risk';
    colorScore.textContent = `Risk Score: ${((report.score || 0) * 100).toFixed(0)}%`;
    colorMessage.textContent = report.message || '';

    const palette = Array.isArray(report.palette) ? report.palette : [];
    paletteGrid.innerHTML = palette.map(g => `
        <div class="palette-item">
            <div class="palette-swatch" style="background-color: ${g.hex}"></div>
            <div class="palette-info">
                <span class="palette-name">${g.name}</span>
                <span class="palette-category">${g.category}</span>
                <span class="palette-share">${((g.share || 0) * 100).toFixed(0)}% of load</span>
            </div>
        </div>
    `).join('');

    const conflicts = Array.isArray(report.conflicts) ? report.conflicts : [];
    if (conflicts.length > 0) {
        safeRemoveHidden('conflictsSection');
        conflictsList.innerHTML = conflicts.map(c => `
            <div class="conflict-item ${c.level}">
                <div class="conflict-pair">
                    ${c.a} ↔ ${c.b}
                    <span class="conflict-level ${c.level}">${c.level}</span>
                </div>
                <div class="conflict-reason">${c.reason}</div>
            </div>
        `).join('');
    } else {
        safeAddHidden('conflictsSection');
    }
}