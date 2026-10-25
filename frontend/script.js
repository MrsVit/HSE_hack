let selectedFile = null;
let cameraStream = null;
let samplesAnalyzed = 0;

// ==================== БЕЗОПАСНЫЕ ХЕЛПЕРЫ ====================
// Если элемента нет в HTML — просто ничего не делаем, без ошибки
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

// ==================== ЭЛЕМЕНТЫ ====================
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const videoPreview = document.getElementById('videoPreview');
const uploadContent = document.getElementById('uploadContent');
const resultsSection = document.getElementById('resultsSection');
const chartSection = document.getElementById('chartSection');
const mediaInfo = document.getElementById('mediaInfo');
const colorAnalysis = document.getElementById('colorAnalysis');

const cameraBtn = document.getElementById('cameraBtn');
const cameraZone = document.getElementById('cameraZone');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const cancelBtn = document.getElementById('cancelBtn');

// ==================== DRAG & DROP ====================
dropZone?.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') fileInput?.click();
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
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        cameraVideo.srcObject = cameraStream;
        safeRemoveHidden('cameraZone');
        uploadContent?.classList.add('hidden');
        preview?.classList.add('hidden');
        videoPreview?.classList.add('hidden');
    } catch (err) {
        alert('Camera access denied: ' + err.message);
    }
});

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

cancelBtn?.addEventListener('click', () => {
    stopCamera();
    uploadContent?.classList.remove('hidden');
    safeAddHidden('cameraZone');
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
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
        reader.onload = (e) => {
            preview.src = e.target.result;
        };
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
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
        samplesAnalyzed++;
        safeSetText('kpiSamples', samplesAnalyzed);
    } catch (error) {
        alert('Analysis failed: ' + error.message);
    }
}

// ==================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ====================
function displayResults(data) {
    safeRemoveHidden('resultsSection');
    safeRemoveHidden('chartSection');
    
    // Media info
    safeRemoveHidden('mediaInfo');
    if (data.is_video) {
        safeSetText('mediaBadge', '🎥 Video');
        safeSetText('framesCount', `${data.frames_analyzed} frames analyzed`);
    } else {
        safeSetText('mediaBadge', '📷 Photo');
        safeSetText('framesCount', '1 frame analyzed');
    }
    
    const mode = data.recommended_mode;
    const topFabric = data.detected_fabrics[0];
    
    // Primary Recommendation
    safeSetText('confidenceBadge', `${topFabric.confidence}% confidence`);
    safeSetText('modeIcon', mode.icon);
    safeSetText('modeName', mode.name);
    safeSetText('modeTemp', `${mode.temp}°C`);
    safeSetText('modeSpin', `${mode.spin} RPM`);
    safeSetText('modeTime', `${mode.time} min`);
    
    // Fabric Detection
    safeSetHTML('fabricsList', data.detected_fabrics.map(f => `
        <div class="fabric-item">
            <span class="fabric-name">${f.fabric.replace('_', ' ')}</span>
            <span class="fabric-confidence">${f.confidence}%</span>
        </div>
    `).join(''));
    
    // Color Analysis
    if (data.color_analysis) {
        safeRemoveHidden('colorAnalysis');
        displayColorAnalysis(data.color_analysis);
    } else {
        safeAddHidden('colorAnalysis');
    }
    
    // Recommendations
    const allTips = [...data.tips];
    if (data.color_analysis && data.color_analysis.severity !== 'ok') {
        allTips.unshift(`🎨 Color Safety: ${data.color_analysis.message}`);
    }
    safeSetHTML('tipsList', allTips.map(t => `<div class="tip-item">${t}</div>`).join(''));
    
    // Confidence Chart
    safeSetHTML('confidenceChart', data.detected_fabrics.map(f => `
        <div class="confidence-bar">
            <div class="confidence-label">${f.fabric.replace('_', ' ')}</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: ${f.confidence}%"></div>
            </div>
            <div class="confidence-value">${f.confidence}%</div>
        </div>
    `).join(''));
    
    // Energy Savings KPI (если блок есть)
    const energySavings = mode.temp < 40 ? '35%' : mode.temp < 60 ? '20%' : '10%';
    safeSetText('kpiEnergy', energySavings);
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
    colorScore.textContent = `Risk Score: ${(report.score * 100).toFixed(0)}%`;
    colorMessage.textContent = report.message;
    
    paletteGrid.innerHTML = report.palette.map(g => `
        <div class="palette-item">
            <div class="palette-swatch" style="background-color: ${g.hex}"></div>
            <div class="palette-info">
                <span class="palette-name">${g.name}</span>
                <span class="palette-category">${g.category}</span>
                <span class="palette-share">${(g.share * 100).toFixed(0)}% of load</span>
            </div>
        </div>
    `).join('');
    
    if (report.conflicts && report.conflicts.length > 0) {
        safeRemoveHidden('conflictsSection');
        conflictsList.innerHTML = report.conflicts.map(c => `
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