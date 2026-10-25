let selectedFile = null;
let cameraStream = null;
let samplesAnalyzed = 0;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const resultsSection = document.getElementById('resultsSection');
const chartSection = document.getElementById('chartSection');

// Camera elements
const cameraBtn = document.getElementById('cameraBtn');
const cameraZone = document.getElementById('cameraZone');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const cancelBtn = document.getElementById('cancelBtn');

// Drag & Drop
dropZone.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// Camera
cameraBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        cameraVideo.srcObject = cameraStream;
        cameraZone.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        preview.classList.add('hidden');
    } catch (err) {
        alert('Camera access denied: ' + err.message);
    }
});

captureBtn.addEventListener('click', () => {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);
    
    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], 'fabric-sample.jpg', { type: 'image/jpeg' });
        stopCamera();
        handleFile(file);
    }, 'image/jpeg', 0.9);
});

cancelBtn.addEventListener('click', () => {
    stopCamera();
    uploadContent.classList.remove('hidden');
    cameraZone.classList.add('hidden');
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraZone.classList.add('hidden');
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        uploadContent.classList.add('hidden');
        
        analyzeFabric();
    };
    reader.readAsDataURL(file);
}

async function analyzeFabric() {
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayResults(data);
        
        samplesAnalyzed++;
        document.getElementById('kpiSamples').textContent = samplesAnalyzed;
    } catch (error) {
        alert('Analysis failed: ' + error.message);
    }
}

function displayResults(data) {
    resultsSection.classList.remove('hidden');
    chartSection.classList.remove('hidden');
    
    const mode = data.recommended_mode;
    const topFabric = data.detected_fabrics[0];
    
    // Primary Recommendation
    document.getElementById('confidenceBadge').textContent = `${topFabric.confidence}% confidence`;
    document.getElementById('modeIcon').textContent = mode.icon;
    document.getElementById('modeName').textContent = mode.name;
    document.getElementById('modeTemp').textContent = `${mode.temp}°C`;
    document.getElementById('modeSpin').textContent = `${mode.spin} RPM`;
    document.getElementById('modeTime').textContent = `${mode.time} min`;
    
    // Fabric Detection
    const fabricsList = document.getElementById('fabricsList');
    fabricsList.innerHTML = data.detected_fabrics.map(f => `
        <div class="fabric-item">
            <span class="fabric-name">${f.fabric.replace('_', ' ')}</span>
            <span class="fabric-confidence">${f.confidence}%</span>
        </div>
    `).join('');
    
    // Recommendations
    const tipsList = document.getElementById('tipsList');
    tipsList.innerHTML = data.tips.map(t => `
        <div class="tip-item">${t}</div>
    `).join('');
    
    // Confidence Chart
    const confidenceChart = document.getElementById('confidenceChart');
    confidenceChart.innerHTML = data.detected_fabrics.map(f => `
        <div class="confidence-bar">
            <div class="confidence-label">${f.fabric.replace('_', ' ')}</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: ${f.confidence}%"></div>
            </div>
            <div class="confidence-value">${f.confidence}%</div>
        </div>
    `).join('');
    
    // Energy Savings KPI
    const energySavings = mode.temp < 40 ? '35%' : mode.temp < 60 ? '20%' : '10%';
    document.getElementById('kpiEnergy').textContent = energySavings;
}