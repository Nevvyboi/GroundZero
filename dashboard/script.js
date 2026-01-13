// GroundZero AI Dashboard - Fixed Neural Stats

const API_BASE = '/api';
const POLL_INTERVAL = 2000;

let lossChart = null;
let lastUpdateTime = null;

// Stats object
let stats = {
    facts: 0, entities: 0, causal: 0, confidence: 70,
    neural: {
        epochs: 0,
        loss: null,
        embedDim: 100,
        predictions: 0,
        hypotheses: 0,
        isTrained: false,
        entities: 0,
        relations: 0,
        triples: 0
    },
    lossHistory: []
};

// =============================================================================
// THEME
// =============================================================================

function toggleMode() {
    const isLight = document.body.classList.toggle('light-mode');
    document.getElementById('modeToggle').textContent = isLight ? '☀️' : '🌙';
    localStorage.setItem('gz-mode', isLight ? 'light' : 'dark');
    updateChartColors();
}

function setTheme(theme) {
    document.body.classList.remove('theme-blue', 'theme-green', 'theme-purple', 'theme-pink', 'theme-orange', 'theme-red');
    document.body.classList.add('theme-' + theme);
    document.getElementById('themeDropdown').value = theme;
    localStorage.setItem('gz-theme', theme);
    updateChartColors();
}

function loadSavedTheme() {
    const mode = localStorage.getItem('gz-mode') || 'dark';
    if (mode === 'light') {
        document.body.classList.add('light-mode');
        document.getElementById('modeToggle').textContent = '☀️';
    }
    setTheme(localStorage.getItem('gz-theme') || 'blue');
}

function updateChartColors() {
    if (!lossChart) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    lossChart.data.datasets[0].borderColor = accent;
    lossChart.update('none');
}

// =============================================================================
// MODALS
// =============================================================================

function openModal(id) {
    document.getElementById('modal-' + id).classList.add('active');
    if (id === 'neural-graph') setTimeout(drawNeuralGraph, 100);
    if (id === 'ai-caps') updateCapabilityModal();
}

function closeModal(id) {
    document.getElementById('modal-' + id).classList.remove('active');
}

document.addEventListener('click', e => {
    if (e.target.classList.contains('modal')) e.target.classList.remove('active');
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
});

// =============================================================================
// MILESTONES
// =============================================================================

const MILESTONES = [
    { level: 1, name: "Pattern Recognition", facts: 100 },
    { level: 2, name: "Knowledge Building", facts: 1000 },
    { level: 3, name: "Causal Understanding", facts: 5000 },
    { level: 4, name: "Reasoning Chains", facts: 20000 },
    { level: 5, name: "Deep Understanding", facts: 100000 },
    { level: 6, name: "Human-Like", facts: 500000 }
];

function getCurrentLevel() {
    for (let i = MILESTONES.length - 1; i >= 0; i--) {
        if (stats.facts >= MILESTONES[i].facts) return i + 1;
    }
    return 0;
}

function getProgress() {
    const lvl = getCurrentLevel();
    if (lvl >= MILESTONES.length) return 100;
    return Math.round((stats.facts / MILESTONES[lvl].facts) * 100);
}

// =============================================================================
// UI UPDATE
// =============================================================================

function updateUI() {
    const lvl = getCurrentLevel();
    const prog = getProgress();
    
    // Stats row
    setText('level', lvl);
    setText('facts', formatNum(stats.facts));
    setText('entities', formatNum(stats.entities));
    setText('causal', formatNum(stats.causal));
    setText('epochs', stats.neural.epochs);
    setText('loss', stats.neural.loss !== null ? stats.neural.loss.toFixed(4) : '-');
    setText('confidence', Math.round(stats.confidence) + '%');
    
    // Progress panel
    setText('level-badge', 'LEVEL ' + lvl);
    setText('level-name', lvl > 0 ? MILESTONES[lvl-1].name : 'Starting Out');
    setStyle('progress-bar', 'width', prog + '%');
    setText('progress-pct', prog + '%');
    
    // Neural panel - ALL stats
    setText('entityCount', formatNum(stats.neural.entities));
    setText('relationCount', stats.neural.relations);
    setText('tripleCount', formatNum(stats.neural.triples));
    setText('trainEpochs', stats.neural.epochs);
    setText('lossVal', stats.neural.loss !== null ? stats.neural.loss.toFixed(4) : '-');
    setText('predictions', stats.neural.predictions);
    setText('hypotheses', stats.neural.hypotheses);
    setText('embedDim', stats.neural.embedDim);
    
    // Progress bars - scaled appropriately
    setBar('entityBar', stats.neural.entities, 10000);
    setBar('relationBar', stats.neural.relations, 50);
    setBar('tripleBar', stats.neural.triples, 10000);
    setBar('trainBar', stats.neural.epochs, 500);
    setBar('lossBar', stats.neural.loss !== null ? (1 - stats.neural.loss) * 100 : 0, 100);
    
    // Badge - shows state: No Data → Loaded → Trained
    const badge = document.getElementById('neuralBadge');
    if (badge) {
        if (stats.neural.isTrained || stats.neural.epochs > 0) {
            badge.textContent = 'Trained ✓';
            badge.className = 'badge trained';
        } else if (stats.neural.triples > 0) {
            badge.textContent = 'Data Loaded';
            badge.className = 'badge loaded';
        } else {
            badge.textContent = 'No Data';
            badge.className = 'badge';
        }
    }
    
    // Graph stats in modal
    setText('gsEntities', formatNum(stats.neural.entities));
    setText('gsRelations', stats.neural.relations);
    setText('gsTriples', formatNum(stats.neural.triples));
    setText('gsEmbedDim', stats.neural.embedDim);
    
    updateCapabilities();
    updateTimeline(lvl);
    updateLossChart();
}

function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setStyle(id, prop, val) {
    const el = document.getElementById(id);
    if (el) el.style[prop] = val;
}

function setBar(id, val, max) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min(100, (val / max) * 100) + '%';
}

function updateCapabilities() {
    const caps = [
        ['capUnderstand', stats.neural.entities > 0],
        ['capGeneralize', stats.neural.isTrained || stats.neural.epochs > 0],
        ['capLearn', stats.neural.triples > 0],
        ['capReason', stats.neural.predictions > 0],
        ['capCreate', stats.neural.hypotheses > 0]
    ];
    
    caps.forEach(([id, active]) => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('active', active);
    });
}

function updateCapabilityModal() {
    const caps = [
        ['capStatusUnderstand', stats.neural.entities > 0],
        ['capStatusGeneralize', stats.neural.isTrained || stats.neural.epochs > 0],
        ['capStatusLearn', stats.neural.triples > 0],
        ['capStatusReason', stats.neural.predictions > 0],
        ['capStatusCreate', stats.neural.hypotheses > 0]
    ];
    
    caps.forEach(([id, active]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = active ? '●' : '○';
            el.classList.toggle('active', active);
        }
    });
}

function updateTimeline(lvl) {
    const tl = document.getElementById('timeline');
    if (!tl) return;
    tl.innerHTML = MILESTONES.map((m, i) => {
        const status = (i + 1) < lvl ? 'done' : (i + 1) === lvl ? 'current' : '';
        return `<div class="tl-item ${status}">
            <div class="tl-dot"></div>
            <div class="tl-info">
                <div class="tl-name">L${m.level}: ${m.name}</div>
                <div class="tl-req">${formatNum(m.facts)} facts</div>
            </div>
        </div>`;
    }).join('');
}

// =============================================================================
// LAST UPDATED
// =============================================================================

function updateLastUpdated() {
    lastUpdateTime = new Date();
    setText('lastUpdated', formatTimeAgo(lastUpdateTime));
}

function formatTimeAgo(date) {
    const s = Math.floor((new Date() - date) / 1000);
    if (s < 5) return 'just now';
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    return Math.floor(s / 3600) + 'h ago';
}

setInterval(() => {
    if (lastUpdateTime) setText('lastUpdated', formatTimeAgo(lastUpdateTime));
}, 1000);

// =============================================================================
// LOSS CHART
// =============================================================================

function initLossChart() {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;
    
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    
    lossChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: accent,
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { 
                    display: true,
                    grid: { color: 'rgba(128,128,128,0.1)' },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });
}

function updateLossChart() {
    if (!lossChart || !stats.lossHistory.length) return;
    lossChart.data.labels = stats.lossHistory.map((_, i) => i);
    lossChart.data.datasets[0].data = stats.lossHistory;
    lossChart.update('none');
}

// =============================================================================
// NEURAL GRAPH - Improved visualization
// =============================================================================

function drawNeuralGraph() {
    const canvas = document.getElementById('neuralGraphCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    const style = getComputedStyle(document.documentElement);
    const accent = style.getPropertyValue('--accent').trim() || '#1d9bf0';
    const accent2 = style.getPropertyValue('--accent2').trim() || '#4db5f9';
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Calculate node count based on actual data
    const entityCount = Math.min(stats.neural.entities || 0, 60);
    const relationCount = stats.neural.relations || 0;
    
    if (entityCount === 0) {
        // No data - show empty state
        ctx.fillStyle = style.getPropertyValue('--text2').trim() || '#8899a6';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data loaded yet', cx, cy);
        return;
    }
    
    // Create nodes in clusters (one cluster per relation type)
    const nodes = [];
    const numClusters = Math.max(relationCount, 1);
    const nodesPerCluster = Math.ceil(entityCount / numClusters);
    const mainRadius = Math.min(canvas.width, canvas.height) * 0.38;
    
    for (let c = 0; c < numClusters; c++) {
        const clusterAngle = (c / numClusters) * Math.PI * 2;
        const clusterCx = cx + Math.cos(clusterAngle) * mainRadius * 0.5;
        const clusterCy = cy + Math.sin(clusterAngle) * mainRadius * 0.5;
        
        for (let i = 0; i < nodesPerCluster && nodes.length < entityCount; i++) {
            const angle = (i / nodesPerCluster) * Math.PI * 2;
            const r = 30 + Math.random() * 40;
            nodes.push({
                x: clusterCx + Math.cos(angle) * r,
                y: clusterCy + Math.sin(angle) * r,
                cluster: c
            });
        }
    }
    
    // Draw connections (more connections = more triples)
    const triplesRatio = Math.min(stats.neural.triples / entityCount, 3);
    ctx.strokeStyle = 'rgba(128,128,128,0.12)';
    ctx.lineWidth = 1;
    
    nodes.forEach((n, i) => {
        const connCount = Math.floor(1 + triplesRatio);
        for (let j = 0; j < connCount; j++) {
            // Prefer connections within same cluster
            let targetIdx;
            if (Math.random() > 0.3) {
                // Same cluster
                const sameCluster = nodes.filter((nn, idx) => nn.cluster === n.cluster && idx !== i);
                if (sameCluster.length > 0) {
                    const target = sameCluster[Math.floor(Math.random() * sameCluster.length)];
                    targetIdx = nodes.indexOf(target);
                }
            }
            if (!targetIdx) {
                targetIdx = Math.floor(Math.random() * nodes.length);
            }
            
            if (targetIdx !== i) {
                ctx.beginPath();
                ctx.moveTo(n.x, n.y);
                ctx.lineTo(nodes[targetIdx].x, nodes[targetIdx].y);
                ctx.stroke();
            }
        }
    });
    
    // Draw nodes with cluster colors
    const clusterColors = [accent, accent2, '#17bf63', '#f97316', '#7856ff', '#e0245e'];
    
    nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = clusterColors[n.cluster % clusterColors.length];
        ctx.fill();
    });
    
    // Draw center node
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();
    
    // Label
    ctx.fillStyle = style.getPropertyValue('--text2').trim() || '#8899a6';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${formatNum(stats.neural.entities)} entities · ${stats.neural.relations} relations`, cx, canvas.height - 10);
}

function refreshGraph() { drawNeuralGraph(); }

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function trainNeural() {
    const btn = event.target;
    btn.textContent = '⏳ Training...';
    btn.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/neural/train`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({ epochs: 50 }) 
        });
        
        if (res.ok) {
            btn.textContent = '✓ Started!';
            setTimeout(() => { btn.textContent = '🧠 Train Now'; btn.disabled = false; }, 2000);
        } else {
            throw new Error('Training failed');
        }
    } catch (e) {
        btn.textContent = '🧠 Train Now';
        btn.disabled = false;
        alert('Start server: python main.py dashboard');
    }
}

async function generateHypotheses() {
    const list = document.getElementById('hypothesesList');
    list.innerHTML = '<div class="hypothesis">Loading...</div>';
    
    try {
        const res = await fetch(`${API_BASE}/neural/hypotheses`);
        if (res.ok) {
            const data = await res.json();
            if (data.hypotheses && data.hypotheses.length) {
                list.innerHTML = data.hypotheses.slice(0, 10).map(h => 
                    `<div class="hypothesis">
                        <span>${h.Head} → ${h.Relation} → ${h.Tail}</span>
                        <span class="conf">${Math.round(h.Confidence * 100)}%</span>
                    </div>`
                ).join('');
            } else {
                list.innerHTML = '<div class="hypothesis">No hypotheses yet - need more data</div>';
            }
        }
    } catch (e) { 
        list.innerHTML = '<div class="hypothesis">Server offline</div>'; 
    }
}

// =============================================================================
// CHAT
// =============================================================================

function initChat() {
    const input = document.getElementById('chatInput');
    const btn = document.getElementById('sendBtn');
    if (!input || !btn) return;
    btn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    
    input.value = '';
    addMessage(text, 'user');
    
    try {
        const res = await fetch(`${API_BASE}/chat`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({ message: text }) 
        });
        
        if (res.ok) {
            const data = await res.json();
            addMessage(data.answer, 'bot');
            if (data.confidence) stats.confidence = data.confidence * 100;
        } else {
            throw new Error();
        }
    } catch (e) { 
        addMessage("Server offline. Start: python main.py dashboard", 'bot'); 
    }
}

function addMessage(text, sender) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.textContent = sender === 'bot' ? '🧠 ' + text : text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// =============================================================================
// API POLLING - Fixed field mapping
// =============================================================================

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) throw new Error('API error');
        
        const data = await res.json();
        
        // Debug: log what we receive
        console.log('API Response:', data);
        
        // Map Knowledge stats
        if (data.Knowledge) {
            stats.facts = data.Knowledge.TotalFacts || 0;
        }
        
        // Map Causal stats
        if (data.Causal) {
            stats.causal = data.Causal.TotalRelations || 0;
        }
        
        // Map Chat stats
        if (data.Chat) {
            stats.confidence = (data.Chat.AverageConfidence || 0.7) * 100;
        }
        
        // Map Neural stats - be careful with field names
        if (data.Neural) {
            stats.neural.entities = data.Neural.TotalEntities || 0;
            stats.neural.relations = data.Neural.TotalRelations || 0;
            stats.neural.triples = data.Neural.TotalTriples || 0;
            stats.neural.embedDim = data.Neural.EmbeddingDim || 100;
            stats.neural.epochs = data.Neural.TrainingEpochs || 0;
            stats.neural.loss = data.Neural.LastLoss;
            stats.neural.isTrained = data.Neural.IsTrained || false;
            stats.neural.predictions = data.Neural.Predictions || 0;
            stats.neural.hypotheses = data.Neural.Hypotheses || 0;
            
            // Also update top-level entities
            stats.entities = stats.neural.entities;
        }
        
        // Loss history
        if (data.LossHistory && Array.isArray(data.LossHistory)) {
            stats.lossHistory = data.LossHistory;
        }
        
        setStatus('connected', 'Live');
        updateLastUpdated();
        
    } catch (e) {
        console.error('Fetch error:', e);
        setStatus('error', 'Offline');
    }
    
    updateUI();
}

function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (dot) dot.className = 'live-dot ' + state;
    if (txt) txt.textContent = text;
}

// =============================================================================
// INIT
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadSavedTheme();
    initLossChart();
    initChat();
    updateUI();
    updateLastUpdated();
    
    // Initial fetch
    fetchStats();
    
    // Poll every 2 seconds
    setInterval(fetchStats, POLL_INTERVAL);
    
    console.log('🧠 GroundZero AI Dashboard loaded');
});