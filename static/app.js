/**
 * GroundZero AI - User Interface v2.3
 * Fixed: Theme toggle, Voice recording with visual feedback
 */

const API = {
    chat: '/api/chat',
    stats: '/api/stats',
    search: '/api/search',
    timeline: '/api/timeline',
    learningStart: '/api/learning/start',
    learningStop: '/api/learning/stop',
    voiceTranscribe: '/api/voice/transcribe'
};

// ============================================================
// Theme Manager - Toggle in Settings
// ============================================================
const ThemeManager = {
    init() {
        const saved = localStorage.getItem('theme') || 'dark';
        this.applyTheme(saved);
        
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            // Set initial state - checked = dark mode
            toggle.checked = (saved === 'dark');
            
            // Listen for changes
            toggle.addEventListener('change', (e) => {
                const newTheme = e.target.checked ? 'dark' : 'light';
                this.applyTheme(newTheme);
                console.log('Theme changed to:', newTheme);
            });
        }
    },
    
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        console.log('Applied theme:', theme);
    }
};

// ============================================================
// Voice Recorder with Visual Feedback
// ============================================================
class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        
        // UI elements
        this.voiceBtn = null;
        this.statusBar = null;
        this.timerEl = null;
        this.messageEl = null;
    }
    
    init() {
        this.voiceBtn = document.getElementById('voice-btn');
        this.statusBar = document.getElementById('voice-status-bar');
        this.timerEl = document.getElementById('voice-timer');
        this.messageEl = document.getElementById('voice-message');
        
        if (!this.voiceBtn) return;
        
        // Hold to record
        this.voiceBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isRecording) this.stopRecording();
        });
        
        // Touch support
        this.voiceBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        
        document.addEventListener('touchend', () => {
            if (this.isRecording) this.stopRecording();
        });
        
        // Cancel button
        document.getElementById('voice-cancel')?.addEventListener('click', () => {
            this.cancelRecording();
        });
    }
    
    async startRecording() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { echoCancellation: true, noiseSuppression: true } 
            });
            
            this.audioChunks = [];
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
            
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.audioChunks.push(e.data);
            };
            
            this.mediaRecorder.start(100);
            this.isRecording = true;
            this.startTime = Date.now();
            
            // Update UI
            this.voiceBtn.classList.add('recording');
            this.statusBar?.classList.remove('hidden');
            this.updateMessage('🎤 Recording... Release to send');
            this.startTimer();
            
        } catch (err) {
            console.error('Mic error:', err);
            showToast('Microphone access denied', 'error');
        }
    }
    
    async stopRecording() {
        if (!this.isRecording) return;
        
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = async () => {
                const blob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
                this.cleanup();
                
                if (blob.size > 1000) {
                    this.updateMessage('⏳ Transcribing...');
                    this.statusBar?.classList.remove('hidden');
                    
                    const text = await this.transcribe(blob);
                    
                    this.statusBar?.classList.add('hidden');
                    
                    if (text && text.trim()) {
                        // Put text in input and send
                        const input = document.getElementById('chat-input');
                        if (input) {
                            input.value = text;
                            window.chatInterface?.sendMessage();
                        }
                    } else {
                        showToast('Could not transcribe audio', 'error');
                    }
                } else {
                    this.statusBar?.classList.add('hidden');
                }
                
                resolve();
            };
            
            this.mediaRecorder.stop();
        });
    }
    
    cancelRecording() {
        this.cleanup();
        this.statusBar?.classList.add('hidden');
        showToast('Recording cancelled', 'info');
    }
    
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
        }
        this.isRecording = false;
        this.voiceBtn?.classList.remove('recording');
        this.stopTimer();
    }
    
    startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const mins = Math.floor(elapsed / 60);
            const secs = elapsed % 60;
            if (this.timerEl) {
                this.timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }
        }, 100);
    }
    
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }
    
    updateMessage(msg) {
        if (this.messageEl) this.messageEl.textContent = msg;
    }
    
    async transcribe(blob) {
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');
        
        try {
            const res = await fetch(API.voiceTranscribe, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (data.error) {
                console.error('Transcribe error:', data.error);
                return '';
            }
            
            return data.text || '';
        } catch (err) {
            console.error('Transcribe failed:', err);
            return '';
        }
    }
}

// ============================================================
// Chat Interface
// ============================================================
class ChatInterface {
    constructor() {
        this.messages = document.getElementById('chat-messages');
        this.input = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        
        this.setupEvents();
    }
    
    setupEvents() {
        this.sendBtn?.addEventListener('click', () => this.sendMessage());
        
        this.input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize
        this.input?.addEventListener('input', () => {
            this.input.style.height = 'auto';
            this.input.style.height = Math.min(this.input.scrollHeight, 120) + 'px';
        });
        
        document.getElementById('btn-clear-chat')?.addEventListener('click', () => {
            if (confirm('Clear chat?')) this.clearChat();
        });
    }
    
    async sendMessage() {
        const text = this.input?.value?.trim();
        if (!text) return;
        
        this.input.value = '';
        this.input.style.height = 'auto';
        
        this.addMessage(text, 'user');
        const typing = this.addTyping();
        
        try {
            const res = await fetch(API.chat, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typing.remove();
            this.addMessage(data.response || 'No response', 'assistant');
        } catch (err) {
            typing.remove();
            this.addMessage('Error: Could not connect', 'error');
        }
    }
    
    addMessage(content, type) {
        const div = document.createElement('div');
        div.className = `message ${type}-message`;
        
        const avatar = type === 'user' ? '👤' : type === 'error' ? '❌' : '🤖';
        div.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.format(content)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        this.messages?.appendChild(div);
        this.scroll();
        return div;
    }
    
    addTyping() {
        const div = document.createElement('div');
        div.className = 'message assistant-message';
        div.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        `;
        this.messages?.appendChild(div);
        this.scroll();
        return div;
    }
    
    format(text) {
        return text
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
    }
    
    scroll() {
        if (this.messages) this.messages.scrollTop = this.messages.scrollHeight;
    }
    
    clearChat() {
        if (this.messages) {
            this.messages.innerHTML = `
                <div class="message system">
                    <div class="message-content"><p>💬 Chat cleared</p></div>
                </div>
            `;
        }
    }
}

// ============================================================
// Stats Dashboard
// ============================================================
class StatsDashboard {
    constructor() {
        this.load();
        setInterval(() => this.load(), 30000);
    }
    
    async load() {
        try {
            const res = await fetch(API.stats);
            const data = await res.json();
            this.update(data);
        } catch (err) {
            console.error('Stats error:', err);
        }
    }
    
    update(s) {
        const params = s.params || s.neural?.parameters || 0;
        const knowledge = s.knowledge || s.facts || 0;
        
        const paramsEl = document.getElementById('stat-params');
        const knowledgeEl = document.getElementById('stat-knowledge');
        
        if (paramsEl) paramsEl.textContent = params > 1e6 ? (params/1e6).toFixed(1) + 'M' : params.toLocaleString();
        if (knowledgeEl) knowledgeEl.textContent = knowledge.toLocaleString();
        
        // Learning tab
        const el = (id, val) => {
            const e = document.getElementById(id);
            if (e) e.textContent = (val || 0).toLocaleString();
        };
        
        el('learning-articles', s.articles_learned);
        el('learning-tokens', s.tokens_trained);
        el('learning-chunks', s.knowledge || s.facts);
        el('learning-vectors', s.vectors);
    }
}

// ============================================================
// Knowledge Search
// ============================================================
class KnowledgeExplorer {
    constructor() {
        this.container = document.getElementById('knowledge-container');
        this.input = document.getElementById('knowledge-search');
        this.btn = document.getElementById('search-btn');
        
        this.btn?.addEventListener('click', () => this.search());
        this.input?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.search();
        });
    }
    
    async search() {
        const q = this.input?.value?.trim();
        if (!q) return;
        
        this.container.innerHTML = '<p class="searching">🔍 Searching...</p>';
        
        try {
            const res = await fetch(`${API.search}?q=${encodeURIComponent(q)}&k=10`);
            const data = await res.json();
            this.render(data.results || [], data.query);
        } catch (err) {
            this.container.innerHTML = '<p class="error">❌ Search failed. Please try again.</p>';
        }
    }
    
    render(results, query) {
        if (!results.length) {
            this.container.innerHTML = `
                <div class="no-results">
                    <p>🔍 No results found for "<strong>${query}</strong>"</p>
                    <p class="hint">Try different keywords or check the Learning tab to add knowledge.</p>
                </div>
            `;
            return;
        }
        
        this.container.innerHTML = `
            <div class="results-header">
                <p>Found <strong>${results.length}</strong> results for "<strong>${query}</strong>"</p>
            </div>
            ${results.map((r, i) => `
                <div class="search-result ${r.type}">
                    <div class="result-header">
                        <span class="result-type">${this.getTypeLabel(r.type)}</span>
                        <span class="result-source">${r.source || ''}</span>
                        ${r.score ? `<span class="result-score">Score: ${r.score}</span>` : ''}
                    </div>
                    <div class="result-content">${this.highlightQuery(r.content || '', query)}</div>
                </div>
            `).join('')}
        `;
    }
    
    getTypeLabel(type) {
        const labels = {
            'knowledge': '📚 Knowledge',
            'fact': '💡 Fact',
            'vector': '🎯 Vector Match',
            'vector_id': '🔢 Vector ID'
        };
        return labels[type] || type;
    }
    
    highlightQuery(content, query) {
        if (!query) return content;
        const words = query.toLowerCase().split(/\s+/);
        let result = content;
        words.forEach(word => {
            if (word.length > 2) {
                const regex = new RegExp(`(${word})`, 'gi');
                result = result.replace(regex, '<mark>$1</mark>');
            }
        });
        return result;
    }
}

// ============================================================
// Tab Navigation
// ============================================================
function setupTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabs = document.querySelectorAll('.tab-content');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tab = item.dataset.tab;
            
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            tabs.forEach(t => {
                t.classList.toggle('active', t.id === `tab-${tab}`);
            });
        });
    });
}

// ============================================================
// Toast
// ============================================================
function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================
// Initialize
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('GroundZero AI v2.3 initializing...');
    
    ThemeManager.init();
    setupTabs();
    
    window.chatInterface = new ChatInterface();
    window.voiceRecorder = new VoiceRecorder();
    window.voiceRecorder.init();
    window.statsDashboard = new StatsDashboard();
    window.knowledgeExplorer = new KnowledgeExplorer();
    
    // Learning buttons
    document.getElementById('btn-start-learning')?.addEventListener('click', async () => {
        try {
            await fetch(API.learningStart, { method: 'POST' });
            showToast('Learning started', 'success');
        } catch (e) {
            showToast('Failed', 'error');
        }
    });
    
    document.getElementById('btn-stop-learning')?.addEventListener('click', async () => {
        try {
            await fetch(API.learningStop, { method: 'POST' });
            showToast('Learning stopped', 'success');
        } catch (e) {
            showToast('Failed', 'error');
        }
    });
    
    console.log('GroundZero AI ready!');
});