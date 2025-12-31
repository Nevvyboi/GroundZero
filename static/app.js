/**
 * NeuralMind - AI Training Lab
 * Frontend JavaScript Application
 */

class NeuralMindApp {
    constructor() {
        this.socket = null;
        this.isLearning = false;
        this.chatMode = 'chat'; // 'chat', 'teach', 'correct'
        
        this.init();
    }
    
    init() {
        this.initSocket();
        this.initParticles();
        this.bindEvents();
        this.loadStatus();
        this.autoResize();
    }
    
    // ============= Socket.IO Connection =============
    
    initSocket() {
        // Connect to the server
        this.socket = io(window.location.origin, {
            transports: ['websocket', 'polling']
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to NeuralMind server');
            this.showToast('Connected to AI server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showToast('Disconnected from server', 'error');
        });
        
        this.socket.on('learning_progress', (data) => {
            this.updateLearningProgress(data);
        });
        
        this.socket.on('learning_content', (data) => {
            this.updateCurrentContent(data);
        });
        
        this.socket.on('learning_error', (data) => {
            this.showToast(`Learning error: ${data.error}`, 'error');
        });
        
        this.socket.on('model_updated', (data) => {
            this.updateModelStats(data);
        });
        
        this.socket.on('status_update', (data) => {
            this.updateStatus(data);
        });
    }
    
    // ============= Particle Animation =============
    
    initParticles() {
        const container = document.getElementById('particles');
        const particleCount = 30;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.top = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 10}s`;
            particle.style.animationDuration = `${5 + Math.random() * 10}s`;
            container.appendChild(particle);
        }
    }
    
    // ============= Event Bindings =============
    
    bindEvents() {
        // Learning controls
        document.getElementById('learn-btn').addEventListener('click', () => this.startLearning());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopLearning());
        document.getElementById('refresh-btn').addEventListener('click', () => this.loadStatus());
        
        // Chat controls
        document.getElementById('send-btn').addEventListener('click', () => this.sendMessage());
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        document.getElementById('clear-chat').addEventListener('click', () => this.clearChat());
        
        // Chat mode buttons
        document.getElementById('teach-btn').addEventListener('click', () => this.toggleMode('teach'));
        document.getElementById('correct-btn').addEventListener('click', () => this.toggleMode('correct'));
        document.getElementById('search-learn-btn').addEventListener('click', () => this.toggleMode('search'));
        
        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
        
        // Teaching
        document.getElementById('submit-teach').addEventListener('click', () => this.submitTeaching());
        document.getElementById('submit-correction').addEventListener('click', () => this.submitCorrection());
        document.getElementById('submit-search').addEventListener('click', () => this.searchAndLearn());
        document.getElementById('submit-url').addEventListener('click', () => this.learnFromUrl());
        
        // Auto-resize textarea
        document.getElementById('chat-input').addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        });
    }
    
    // ============= API Calls =============
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.updateStatus(data);
            this.loadLearningHistory();
        } catch (error) {
            console.error('Failed to load status:', error);
        }
    }
    
    async startLearning() {
        try {
            this.showLoading(true);
            const response = await fetch('/api/learn/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'started') {
                this.isLearning = true;
                this.updateLearningUI(true);
                this.showToast('Learning started! Exploring the internet...', 'success');
            } else if (data.status === 'already_running') {
                this.showToast('Learning is already in progress', 'info');
            }
        } catch (error) {
            this.showToast('Failed to start learning', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async stopLearning() {
        try {
            this.showLoading(true);
            const response = await fetch('/api/learn/stop', { method: 'POST' });
            const data = await response.json();
            
            this.isLearning = false;
            this.updateLearningUI(false);
            this.updateModelStats(data.stats);
            this.showToast('Learning stopped. Model saved!', 'info');
        } catch (error) {
            this.showToast('Failed to stop learning', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        input.value = '';
        input.style.height = 'auto';
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, show_reasoning: true })
            });
            
            const data = await response.json();
            
            // Check if reasoning was used
            if (data.used_reasoning && data.steps && data.steps.length > 0) {
                this.addReasoningMessage(data);
            } else {
                this.addMessage(data.response, 'ai');
            }
            
            this.updateModelStats(data.stats);
        } catch (error) {
            this.addMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }
    }
    
    addReasoningMessage(data) {
        const container = document.getElementById('chat-messages');
        const message = document.createElement('div');
        message.className = 'message ai-message reasoning-message';
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Build reasoning steps HTML
        const stepsHtml = data.steps.map(step => `
            <div class="reasoning-step">
                <div class="step-number">${step.step}</div>
                <div class="step-content">
                    <div class="step-description">${this.escapeHtml(step.description)}</div>
                    <div class="step-result"><code>${this.escapeHtml(String(step.result).substring(0, 200))}</code></div>
                </div>
            </div>
        `).join('');
        
        // Reasoning type badge - professional colors
        const typeColors = {
            'logic': '#1a73e8',
            'math': '#1e8e3e',
            'code': '#673ab7',
            'analysis': '#e37400'
        };
        const typeColor = typeColors[data.reasoning_type] || '#5f6368';
        
        message.innerHTML = `
            <div class="message-avatar">🧠</div>
            <div class="message-content reasoning-content">
                <div class="reasoning-header">
                    <span class="reasoning-badge" style="background: ${typeColor}20; border-color: ${typeColor}; color: ${typeColor}">
                        ${data.reasoning_type.toUpperCase()} REASONING
                    </span>
                    <span class="confidence">Confidence: ${Math.round(data.confidence * 100)}%</span>
                </div>
                <div class="reasoning-steps-container">
                    <div class="reasoning-steps-header" onclick="this.parentElement.classList.toggle('expanded')">
                        <span>📊 Reasoning Steps (${data.steps.length})</span>
                        <span class="expand-icon">▼</span>
                    </div>
                    <div class="reasoning-steps">
                        ${stepsHtml}
                    </div>
                </div>
                <div class="reasoning-answer">
                    <div class="answer-label">Answer:</div>
                    <div class="message-text">${this.formatAnswer(data.response)}</div>
                </div>
                <div class="message-time">${time}</div>
            </div>
        `;
        
        container.appendChild(message);
        container.scrollTop = container.scrollHeight;
    }
    
    formatAnswer(text) {
        // Convert markdown-like formatting
        let formatted = this.escapeHtml(text);
        
        // Bold
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        return formatted;
    }
    
    async submitTeaching() {
        const content = document.getElementById('teach-content').value.trim();
        const source = document.getElementById('teach-source').value.trim() || 'user_teaching';
        
        if (!content) {
            this.showToast('Please enter some content to teach', 'error');
            return;
        }
        
        try {
            this.showLoading(true);
            const response = await fetch('/api/teach', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content, source })
            });
            
            const data = await response.json();
            this.updateModelStats(data.stats);
            this.showToast('Knowledge acquired! I learned something new.', 'success');
            
            // Clear inputs
            document.getElementById('teach-content').value = '';
            document.getElementById('teach-source').value = '';
            
            // Add confirmation to chat
            this.addMessage(`I've learned this new information: "${content.substring(0, 100)}..."`, 'ai');
        } catch (error) {
            this.showToast('Failed to teach the AI', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async submitCorrection() {
        const wrongResponse = document.getElementById('wrong-response').value.trim();
        const correctInfo = document.getElementById('correct-info').value.trim();
        const shouldSearch = document.getElementById('search-correct').checked;
        
        if (!correctInfo) {
            this.showToast('Please provide the correct information', 'error');
            return;
        }
        
        try {
            this.showLoading(true);
            const response = await fetch('/api/correct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    wrong_response: wrongResponse,
                    correct_info: correctInfo,
                    search: shouldSearch
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'searched_and_learned') {
                this.showToast(`Learned from ${data.sources?.length || 0} sources!`, 'success');
                this.addMessage(`Thank you for the correction! I searched and learned from: ${data.sources?.map(s => s.title).join(', ')}`, 'ai');
            } else {
                this.showToast('Correction recorded!', 'success');
                this.addMessage('Thank you for correcting me! I\'ve updated my knowledge.', 'ai');
            }
            
            this.updateModelStats(data.stats);
            
            // Clear inputs
            document.getElementById('wrong-response').value = '';
            document.getElementById('correct-info').value = '';
        } catch (error) {
            this.showToast('Failed to submit correction', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async searchAndLearn() {
        const query = document.getElementById('search-query').value.trim();
        
        if (!query) {
            this.showToast('Please enter a search query', 'error');
            return;
        }
        
        try {
            this.showLoading(true);
            const response = await fetch('/api/learn/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showToast(`Learned about "${query}" from ${data.count} sources!`, 'success');
                this.displaySearchResults(data.learned_from);
                this.addMessage(`I just learned about "${query}" from ${data.count} Wikipedia articles!`, 'ai');
                this.updateModelStats(data.model_stats);
            } else {
                this.showToast('No results found', 'info');
            }
        } catch (error) {
            this.showToast('Search failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async learnFromUrl() {
        const url = document.getElementById('learn-url').value.trim();
        
        if (!url) {
            this.showToast('Please enter a URL', 'error');
            return;
        }
        
        try {
            this.showLoading(true);
            const response = await fetch('/api/learn/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showToast('Learned from URL!', 'success');
                this.addMessage(`I learned from ${url}!`, 'ai');
                this.updateModelStats(data.stats);
            } else {
                this.showToast(`Failed: ${data.error || 'Unknown error'}`, 'error');
            }
            
            document.getElementById('learn-url').value = '';
        } catch (error) {
            this.showToast('Failed to learn from URL', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadLearningHistory() {
        try {
            const response = await fetch('/api/learn/history?n=20');
            const data = await response.json();
            this.displayHistory(data.history);
            this.displayRecentSites(data.history);
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    // ============= UI Updates =============
    
    updateStatus(data) {
        // Update header stats
        if (data.model) {
            document.getElementById('vocab-count').textContent = this.formatNumber(data.model.vocab_size);
            document.getElementById('memory-count').textContent = this.formatNumber(data.model.memory_size);
            document.getElementById('tokens-learned').textContent = this.formatNumber(data.model.tokens_learned);
            document.getElementById('training-steps').textContent = this.formatNumber(data.model.training_steps);
            document.getElementById('memory-info').textContent = `${data.model.memory_size} entries`;
            
            if (data.model.model_params) {
                document.getElementById('embedding-dim').textContent = data.model.model_params.d_model;
                document.getElementById('attention-heads').textContent = data.model.model_params.n_heads;
            }
        }
        
        if (data.learner) {
            document.getElementById('sites-count').textContent = this.formatNumber(data.learner.sites_learned);
            
            if (data.learner.is_learning) {
                this.isLearning = true;
                this.updateLearningUI(true);
                
                if (data.learner.current_url) {
                    document.getElementById('current-url').textContent = data.learner.current_url;
                }
                if (data.learner.current_preview) {
                    document.getElementById('content-preview').textContent = data.learner.current_preview;
                }
            }
        }
    }
    
    updateModelStats(stats) {
        if (!stats) return;
        
        document.getElementById('vocab-count').textContent = this.formatNumber(stats.vocab_size);
        document.getElementById('memory-count').textContent = this.formatNumber(stats.memory_size);
        document.getElementById('tokens-learned').textContent = this.formatNumber(stats.tokens_learned);
        document.getElementById('training-steps').textContent = this.formatNumber(stats.training_steps);
        document.getElementById('memory-info').textContent = `${stats.memory_size} entries`;
    }
    
    updateLearningProgress(data) {
        // Update current URL
        if (data.current_url) {
            document.getElementById('current-url').textContent = data.current_url;
        }
        
        // Update sites count
        if (data.sites_learned !== undefined) {
            document.getElementById('sites-count').textContent = this.formatNumber(data.sites_learned);
        }
        
        // Update progress bar (simulate progress based on queue)
        const progress = Math.min(100, (data.urls_visited / (data.urls_visited + data.urls_in_queue)) * 100);
        document.getElementById('progress-fill').style.width = `${progress}%`;
        document.getElementById('progress-percent').textContent = `${Math.round(progress)}%`;
        
        // Add to recent sites
        if (data.recent_history && data.recent_history.length > 0) {
            this.displayRecentSites(data.recent_history);
        }
    }
    
    updateCurrentContent(data) {
        document.getElementById('current-url').textContent = data.url;
        document.getElementById('content-preview').textContent = data.preview;
        
        // Flash effect on learning status
        const statusEl = document.getElementById('learning-status');
        statusEl.classList.add('flash');
        setTimeout(() => statusEl.classList.remove('flash'), 300);
    }
    
    updateLearningUI(isLearning) {
        const learnBtn = document.getElementById('learn-btn');
        const stopBtn = document.getElementById('stop-btn');
        const statusPill = document.getElementById('status-pill');
        const learningStatus = document.getElementById('learning-status');
        
        if (isLearning) {
            learnBtn.disabled = true;
            stopBtn.disabled = false;
            learnBtn.querySelector('.btn-text').textContent = 'LEARNING...';
            learnBtn.querySelector('.btn-icon-left').textContent = '⏳';
            statusPill.classList.add('learning');
            statusPill.querySelector('.stat-value').textContent = 'LEARNING';
            learningStatus.classList.add('active');
        } else {
            learnBtn.disabled = false;
            stopBtn.disabled = true;
            learnBtn.querySelector('.btn-text').textContent = 'START LEARNING';
            learnBtn.querySelector('.btn-icon-left').textContent = '▶';
            statusPill.classList.remove('learning');
            statusPill.querySelector('.stat-value').textContent = 'IDLE';
            learningStatus.classList.remove('active');
            document.getElementById('current-url').textContent = 'Waiting to start...';
            document.getElementById('content-preview').textContent = '';
        }
    }
    
    // ============= Chat Functions =============
    
    addMessage(text, type) {
        const container = document.getElementById('chat-messages');
        const message = document.createElement('div');
        message.className = `message ${type}-message`;
        
        const avatar = type === 'user' ? '👤' : '🧠';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        message.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
                <div class="message-time">${time}</div>
            </div>
        `;
        
        container.appendChild(message);
        container.scrollTop = container.scrollHeight;
    }
    
    clearChat() {
        const container = document.getElementById('chat-messages');
        container.innerHTML = `
            <div class="message ai-message">
                <div class="message-avatar">🧠</div>
                <div class="message-content">
                    <div class="message-text">Chat cleared! How can I help you?</div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
        `;
    }
    
    toggleMode(mode) {
        const buttons = document.querySelectorAll('.action-btn');
        buttons.forEach(btn => btn.classList.remove('active'));
        
        if (this.chatMode === mode) {
            this.chatMode = 'chat';
        } else {
            this.chatMode = mode;
            document.getElementById(`${mode}-btn`).classList.add('active');
        }
        
        const input = document.getElementById('chat-input');
        switch (this.chatMode) {
            case 'teach':
                input.placeholder = 'Enter information to teach me...';
                break;
            case 'correct':
                input.placeholder = 'Tell me what I got wrong and the correct information...';
                break;
            case 'search':
                input.placeholder = 'Enter a topic to search and learn about...';
                break;
            default:
                input.placeholder = 'Type a message or teach me something...';
        }
    }
    
    // ============= Tab Functions =============
    
    switchTab(tabId) {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tabId}`);
        });
        
        if (tabId === 'history') {
            this.loadLearningHistory();
        }
    }
    
    // ============= Display Functions =============
    
    displayRecentSites(sites) {
        const container = document.getElementById('sites-list');
        
        if (!sites || sites.length === 0) {
            container.innerHTML = '<div class="empty-state">No sites learned yet. Click "Start Learning" to begin!</div>';
            return;
        }
        
        container.innerHTML = sites.slice(-10).reverse().map(site => `
            <div class="site-item">
                <div class="site-icon">📄</div>
                <div class="site-info">
                    <div class="site-title">${this.escapeHtml(site.title || 'Unknown')}</div>
                    <div class="site-url">${this.escapeHtml(site.url || '')}</div>
                    <div class="site-time">${site.chars_learned ? `${this.formatNumber(site.chars_learned)} chars` : ''}</div>
                </div>
            </div>
        `).join('');
    }
    
    displayHistory(history) {
        const container = document.getElementById('history-timeline');
        
        if (!history || history.length === 0) {
            container.innerHTML = '<div class="empty-state">No learning history yet.</div>';
            return;
        }
        
        container.innerHTML = history.slice().reverse().map(item => {
            const time = item.time ? new Date(item.time).toLocaleString() : '';
            return `
                <div class="history-item">
                    <div class="history-dot"></div>
                    <div class="history-content">
                        <div class="history-title">${this.escapeHtml(item.title || 'Unknown')}</div>
                        <div class="history-meta">${time} • ${this.formatNumber(item.chars_learned || 0)} characters</div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    displaySearchResults(results) {
        const container = document.getElementById('search-results');
        
        if (!results || results.length === 0) {
            container.innerHTML = '<div class="empty-state">No results found.</div>';
            return;
        }
        
        container.innerHTML = results.map(result => `
            <div class="search-result-item">
                <div class="search-result-title">✓ ${this.escapeHtml(result.title)}</div>
                <div class="search-result-snippet">Learned from this article</div>
            </div>
        `).join('');
    }
    
    // ============= Utility Functions =============
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '✓',
            error: '✗',
            info: 'ℹ'
        };
        
        toast.innerHTML = `
            <span class="toast-icon">${icons[type]}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
            <button class="toast-close">&times;</button>
        `;
        
        toast.querySelector('.toast-close').addEventListener('click', () => toast.remove());
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'toastOut 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
    
    showLoading(show) {
        document.getElementById('loading-overlay').classList.toggle('active', show);
    }
    
    formatNumber(num) {
        if (num === undefined || num === null) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    autoResize() {
        const textarea = document.getElementById('chat-input');
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
}

// Add toast out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes toastOut {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100%); }
    }
    
    .flash {
        animation: statusFlash 0.3s ease;
    }
    
    @keyframes statusFlash {
        0%, 100% { border-color: var(--border-color); }
        50% { border-color: var(--cyan); box-shadow: 0 0 20px rgba(0, 245, 255, 0.5); }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.neuralMind = new NeuralMindApp();
});