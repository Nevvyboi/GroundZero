/**
 * GroundZero - Frontend Application
 * With Knowledge Explorer and real-time source updates
 */

class GroundZeroApp {
    constructor() {
        this.isLearning = false;
        this.isProcessing = false;
        this.pollInterval = null;
        
        // Learning timer
        this.timerInterval = null;
        this.timerStartTime = null;
        this.timerElapsed = 0;
        
        // Content storage for zoom view
        this.currentFullContent = '';
        this.currentTitle = '';
        this.currentUrl = '';
        
        // Knowledge Explorer
        this.knowledgeEntries = [];
        this.selectedEntryId = null;
        
        // Track source count - only refresh every 30 new sources
        this.lastSourceCount = 0;
        
        // Prevent concurrent status calls
        this.isLoadingStatus = false;
        
        this.init();
    }
    
    init() {
        this.connectSocket();
        this.bindEvents();
        this.loadStatus();
        this.loadRecentSources();
        this.initSessionTracking();
    }
    
    connectSocket() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
            this.socket = new WebSocket(wsUrl);
            this.socket.onopen = () => console.log('WebSocket connected');
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'article_complete') {
                        this.loadStatus();
                        this.loadRecentSources();
                    }
                } catch (e) {}
            };
            this.socket.onerror = () => console.log('WebSocket error, using polling');
        } catch (e) {
            console.log('WebSocket not available');
        }
    }
    
    bindEvents() {
        // Learning controls
        document.getElementById('start-btn').onclick = () => this.startLearning();
        document.getElementById('stop-btn').onclick = () => this.stopLearning();
        
        // Chat
        document.getElementById('send-btn').onclick = () => this.sendMessage();
        document.getElementById('chat-input').onkeydown = e => {
            if (e.key === 'Enter' && !e.shiftKey) { 
                e.preventDefault(); 
                this.sendMessage(); 
            }
        };
        
        // Auto-resize textarea
        document.getElementById('chat-input').oninput = e => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
        };
        
        // Teach
        document.getElementById('teach-btn').onclick = () => this.teach();
        
        // Learn from URL
        document.getElementById('learn-url-btn').onclick = () => this.learnFromUrl();
        
        // Zoom modal
        document.getElementById('zoom-btn').onclick = () => this.openZoomModal();
        document.getElementById('zoom-close').onclick = () => this.closeZoomModal();
        document.getElementById('zoom-modal').onclick = (e) => {
            if (e.target.id === 'zoom-modal') this.closeZoomModal();
        };
        
        // Knowledge Explorer
        document.getElementById('knowledge-explorer-btn').onclick = () => this.openKnowledgeExplorer();
        document.getElementById('ke-close').onclick = () => this.closeKnowledgeExplorer();
        document.getElementById('knowledge-explorer-modal').onclick = (e) => {
            if (e.target.id === 'knowledge-explorer-modal') this.closeKnowledgeExplorer();
        };
        document.getElementById('ke-search-input').oninput = (e) => this.filterKnowledge(e.target.value);
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeZoomModal();
                this.closeKnowledgeExplorer();
            }
        });
    }
    
    // ========== ZOOM MODAL ==========
    
    openZoomModal() {
        const modal = document.getElementById('zoom-modal');
        const content = document.getElementById('zoom-content');
        const title = document.getElementById('zoom-title');
        const stats = document.getElementById('zoom-stats');
        const source = document.getElementById('zoom-source');
        
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        title.textContent = this.currentTitle || 'Full Content';
        source.href = this.currentUrl || '#';
        source.textContent = this.currentUrl ? 'View Source' : '';
        
        if (this.currentFullContent) {
            const words = this.currentFullContent.split(/\s+/).filter(w => w.length > 0);
            stats.textContent = `${words.length.toLocaleString()} words`;
            content.textContent = this.currentFullContent;
        } else {
            content.innerHTML = '<p class="empty-text">No content loaded yet. Start learning to see full content here.</p>';
            stats.textContent = '0 words';
        }
    }
    
    closeZoomModal() {
        document.getElementById('zoom-modal').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    // ========== TIMER ==========
    
    startTimer() {
        this.timerElapsed = 0;
        this.timerStartTime = Date.now();
        
        document.getElementById('learning-timer').classList.add('active');
        document.getElementById('timer-status').textContent = 'RUNNING';
        
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => this.updateTimer(), 1000);
        this.updateTimer();
    }
    
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        document.getElementById('learning-timer').classList.remove('active');
        document.getElementById('timer-status').textContent = 'STOPPED';
    }
    
    updateTimer() {
        if (!this.timerStartTime) return;
        const elapsed = Date.now() - this.timerStartTime;
        document.getElementById('timer-value').textContent = this.formatTime(elapsed);
    }
    
    formatTime(ms) {
        const totalSeconds = Math.floor(ms / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    // ========== STATUS & STATS ==========
    
    async loadStatus() {
        if (this.isLoadingStatus) return;
        this.isLoadingStatus = true;
        
        try {
            const res = await fetch('/api/status');
            if (!res.ok) throw new Error('Status fetch failed');
            
            const data = await res.json();
            
            if (data.stats) {
                const stats = data.stats;
                
                // Update article title, URL, and content
                if (stats.current_article) {
                    this.safeSetText('current-url', stats.current_article);
                    this.currentTitle = stats.current_article;
                }
                if (stats.current_url) {
                    const urlEl = document.getElementById('current-url');
                    if (urlEl) urlEl.href = stats.current_url;
                    this.currentUrl = stats.current_url;
                }
                if (stats.current_content) {
                    this.currentFullContent = stats.current_content;
                    const previewEl = document.getElementById('content-preview');
                    if (previewEl) previewEl.textContent = stats.current_content.substring(0, 500) + '...';
                }
                
                // Update session stats
                if (stats.current_session) {
                    this.safeSetText('current-articles', stats.current_session.articles_read || 0);
                    this.safeSetText('current-words', this.formatNum(stats.current_session.words_learned || 0));
                    this.safeSetText('current-knowledge', stats.current_session.knowledge_added || 0);
                }
                
                // Update running state
                if (stats.is_running && !this.isLearning) {
                    this.isLearning = true;
                    this.updateLearningUI(true);
                    this.startPolling();
                    this.startTimer();
                } else if (!stats.is_running && this.isLearning) {
                    this.isLearning = false;
                    this.updateLearningUI(false);
                    this.stopPolling();
                    this.stopTimer();
                    // Load sources when learning stops
                    this.loadRecentSources();
                }
                
                // Update totals
                if (stats.total) {
                    const total = stats.total;
                    this.safeSetText('vocab-count', this.formatNum(total.vocabulary_size || 0));
                    this.safeSetText('vocab-stat', this.formatNum(total.vocabulary_size || 0));
                    this.safeSetText('knowledge-count', this.formatNum(total.total_knowledge || 0));
                    this.safeSetText('knowledge-stat', this.formatNum(total.total_knowledge || 0));
                    this.safeSetText('sources-count', this.formatNum(total.total_sources || 0));
                    this.safeSetText('sources-stat', this.formatNum(total.total_sources || 0));
                    this.safeSetText('tokens-count', this.formatNum(total.total_words || 0));
                    
                    if (total.vectors) {
                        this.safeSetText('vectors-count', this.formatNum(total.vectors.total_vectors || 0));
                    }
                    if (total.total_learning_time !== undefined) {
                        this.safeSetText('total-learning-time', this.formatDuration(total.total_learning_time));
                    }
                    if (total.total_sessions !== undefined) {
                        this.safeSetText('sessions-count', this.formatNum(total.total_sessions));
                    }
                    
                    // Only update sources every 30 new sources
                    const newSourceCount = total.total_sources || 0;
                    if (newSourceCount >= this.lastSourceCount + 30) {
                        this.lastSourceCount = newSourceCount;
                        this.loadRecentSources();
                    }
                }
            }
        } catch (e) {
            console.error('Failed to load status:', e);
        } finally {
            this.isLoadingStatus = false;
        }
    }
    
    async loadRecentSources() {
        try {
            const res = await fetch('/api/knowledge/recent?limit=20');
            if (!res.ok) return;
            
            const data = await res.json();
            const sources = data.recent || [];
            const sourcesEl = document.getElementById('learned-sources');
            
            if (!sourcesEl) return;
            
            if (sources.length > 0) {
                sourcesEl.innerHTML = sources.map(s => 
                    `<div class="source-item">
                        <a href="${s.source_url || '#'}" target="_blank">${this.escapeHtml(s.source_title || 'Source')}</a>
                    </div>`
                ).join('');
            } else {
                sourcesEl.innerHTML = '<p class="empty-text">No sources yet</p>';
            }
        } catch (e) {
            console.error('Failed to load recent sources:', e);
        }
    }
    
    safeSetText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }
    
    // ========== LEARNING ==========
    
    async startLearning() {
        try {
            document.getElementById('current-url').textContent = 'Finding content...';
            document.getElementById('content-preview').textContent = 'Searching for knowledge sources...';
            
            const res = await fetch('/api/learn/start', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started') {
                this.isLearning = true;
                this.updateLearningUI(true);
                this.startPolling();
                this.startTimer();
                this.toast('Learning started!', 'success');
            }
        } catch (e) {
            this.toast('Failed to start learning', 'error');
        }
    }
    
    async stopLearning() {
        try {
            await fetch('/api/learn/stop', { method: 'POST' });
            this.isLearning = false;
            this.updateLearningUI(false);
            this.stopPolling();
            this.stopTimer();
            this.toast('Learning stopped', 'info');
            this.loadRecentSources();
            this.loadStatus();
            this.loadSessions();
            
            // Reset current session display
            document.getElementById('current-articles').textContent = '0';
            document.getElementById('current-words').textContent = '0';
            document.getElementById('current-knowledge').textContent = '0';
        } catch (e) {
            this.toast('Failed to stop learning', 'error');
        }
    }
    
    startPolling() {
        if (this.pollInterval) return;
        // Poll every 2 seconds - gives time for requests to complete
        this.pollInterval = setInterval(() => this.loadStatus(), 2000);
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    updateLearningUI(isLearning) {
        document.getElementById('start-btn').disabled = isLearning;
        document.getElementById('stop-btn').disabled = !isLearning;
        document.getElementById('start-btn').textContent = isLearning ? '⏳ Learning...' : '▶ Start';
        
        const status = document.getElementById('status-indicator');
        status.className = isLearning ? 'stat-pill status learning' : 'stat-pill status';
        status.innerHTML = `<span class="pulse"></span>${isLearning ? 'Learning' : 'Ready'}`;
    }
    
    // ========== CHAT ==========
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        input.value = '';
        input.style.height = 'auto';
        
        this.addUserMessage(message);
        const thinkingId = this.addThinkingMessage();
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, auto_search: true })
            });
            
            const data = await res.json();
            this.removeThinkingMessage(thinkingId);
            this.addAIMessage(data);
            this.loadStatus();
            this.loadRecentSources();
        } catch (e) {
            this.removeThinkingMessage(thinkingId);
            this.addAIMessage({ response: "Sorry, I encountered an error. Please try again." });
        }
        
        this.isProcessing = false;
    }
    
    addUserMessage(text) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `<div class="content"><p>${this.escapeHtml(text)}</p><span class="time">Just now</span></div>`;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    addThinkingMessage() {
        const container = document.getElementById('chat-messages');
        const id = 'thinking-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message ai thinking';
        div.id = id;
        div.innerHTML = `<div class="avatar">🧠</div><div class="content"><div class="thinking-dots"><span></span><span></span><span></span></div></div>`;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
        return id;
    }
    
    removeThinkingMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
    
    addAIMessage(data) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'message ai';
        
        let html = `<div class="avatar">🧠</div><div class="content">`;
        
        const responseHtml = this.parseMarkdown(data.response || data.answer || "I don't have an answer for that.");
        html += responseHtml;
        
        if (data.sources && data.sources.length > 0) {
            html += `<div class="sources"><strong>Sources:</strong>`;
            data.sources.forEach(s => {
                html += `<a href="${s.url}" target="_blank">${this.escapeHtml(s.title || 'Source')}</a>`;
            });
            html += `</div>`;
        }
        
        if (data.learned_from && data.learned_from.length > 0) {
            html += `<div class="learned-from"><strong>🎓 Learned from:</strong>`;
            data.learned_from.forEach(s => {
                html += `<a href="${s.url}" target="_blank">${this.escapeHtml(s.title)}</a>`;
            });
            html += `</div>`;
        }
        
        html += `<span class="time">Just now</span></div>`;
        div.innerHTML = html;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    parseMarkdown(text) {
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .split('<br><br>').map(p => `<p>${p}</p>`).join('');
    }
    
    // ========== TEACH ==========
    
    async teach() {
        const content = document.getElementById('teach-content').value.trim();
        if (!content) {
            this.toast('Enter some knowledge to teach', 'error');
            return;
        }
        
        try {
            const res = await fetch('/api/teach', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content, title: 'User taught knowledge' })
            });
            
            const data = await res.json();
            if (data.success) {
                this.toast('Knowledge taught successfully!', 'success');
                document.getElementById('teach-content').value = '';
                this.loadStatus();
                this.loadRecentSources();
            } else {
                this.toast(data.error || 'Failed to teach', 'error');
            }
        } catch (e) {
            this.toast('Failed to teach', 'error');
        }
    }
    
    // ========== LEARN FROM URL ==========
    
    async learnFromUrl() {
        const url = document.getElementById('learn-url').value.trim();
        if (!url) {
            this.toast('Enter a URL', 'error');
            return;
        }
        if (!url.startsWith('http')) {
            this.toast('URL must start with http:// or https://', 'error');
            return;
        }
        
        this.toast('Fetching and learning...', 'info');
        
        try {
            const res = await fetch('/api/learn/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            
            const data = await res.json();
            if (data.success) {
                this.toast('Learned from URL!', 'success');
                this.addAIMessage({
                    response: `I learned from: **${data.title || 'Web page'}**\n\nI processed the content and added it to my knowledge base.`,
                    sources: [{ url, title: data.title || 'Web Page' }]
                });
                this.loadRecentSources();
            } else {
                this.toast(data.message || data.reason || 'Failed to learn from URL', 'error');
            }
            document.getElementById('learn-url').value = '';
        } catch (e) {
            this.toast('Failed to learn from URL', 'error');
        }
    }
    
    // ========== SESSION TRACKING ==========
    
    initSessionTracking() {
        const expandBtn = document.getElementById('expand-sessions-btn');
        if (expandBtn) {
            expandBtn.onclick = () => this.toggleSessionHistory();
        }
        this.loadSessions();
    }
    
    toggleSessionHistory() {
        const btn = document.getElementById('expand-sessions-btn');
        const history = document.getElementById('session-history');
        
        if (history.classList.contains('expanded')) {
            history.classList.remove('expanded');
            btn.classList.remove('expanded');
        } else {
            history.classList.add('expanded');
            btn.classList.add('expanded');
            this.loadSessions();
        }
    }
    
    async loadSessions() {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            
            if (data.summary) {
                document.getElementById('total-sessions').textContent = this.formatNum(data.summary.total_sessions || 0);
                document.getElementById('total-articles').textContent = this.formatNum(data.summary.total_articles || 0);
                document.getElementById('total-words-learned').textContent = this.formatNum(data.summary.total_words || 0);
                document.getElementById('total-knowledge-added').textContent = this.formatNum(data.summary.total_knowledge || 0);
                document.getElementById('total-learning-time').textContent = this.formatDuration(data.summary.total_time_seconds || 0);
                document.getElementById('sessions-count').textContent = this.formatNum(data.summary.total_sessions || 0);
            }
            
            const listEl = document.getElementById('session-list');
            if (data.sessions && data.sessions.length > 0) {
                listEl.innerHTML = data.sessions.map(s => `
                    <div class="session-item">
                        <div class="session-header">
                            <span class="session-id">Session #${s.id}</span>
                            <span class="session-status ${s.status === 'active' ? 'active' : ''}">${s.status === 'active' ? 'Active' : 'Completed'}</span>
                        </div>
                        <div class="session-stats-row">
                            <span>⏱️ ${this.formatDuration(s.duration_seconds || 0)}</span>
                            <span>📄 ${s.articles_learned || 0}</span>
                            <span>📝 ${this.formatNum(s.words_learned || 0)}</span>
                        </div>
                    </div>
                `).join('');
            } else {
                listEl.innerHTML = '<div class="no-sessions">No sessions yet</div>';
            }
        } catch (e) {
            console.error('Failed to load sessions:', e);
        }
    }
    
    // ========== KNOWLEDGE EXPLORER ==========
    
    async openKnowledgeExplorer() {
        const modal = document.getElementById('knowledge-explorer-modal');
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Reset state
        this.knowledgeEntries = [];
        this.selectedEntryId = null;
        document.getElementById('ke-search-input').value = '';
        document.getElementById('ke-detail-title').textContent = 'Select a topic';
        document.getElementById('ke-detail-confidence').textContent = '';
        document.getElementById('ke-detail-confidence').style.display = 'none';
        document.getElementById('ke-related-list').innerHTML = '<p class="ke-hint">Click on a topic to see related concepts</p>';
        
        await this.loadKnowledgeEntries();
    }
    
    closeKnowledgeExplorer() {
        document.getElementById('knowledge-explorer-modal').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    async loadKnowledgeEntries() {
        const listEl = document.getElementById('ke-list');
        listEl.innerHTML = '<div class="ke-loading">Loading knowledge base...</div>';
        
        try {
            const res = await fetch('/api/knowledge/all?limit=200');
            const data = await res.json();
            
            this.knowledgeEntries = data.entries || [];
            this.renderKnowledgeList(this.knowledgeEntries);
        } catch (e) {
            listEl.innerHTML = '<div class="ke-loading">Failed to load knowledge base</div>';
            console.error('Failed to load knowledge:', e);
        }
    }
    
    renderKnowledgeList(entries) {
        const listEl = document.getElementById('ke-list');
        
        if (entries.length === 0) {
            listEl.innerHTML = '<div class="ke-empty">No knowledge entries yet.<br>Start learning to build your knowledge base!</div>';
            return;
        }
        
        listEl.innerHTML = entries.map(entry => `
            <div class="ke-item ${this.selectedEntryId === entry.id ? 'selected' : ''}" data-id="${entry.id}">
                <span class="ke-item-title">${this.escapeHtml(entry.title || 'Untitled')}</span>
                <span class="ke-item-confidence">${Math.round((entry.confidence || 0.5) * 100)}%</span>
            </div>
        `).join('');
        
        // Add click handlers
        listEl.querySelectorAll('.ke-item').forEach(item => {
            item.onclick = () => this.selectKnowledgeEntry(parseInt(item.dataset.id));
        });
    }
    
    filterKnowledge(query) {
        if (!this.knowledgeEntries) return;
        
        const filtered = query.trim() === '' 
            ? this.knowledgeEntries 
            : this.knowledgeEntries.filter(e => 
                (e.title || '').toLowerCase().includes(query.toLowerCase())
            );
        
        this.renderKnowledgeList(filtered);
    }
    
    async selectKnowledgeEntry(entryId) {
        this.selectedEntryId = entryId;
        
        // Update selection in list
        document.querySelectorAll('.ke-item').forEach(item => {
            item.classList.toggle('selected', parseInt(item.dataset.id) === entryId);
        });
        
        // Find entry
        const entry = this.knowledgeEntries.find(e => e.id === entryId);
        if (!entry) return;
        
        // Update detail panel
        document.getElementById('ke-detail-title').textContent = entry.title || 'Untitled';
        const confEl = document.getElementById('ke-detail-confidence');
        confEl.textContent = `${Math.round((entry.confidence || 0.5) * 100)}% confidence`;
        confEl.style.display = 'inline-block';
        
        // Load related topics
        const relatedEl = document.getElementById('ke-related-list');
        relatedEl.innerHTML = '<div class="ke-loading-small">Finding related topics...</div>';
        
        try {
            const res = await fetch(`/api/knowledge/${entryId}/related?limit=10`);
            const data = await res.json();
            
            if (data.related && data.related.length > 0) {
                relatedEl.innerHTML = data.related.map(r => `
                    <div class="ke-related-item" data-id="${r.id}">
                        <span class="ke-related-title">${this.escapeHtml(r.title || 'Untitled')}</span>
                        <span class="ke-related-similarity">${r.similarity}%</span>
                    </div>
                `).join('');
                
                // Add click handlers for related items
                relatedEl.querySelectorAll('.ke-related-item').forEach(item => {
                    item.onclick = () => this.selectKnowledgeEntry(parseInt(item.dataset.id));
                });
            } else {
                relatedEl.innerHTML = '<p class="ke-hint">No related topics found</p>';
            }
        } catch (e) {
            relatedEl.innerHTML = '<p class="ke-hint">Failed to load related topics</p>';
            console.error('Failed to load related:', e);
        }
    }
    
    // ========== UTILITIES ==========
    
    toast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
    
    formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }
    
    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => new GroundZeroApp());