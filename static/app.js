/**
 * NeuralMind - Frontend Application
 * ChatGPT-like experience with auto-search
 */

class NeuralMindApp {
    constructor() {
        this.socket = null;
        this.isLearning = false;
        this.isProcessing = false;
        this.pollInterval = null;
        this.init();
    }
    
    init() {
        this.connectSocket();
        this.bindEvents();
        this.loadStatus();
        this.loadKnowledgeStats();
    }
    
    connectSocket() {
        this.socket = io(window.location.origin);
        
        this.socket.on('connect', () => console.log('Connected'));
        this.socket.on('disconnect', () => this.toast('Disconnected', 'error'));
        
        this.socket.on('learning_progress', data => this.updateLearningProgress(data));
        this.socket.on('learning_content', data => this.updateCurrentContent(data));
        this.socket.on('learning_complete', data => this.handleLearningComplete(data));
        this.socket.on('learning_error', data => console.error('Learning error:', data.error));
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
    }
    
    async loadStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            this.updateStats(data);
        } catch (e) {
            console.error('Failed to load status:', e);
        }
    }
    
    async loadKnowledgeStats() {
        try {
            const res = await fetch('/api/knowledge/stats');
            const data = await res.json();
            this.displayKnowledgeStats(data);
        } catch (e) {
            console.error('Failed to load knowledge stats:', e);
        }
    }
    
    updateStats(data) {
        if (data.memory) {
            document.getElementById('vocab-count').textContent = this.formatNum(data.memory.vocabulary_size || 0);
            document.getElementById('vocab-stat').textContent = this.formatNum(data.memory.vocabulary_size || 0);
            document.getElementById('knowledge-count').textContent = this.formatNum(data.memory.knowledge_count || 0);
            document.getElementById('knowledge-stat').textContent = this.formatNum(data.memory.knowledge_count || 0);
            document.getElementById('sources-count').textContent = this.formatNum(data.memory.sources_learned || 0);
            document.getElementById('sources-stat').textContent = this.formatNum(data.memory.sources_learned || 0);
        }
        
        if (data.model) {
            document.getElementById('tokens-count').textContent = this.formatNum(data.model.total_tokens_learned || 0);
            document.getElementById('steps-count').textContent = this.formatNum(data.model.training_steps || 0);
        }
        
        if (data.learner) {
            const progress = data.learner.progress || 0;
            document.getElementById('progress-fill').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
            document.getElementById('chunks-stat').textContent = this.formatNum(data.learner.total_chunks || 0);
            
            // Update current content preview
            if (data.learner.current_title) {
                document.getElementById('current-url').textContent = data.learner.current_title;
                document.getElementById('current-url').href = data.learner.current_url || '#';
            }
            if (data.learner.current_preview) {
                document.getElementById('content-preview').textContent = data.learner.current_preview;
            }
            
            if (data.learner.is_running && !this.isLearning) {
                this.isLearning = true;
                this.updateLearningUI(true);
            }
        }
    }
    
    displayKnowledgeStats(data) {
        // Learned sources
        const sourcesEl = document.getElementById('learned-sources');
        if (data.recent_sources?.length) {
            sourcesEl.innerHTML = data.recent_sources.slice(0, 10).map(s => 
                `<div class="source-item"><a href="${s.url}" target="_blank">${this.escapeHtml(s.title || 'Source')}</a></div>`
            ).join('');
        } else {
            sourcesEl.innerHTML = '<p class="empty-text">No sources yet</p>';
        }
        
        // Recent words (filter out punctuation and short stuff)
        const wordsEl = document.getElementById('recent-words');
        if (data.recent_words?.length) {
            const validWords = data.recent_words.filter(w => {
                const word = w.word || w;
                // Filter out punctuation, numbers only, and very short words
                return word.length > 2 && /^[a-zA-Z]+$/.test(word);
            }).slice(0, 30);
            
            if (validWords.length > 0) {
                wordsEl.innerHTML = validWords.map((w, i) => {
                    const word = w.word || w;
                    const isNew = i < 5;
                    return `<span class="word-tag${isNew ? ' new' : ''}">${this.escapeHtml(word)}</span>`;
                }).join('');
            } else {
                wordsEl.innerHTML = '<p class="empty-text">No words yet</p>';
            }
        } else {
            wordsEl.innerHTML = '<p class="empty-text">No words yet</p>';
        }
    }
    
    // === LEARNING ===
    
    async startLearning() {
        try {
            const res = await fetch('/api/learn/start', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started') {
                this.isLearning = true;
                this.updateLearningUI(true);
                this.startPolling();
                this.toast('Learning started!', 'success');
            } else if (data.status === 'complete') {
                this.toast(data.message, 'info');
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
            this.toast('Learning stopped', 'info');
            this.loadKnowledgeStats();
        } catch (e) {
            this.toast('Failed to stop learning', 'error');
        }
    }
    
    startPolling() {
        if (this.pollInterval) return;
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
        
        // Don't show percentage, just the learning status
        document.getElementById('start-btn').textContent = isLearning ? '⏳ Learning...' : '▶ Start';
        
        const status = document.getElementById('status-indicator');
        status.className = isLearning ? 'stat-pill status learning' : 'stat-pill status';
        status.innerHTML = `<span class="pulse"></span>${isLearning ? 'Learning' : 'Ready'}`;
    }
    
    updateLearningProgress(data) {
        // Update the progress bar (but hide percentage text)
        const progress = data.progress || 0;
        document.getElementById('progress-fill').style.width = `${progress}%`;
        
        // Show source count instead of percentage
        document.getElementById('progress-text').textContent = `${data.sites_learned || 0} sources`;
        
        // Update all stats
        if (data.model) {
            document.getElementById('vocab-count').textContent = this.formatNum(data.model.vocab_size || 0);
            document.getElementById('vocab-stat').textContent = this.formatNum(data.model.vocab_size || 0);
            document.getElementById('tokens-count').textContent = this.formatNum(data.model.total_tokens_learned || 0);
            document.getElementById('steps-count').textContent = this.formatNum(data.model.training_steps || 0);
        }
        
        document.getElementById('sources-count').textContent = this.formatNum(data.sites_learned || 0);
        document.getElementById('sources-stat').textContent = this.formatNum(data.sites_learned || 0);
        document.getElementById('chunks-stat').textContent = this.formatNum(data.total_chunks || 0);
        
        // Update current reading with animated word display
        if (data.current_title) {
            const urlEl = document.getElementById('current-url');
            urlEl.textContent = data.current_title;
            urlEl.href = data.current_url || '#';
            
            // Show source domain
            let domain = '';
            try {
                domain = new URL(data.current_url).hostname.replace('www.', '').replace('en.', '');
            } catch { domain = ''; }
            
            if (domain) {
                document.querySelector('.currently-reading h4').innerHTML = 
                    `📖 Reading from <span class="domain-badge">${domain}</span>`;
            }
        }
        
        // Animated text preview with word highlighting
        if (data.current_preview) {
            this.showAnimatedReading(data.current_preview);
        }
        
        // Update learned sources list dynamically
        this.loadKnowledgeStats();
    }
    
    showAnimatedReading(text) {
        const previewEl = document.getElementById('content-preview');
        if (!previewEl || !text) return;
        
        // Split into words and show with animation
        const words = text.substring(0, 300).split(/\s+/);
        const highlighted = words.map((word, i) => {
            // Highlight current word (simulate reading)
            const isCurrentWord = i === Math.floor(Date.now() / 100) % words.length;
            return isCurrentWord 
                ? `<span class="reading-word highlighted">${this.escapeHtml(word)}</span>`
                : `<span class="reading-word">${this.escapeHtml(word)}</span>`;
        }).join(' ');
        
        previewEl.innerHTML = highlighted + (text.length > 300 ? '...' : '');
    }
    
    updateCurrentContent(data) {
        const urlEl = document.getElementById('current-url');
        urlEl.textContent = data.title || data.url;
        urlEl.href = data.url;
        
        // Show content preview
        if (data.preview) {
            document.getElementById('content-preview').textContent = data.preview;
        }
        
        // Show length info
        if (data.length) {
            const kb = (data.length / 1024).toFixed(1);
            console.log(`Reading: ${data.title} (${kb} KB)`);
        }
    }
    
    handleLearningComplete(data) {
        this.isLearning = false;
        this.updateLearningUI(false);
        this.stopPolling();
        this.toast(data.message || '🎉 Learning complete!', 'success');
        this.loadKnowledgeStats();
    }
    
    // === CHAT ===
    
    async sendMessage() {
        if (this.isProcessing) return;
        
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message) return;
        
        // Add user message
        this.addMessage(message, 'user');
        input.value = '';
        input.style.height = 'auto';
        
        // Show typing indicator
        const typingId = this.showTyping();
        this.isProcessing = true;
        
        try {
            // First try to get a response
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, auto_search: true })
            });
            
            const data = await res.json();
            this.removeTyping(typingId);
            
            // Check if we need to search
            if (data.needs_search) {
                await this.handleAutoSearch(message, data);
            } else {
                this.addAIMessage(data);
            }
            
            // Update stats
            if (data.stats) {
                document.getElementById('vocab-count').textContent = this.formatNum(data.stats.vocab_size || 0);
                document.getElementById('tokens-count').textContent = this.formatNum(data.stats.total_tokens_learned || 0);
            }
            
            this.loadKnowledgeStats();
            
        } catch (e) {
            this.removeTyping(typingId);
            this.addMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }
        
        this.isProcessing = false;
    }
    
    async handleAutoSearch(query, initialData) {
        // Show search progress UI
        const searchId = this.showSearchProgress(query);
        
        try {
            // Step 1: Searching
            this.updateSearchStep(searchId, 'search', 'active', `🔍 Searching the web for "${query.substring(0, 30)}${query.length > 30 ? '...' : ''}"...`);
            
            // Call the search endpoint
            const res = await fetch('/api/chat/search-and-respond', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const data = await res.json();
            
            // Step 2: Search complete
            this.updateSearchStep(searchId, 'search', 'done', `✓ Found relevant sources`);
            
            if (data.sources_count > 0 || data.learned_sources?.length > 0) {
                const sources = data.learned_sources || data.sources || [];
                
                // Show each source being learned
                for (let i = 0; i < sources.length && i < 3; i++) {
                    const source = sources[i];
                    await this.sleep(200);
                    
                    // Extract domain for display
                    let domain = '';
                    try {
                        domain = new URL(source.url).hostname.replace('www.', '');
                    } catch { domain = 'web'; }
                    
                    this.addSearchStep(searchId, `source-${i}`, 'done', 
                        `📖 Learning from ${domain}: ${source.title?.substring(0, 35) || 'article'}${source.title?.length > 35 ? '...' : ''}`
                    );
                }
                
                // Final step
                await this.sleep(300);
                this.addSearchStep(searchId, 'done', 'done', `✨ Learned from ${sources.length} source${sources.length > 1 ? 's' : ''} - Ready!`);
            } else {
                this.addSearchStep(searchId, 'none', 'done', `No new information found, using existing knowledge`);
            }
            
            // Wait then show response
            await this.sleep(500);
            this.removeSearchProgress(searchId);
            
            this.addAIMessage(data);
            
            // Update all stats immediately
            this.loadStatus();
            this.loadKnowledgeStats();
            
        } catch (e) {
            this.removeSearchProgress(searchId);
            this.addMessage("I tried searching but encountered an error. Please try again.", 'ai');
        }
    }
    
    showSearchProgress(query) {
        const container = document.getElementById('chat-messages');
        const id = 'search-' + Date.now();
        const msg = document.createElement('div');
        msg.id = id;
        msg.className = 'message ai';
        msg.innerHTML = `
            <div class="avatar">🧠</div>
            <div class="content">
                <div class="search-indicator" id="${id}-indicator">
                    <div class="search-header">
                        <div class="spinner"></div>
                        <span>Searching the web...</span>
                    </div>
                    <div class="search-steps" id="${id}-steps">
                    </div>
                </div>
            </div>
        `;
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        return id;
    }
    
    updateSearchStep(searchId, stepId, status, text) {
        const stepsContainer = document.getElementById(`${searchId}-steps`);
        if (!stepsContainer) return;
        
        // Check if step already exists
        let step = document.getElementById(`${searchId}-${stepId}`);
        if (step) {
            step.className = `search-step ${status}`;
            step.querySelector('.step-text').textContent = text;
        } else {
            this.addSearchStep(searchId, stepId, status, text);
        }
    }
    
    addSearchStep(searchId, stepId, status, text) {
        const stepsContainer = document.getElementById(`${searchId}-steps`);
        if (!stepsContainer) return;
        
        const step = document.createElement('div');
        step.id = `${searchId}-${stepId}`;
        step.className = `search-step ${status}`;
        step.innerHTML = `<span class="step-icon"></span><span class="step-text">${this.escapeHtml(text)}</span>`;
        stepsContainer.appendChild(step);
        
        // Scroll to bottom
        const container = document.getElementById('chat-messages');
        container.scrollTop = container.scrollHeight;
    }
    
    removeSearchProgress(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    showTyping() {
        const container = document.getElementById('chat-messages');
        const id = 'typing-' + Date.now();
        const msg = document.createElement('div');
        msg.id = id;
        msg.className = 'message ai';
        msg.innerHTML = `
            <div class="avatar">🧠</div>
            <div class="content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        return id;
    }
    
    removeTyping(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
    
    addMessage(text, type) {
        const container = document.getElementById('chat-messages');
        const msg = document.createElement('div');
        msg.className = `message ${type}`;
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        msg.innerHTML = `
            <div class="avatar">${type === 'user' ? '👤' : '🧠'}</div>
            <div class="content">
                <p>${this.escapeHtml(text)}</p>
                <span class="time">${time}</span>
            </div>
        `;
        
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
    }
    
    addAIMessage(data) {
        const container = document.getElementById('chat-messages');
        const msg = document.createElement('div');
        msg.className = 'message ai';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        let content = '';
        
        // Show thinking mode if available
        if (data.thinking_mode) {
            const modeIcon = data.thinking_mode === 'slow' ? '🧠' : '⚡';
            const modeText = data.thinking_mode === 'slow' ? 'Deep Analysis' : 'Quick Response';
            content += `<div class="thinking-badge">${modeIcon} ${modeText}</div>`;
        }
        
        // Show thought process if available (collapsible)
        if (data.thought_process?.length > 0) {
            content += `<details class="thought-process">
                <summary>💭 View reasoning steps (${data.thought_process.length} steps)</summary>
                <div class="thought-steps">`;
            
            data.thought_process.forEach((step, i) => {
                const icon = this.getThoughtIcon(step.type);
                const conf = Math.round((step.confidence || 0.5) * 100);
                content += `<div class="thought-step">
                    <span class="step-icon">${icon}</span>
                    <span class="step-content">${this.escapeHtml(step.step)}</span>
                    <span class="step-confidence">${conf}%</span>
                </div>`;
            });
            
            content += `</div></details>`;
        }
        
        // Format response
        const responseText = this.formatResponse(data.response || 'No response');
        content += `<p>${responseText}</p>`;
        
        // Show confidence if not 100%
        if (data.confidence && data.confidence < 0.95) {
            const confPercent = Math.round(data.confidence * 100);
            const confClass = confPercent >= 70 ? 'high' : confPercent >= 50 ? 'medium' : 'low';
            content += `<div class="confidence-indicator ${confClass}">Confidence: ${confPercent}%</div>`;
        }
        
        // Add sources if available
        if (data.sources?.length) {
            content += '<div class="message-sources">📚 Sources: ';
            content += data.sources.map(s => 
                `<a href="${s.url}" target="_blank">${this.escapeHtml(s.title || 'Link')}</a>`
            ).join(', ');
            content += '</div>';
        }
        
        content += `<span class="time">${time}</span>`;
        
        msg.innerHTML = `
            <div class="avatar">🧠</div>
            <div class="content">${content}</div>
        `;
        
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
    }
    
    getThoughtIcon(type) {
        const icons = {
            'observation': '👁️',
            'analysis': '🔍',
            'hypothesis': '💡',
            'verification': '✓',
            'synthesis': '🔗',
            'conclusion': '✅',
            'uncertainty': '❓',
            'reflection': '🤔',
            'specialized': '⚙️',
            'intuitive': '⚡'
        };
        return icons[type] || '•';
    }
    
    formatResponse(text) {
        let html = this.escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        return html;
    }
    
    // === TEACH & LEARN ===
    
    async teach() {
        const content = document.getElementById('teach-content').value.trim();
        
        if (!content) {
            this.toast('Enter content to teach', 'error');
            return;
        }
        
        try {
            const res = await fetch('/api/teach', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content, source: 'user_teaching' })
            });
            
            await res.json();
            this.toast('Knowledge acquired!', 'success');
            document.getElementById('teach-content').value = '';
            this.loadKnowledgeStats();
            
        } catch (e) {
            this.toast('Failed to teach', 'error');
        }
    }
    
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
                this.toast(`Learned from URL!`, 'success');
                this.addAIMessage({
                    response: `I learned from: **${data.title || 'Web page'}**\n\nI processed ${data.chunks || 0} chunks of content and added it to my knowledge base. You can now ask me about this topic!`,
                    sources: [{ url: url, title: data.title || 'Web Page' }]
                });
            } else {
                this.toast(data.reason || 'Failed to learn from URL', 'error');
            }
            
            document.getElementById('learn-url').value = '';
            this.loadKnowledgeStats();
            
        } catch (e) {
            this.toast('Failed to learn from URL', 'error');
        }
    }
    
    // === UTILITIES ===
    
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
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => new NeuralMindApp());