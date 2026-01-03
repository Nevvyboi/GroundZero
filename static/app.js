/**
 * NeuralMind - Frontend Application
 * ChatGPT-like experience with auto-search
 */

class NeuralMindApp {
    constructor() {
        this.socket = null;
        this.isLearning = false;
        this.isSearching = false;  // For manual search animation
        this.isProcessing = false;
        this.pollInterval = null;
        
        // Learning timer
        this.timerInterval = null;
        this.timerStartTime = null;
        this.timerElapsed = 0;
        
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
    
    // ========== LEARNING TIMER METHODS ==========
    
    startTimer() {
        // Reset timer on each start
        this.timerElapsed = 0;
        this.timerStartTime = Date.now();
        
        // Update UI
        const timerEl = document.getElementById('learning-timer');
        const statusEl = document.getElementById('timer-status');
        
        timerEl.classList.add('active');
        statusEl.textContent = 'Running';
        
        // Clear any existing interval
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        // Update timer every second
        this.timerInterval = setInterval(() => this.updateTimer(), 1000);
        
        // Initial update
        this.updateTimer();
    }
    
    stopTimer() {
        // Stop the interval
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // Calculate final elapsed time
        if (this.timerStartTime) {
            this.timerElapsed = Date.now() - this.timerStartTime;
        }
        
        // Update UI
        const timerEl = document.getElementById('learning-timer');
        const statusEl = document.getElementById('timer-status');
        
        timerEl.classList.remove('active');
        statusEl.textContent = 'Stopped';
    }
    
    updateTimer() {
        if (!this.timerStartTime) return;
        
        const elapsed = Date.now() - this.timerStartTime;
        const timerValueEl = document.getElementById('timer-value');
        
        timerValueEl.textContent = this.formatTime(elapsed);
    }
    
    formatTime(ms) {
        const totalSeconds = Math.floor(ms / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        return [
            hours.toString().padStart(2, '0'),
            minutes.toString().padStart(2, '0'),
            seconds.toString().padStart(2, '0')
        ].join(':');
    }
    
    resetTimer() {
        this.timerElapsed = 0;
        this.timerStartTime = null;
        
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        document.getElementById('timer-value').textContent = '00:00:00';
        document.getElementById('timer-status').textContent = 'Stopped';
        document.getElementById('learning-timer').classList.remove('active');
    }
    
    // ========== END TIMER METHODS ==========
    
    async loadStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            this.updateStats(data, false); // Don't reset on initial load
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
    
    updateStats(data, allowReset = true) {
        // Get current values to prevent unwanted resets
        const currentSources = parseInt(document.getElementById('sources-count').textContent.replace(/[^\d]/g, '')) || 0;
        
        if (data.memory) {
            document.getElementById('vocab-count').textContent = this.formatNum(data.memory.vocabulary_size || 0);
            document.getElementById('vocab-stat').textContent = this.formatNum(data.memory.vocabulary_size || 0);
            document.getElementById('knowledge-count').textContent = this.formatNum(data.memory.knowledge_count || 0);
            document.getElementById('knowledge-stat').textContent = this.formatNum(data.memory.knowledge_count || 0);
            
            // Only update sources if we have a valid value > 0
            const newSources = data.memory.sources_learned || 0;
            if (newSources > 0) {
                document.getElementById('sources-count').textContent = this.formatNum(newSources);
                document.getElementById('sources-stat').textContent = this.formatNum(newSources);
                document.getElementById('progress-text').textContent = `${newSources} sources`;
            }
        }
        
        if (data.model) {
            document.getElementById('tokens-count').textContent = this.formatNum(data.model.total_tokens_learned || 0);
            document.getElementById('steps-count').textContent = this.formatNum(data.model.training_steps || 0);
        }
        
        if (data.learner) {
            // FIXED: Always prefer memory.sources_learned (persistent total)
            // Only use learner.sites_learned if memory count is unavailable
            const totalSources = data.memory?.sources_learned || 0;
            const sessionSources = data.learner.sites_learned || 0;
            const displaySources = totalSources > 0 ? totalSources : sessionSources;
            
            // Always update sources count
            document.getElementById('progress-text').textContent = `${displaySources} sources`;
            
            // Hide progress bar (we show sources count instead)
            document.getElementById('progress-fill').style.width = '0%';
            
            document.getElementById('chunks-stat').textContent = this.formatNum(data.learner.total_chunks || 0);
            
            // Update current content preview
            if (data.learner.current_title) {
                document.getElementById('current-url').textContent = data.learner.current_title;
                document.getElementById('current-url').href = data.learner.current_url || '#';
            }
            
            // FIXED: Only update animation when content TRULY changes
            if (data.learner.current_preview) {
                const previewEl = document.getElementById('content-preview');
                const oldText = previewEl.dataset.text || '';
                const newPreview = data.learner.current_preview;
                
                // Check if content is genuinely new (compare first 100 chars)
                const newStart = newPreview.substring(0, 100);
                const oldStart = oldText.substring(0, 100);
                const isNewContent = newStart !== oldStart;
                
                if (isNewContent) {
                    // Truly new content - reset and start animation
                    previewEl.dataset.wordPos = '0';
                    previewEl.dataset.animationComplete = 'false';
                    previewEl.dataset.text = newPreview;
                    this.showAnimatedReading(newPreview);
                }
                // If same content, don't touch the animation - let it continue/complete naturally
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
            // Show immediate UI feedback
            document.getElementById('current-url').textContent = 'Finding content...';
            const readingHeader = document.querySelector('.currently-reading h4');
            if (readingHeader) {
                readingHeader.innerHTML = '📖 Currently Reading';
            }
            document.getElementById('content-preview').innerHTML = '<span class="reading-word pending">Searching for knowledge sources...</span>';
            
            const res = await fetch('/api/learn/start', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started') {
                this.isLearning = true;
                this.updateLearningUI(true);
                this.startPolling();
                this.startTimer();  // Start learning timer
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
            this.stopWordAnimation();
            this.stopTimer();  // Stop learning timer
            this.toast('Learning stopped', 'info');
            this.loadKnowledgeStats();
            this.loadStatus();  // Refresh stats from server
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
        // Hide progress bar completely (we use sources count instead)
        document.getElementById('progress-fill').style.width = '0%';
        
        // Show source count - only update if we have sources (prevents 0 flicker)
        const sourcesLearned = data.sites_learned || 0;
        if (sourcesLearned > 0) {
            document.getElementById('progress-text').textContent = `${sourcesLearned} sources`;
            document.getElementById('sources-count').textContent = this.formatNum(sourcesLearned);
            document.getElementById('sources-stat').textContent = this.formatNum(sourcesLearned);
        }
        
        // Update all stats
        if (data.model) {
            document.getElementById('vocab-count').textContent = this.formatNum(data.model.vocab_size || 0);
            document.getElementById('vocab-stat').textContent = this.formatNum(data.model.vocab_size || 0);
            document.getElementById('tokens-count').textContent = this.formatNum(data.model.total_tokens_learned || 0);
            document.getElementById('steps-count').textContent = this.formatNum(data.model.training_steps || 0);
        }
        
        document.getElementById('chunks-stat').textContent = this.formatNum(data.total_chunks || 0);
        
        // Update current reading display
        if (data.current_title) {
            const urlEl = document.getElementById('current-url');
            urlEl.textContent = data.current_title;
            urlEl.href = data.current_url || '#';
            
            // Show source domain and type in header
            let domain = '';
            try {
                domain = new URL(data.current_url).hostname.replace('www.', '').replace('en.', '');
            } catch { domain = 'web'; }
            
            // Get source type icon
            const sourceType = data.current_source_type || 'wikipedia';
            const sourceIcons = {
                'wikipedia': '📚',
                'news': '📰',
                'blog': '📝',
                'academic': '🎓',
                'web': '🌐',
                'general': '🌐'
            };
            const icon = sourceIcons[sourceType] || '📖';
            
            const readingHeader = document.querySelector('.currently-reading h4');
            if (readingHeader) {
                readingHeader.innerHTML = `${icon} Reading from <span class="domain-badge ${sourceType}">${domain}</span>`;
            }
        }
        
        // Store and animate text preview - reset position when content changes
        if (data.current_preview) {
            const previewEl = document.getElementById('content-preview');
            const oldText = previewEl.dataset.text || '';
            
            // Detect if content changed - compare first 50 chars (faster) 
            const newStart = data.current_preview.substring(0, 50);
            const oldStart = oldText.substring(0, 50);
            
            if (newStart !== oldStart) {
                // Content changed - reset word position
                previewEl.dataset.wordPos = '0';
                previewEl.dataset.text = data.current_preview;
            } else if (data.current_preview.length !== oldText.length) {
                // Same start but different length - also reset
                previewEl.dataset.wordPos = '0';
                previewEl.dataset.text = data.current_preview;
            }
            
            // Always update text reference
            previewEl.dataset.text = data.current_preview;
            this.showAnimatedReading(data.current_preview);
            this.startWordAnimation();
        }
        
        // Update learned sources list (but not too often)
        if (!this.lastKnowledgeUpdate || Date.now() - this.lastKnowledgeUpdate > 5000) {
            this.loadKnowledgeStats();
            this.lastKnowledgeUpdate = Date.now();
        }
    }
    
    showAnimatedReading(text) {
        const previewEl = document.getElementById('content-preview');
        if (!previewEl || !text) return;
        
        // Split into words
        const allWords = text.split(/\s+/).filter(w => w.length > 0);
        if (allWords.length === 0) return;
        
        // Track current position (increment each call, stored in dataset)
        let currentPos = parseInt(previewEl.dataset.wordPos || '0');
        
        // If we've read all words, stay at the end (don't cycle)
        if (currentPos >= allWords.length) {
            currentPos = allWords.length - 1;
        }
        
        // Show a window of ~40 words around the current position (more space now)
        const windowSize = 40;
        const startIdx = Math.max(0, currentPos - 5); // Show 5 words before current
        const endIdx = Math.min(allWords.length, startIdx + windowSize);
        
        // Get the visible window of words
        const visibleWords = allWords.slice(startIdx, endIdx);
        const highlightIdx = currentPos - startIdx; // Position within visible window
        
        // Build HTML with highlighted word
        const highlighted = visibleWords.map((word, i) => {
            if (i === highlightIdx) {
                return `<span class="reading-word current">${this.escapeHtml(word)}</span>`;
            } else if (i < highlightIdx) {
                return `<span class="reading-word read">${this.escapeHtml(word)}</span>`;
            } else {
                return `<span class="reading-word pending">${this.escapeHtml(word)}</span>`;
            }
        }).join(' ');
        
        // Add ellipsis indicators
        const prefix = startIdx > 0 ? '... ' : '';
        const suffix = endIdx < allWords.length ? ' ...' : '';
        
        previewEl.innerHTML = prefix + highlighted + suffix;
        
        // Advance position for next call
        // FIXED: Only advance if not at end, mark as complete when done
        if (currentPos < allWords.length - 1) {
            currentPos++;
            previewEl.dataset.animationComplete = 'false';
        } else {
            // Mark animation as complete to prevent vibration
            previewEl.dataset.animationComplete = 'true';
        }
        previewEl.dataset.wordPos = currentPos;
    }
    
    // Start word animation when learning or searching
    startWordAnimation() {
        if (this.wordAnimationInterval) return;
        this.wordAnimationInterval = setInterval(() => {
            const previewEl = document.getElementById('content-preview');
            // FIXED: Also animate during manual search (not just continuous learning)
            if (previewEl && (this.isLearning || this.isSearching) && previewEl.dataset.text) {
                // FIXED: Don't animate if already at end (prevents vibration)
                if (previewEl.dataset.animationComplete !== 'true') {
                    this.showAnimatedReading(previewEl.dataset.text);
                }
            }
        }, 120); // Slightly faster for smoother reading feel
    }
    
    stopWordAnimation() {
        if (this.wordAnimationInterval) {
            clearInterval(this.wordAnimationInterval);
            this.wordAnimationInterval = null;
        }
        this.isSearching = false;
    }
    
    updateCurrentContent(data) {
        const urlEl = document.getElementById('current-url');
        urlEl.textContent = data.title || data.url;
        urlEl.href = data.url;
        
        // Show content preview with proper animation sync
        if (data.preview) {
            const previewEl = document.getElementById('content-preview');
            // Reset word position and animation state for new content
            previewEl.dataset.wordPos = '0';
            previewEl.dataset.animationComplete = 'false';
            previewEl.dataset.text = data.preview;
            this.showAnimatedReading(data.preview);
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
        this.stopWordAnimation();
        this.stopTimer();  // Stop learning timer
        this.toast(data.message || '🎉 Learning complete!', 'success');
        this.loadKnowledgeStats();
        this.loadStatus();  // Refresh stats from server
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
        // Set search state for animation
        this.isSearching = true;
        
        // Show search progress UI
        const searchId = this.showSearchProgress(query);
        
        // Update the right panel to show we're searching
        const urlEl = document.getElementById('current-url');
        const previewEl = document.getElementById('content-preview');
        const readingHeader = document.querySelector('.currently-reading h4');
        
        if (readingHeader) {
            readingHeader.innerHTML = '🔍 Searching...';
        }
        urlEl.textContent = `Searching for: ${query}`;
        urlEl.href = '#';
        previewEl.innerHTML = '<span class="reading-word pending">Finding relevant sources...</span>';
        
        // Start word animation for search
        this.startWordAnimation();
        
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
            
            // Update reading header
            if (readingHeader) {
                readingHeader.innerHTML = '📖 Currently Reading';
            }
            
            if (data.sources_count > 0 || data.learned_sources?.length > 0) {
                const sources = data.learned_sources || data.sources || [];
                
                // Show each source being learned with reading animation
                for (let i = 0; i < sources.length && i < 3; i++) {
                    const source = sources[i];
                    await this.sleep(200);
                    
                    // Extract domain for display
                    let domain = '';
                    try {
                        domain = new URL(source.url).hostname.replace('www.', '');
                    } catch { domain = 'web'; }
                    
                    // Update right panel with source info and animate reading
                    urlEl.textContent = source.title || domain;
                    urlEl.href = source.url || '#';
                    
                    // Show reading animation for the snippet/content
                    const content = source.snippet || source.content || `Learning about ${query} from ${domain}...`;
                    previewEl.dataset.wordPos = '0';
                    previewEl.dataset.animationComplete = 'false';
                    previewEl.dataset.text = content;
                    this.showAnimatedReading(content);
                    
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
            
            // Stop reading animation and show completion
            this.isSearching = false;
            this.stopWordAnimation();
            if (readingHeader) {
                readingHeader.innerHTML = '✅ Done Reading';
            }
            previewEl.innerHTML = '<span class="reading-word read">Search complete - knowledge acquired!</span>';
            
            this.addAIMessage(data);
            
            // Update all stats immediately
            this.loadStatus();
            this.loadKnowledgeStats();
            
        } catch (e) {
            this.isSearching = false;
            this.removeSearchProgress(searchId);
            this.stopWordAnimation();
            
            // Reset right panel
            const readingHeader = document.querySelector('.currently-reading h4');
            if (readingHeader) {
                readingHeader.innerHTML = '❌ Search Failed';
            }
            document.getElementById('content-preview').innerHTML = '<span class="reading-word pending">Search error occurred</span>';
            
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