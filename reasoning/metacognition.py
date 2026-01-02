"""
Metacognition
=============
Self-awareness, introspection, and capability assessment.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class Metacognition:
    """
    Handles self-awareness and introspection.
    Answers questions about the AI's own knowledge, capabilities, and state.
    """
    
    def __init__(self, memory_store=None, neural_model=None):
        self.memory = memory_store
        self.model = neural_model
    
    def set_components(self, memory_store=None, neural_model=None) -> None:
        """Set or update component references"""
        if memory_store:
            self.memory = memory_store
        if neural_model:
            self.model = neural_model
    
    def reflect(self, query: str) -> str:
        """
        Generate self-reflective response based on actual internal state.
        Responses are grounded in real data, not generic claims.
        """
        query_lower = query.lower()
        
        # Identity questions
        if any(t in query_lower for t in ['who are you', 'what are you', 'about yourself', 'your name']):
            return self._describe_identity()
        
        # Capability questions
        if any(t in query_lower for t in ['what can you do', 'capabilities', 'your abilities']):
            return self._describe_capabilities()
        
        # Knowledge questions
        if any(t in query_lower for t in ['what do you know', 'what have you learned', 'your knowledge']):
            return self._describe_knowledge()
        
        # Topic expertise questions
        if any(t in query_lower for t in ['what topics', 'understand well', 'confident about', 'expertise']):
            return self._describe_expertise()
        
        # Recent learning questions
        if any(t in query_lower for t in ['learned recently', 'new things', 'latest']):
            return self._describe_recent_learning()
        
        # Limitation questions
        if any(t in query_lower for t in ["don't know", 'limitations', "can't do", 'weaknesses']):
            return self._describe_limitations()
        
        # Confidence questions
        if any(t in query_lower for t in ['how confident', 'how sure', 'certainty']):
            return self._describe_confidence()
        
        # Growth/progress questions
        if any(t in query_lower for t in ['how much', 'progress', 'growth', 'improved']):
            return self._describe_progress()
        
        # Default: general self-description
        return self._describe_identity()
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get current statistics from memory and model"""
        stats = {
            'vocab_size': 4,
            'knowledge_count': 0,
            'sources_learned': 0,
            'concepts_count': 0,
            'total_tokens': 0,
            'training_steps': 0,
            'avg_confidence': 0.0
        }
        
        if self.memory:
            mem_stats = self.memory.get_statistics()
            stats.update(mem_stats)
        
        if self.model:
            model_stats = self.model.get_stats()
            stats['vocab_size'] = model_stats.get('vocab_size', stats['vocab_size'])
            stats['total_tokens'] = model_stats.get('total_tokens_learned', 0)
            stats['training_steps'] = model_stats.get('training_steps', 0)
        
        return stats
    
    def _describe_identity(self) -> str:
        """Describe what the AI is"""
        stats = self._get_stats()
        
        return f"""I'm NeuralMind, an AI that learns from scratch through continuous knowledge acquisition.

**What makes me different:**
Unlike pre-trained models, I start with zero knowledge and build understanding by:
• Reading and processing web content
• Learning from conversations with you
• Building relationships between concepts
• Continuously updating my neural representations

**My Current State:**
• Vocabulary: {stats['vocab_size']:,} words
• Knowledge entries: {stats['knowledge_count']:,}
• Sources learned: {stats['sources_learned']:,}
• Tokens processed: {stats['total_tokens']:,}
• Training steps: {stats['training_steps']:,}

I'm designed to grow smarter with every interaction. What would you like to explore?"""
    
    def _describe_capabilities(self) -> str:
        """Describe what the AI can do"""
        stats = self._get_stats()
        
        capabilities = """**My Capabilities:**

🧠 **Learning**
• Continuously acquire knowledge from the web
• Learn from direct teaching and conversations
• Build and strengthen concept relationships

💬 **Conversation**
• Answer questions based on learned knowledge
• Ask clarifying questions when uncertain
• Explain my reasoning process

🔢 **Mathematical Reasoning**
• Solve arithmetic and algebraic expressions
• Compute percentages, averages, and more
• Show step-by-step solutions

🔍 **Logical Reasoning**
• Analyze conditional statements
• Perform syllogistic reasoning
• Evaluate Boolean expressions

💻 **Code Analysis**
• Detect syntax errors
• Identify common bugs
• Suggest improvements

📊 **Self-Reflection**
• Report on what I know and don't know
• Assess my confidence in different topics
• Explain how I learned something"""
        
        if stats['knowledge_count'] > 0:
            capabilities += f"\n\n**Current Knowledge Base:** {stats['knowledge_count']:,} entries from {stats['sources_learned']:,} sources"
        else:
            capabilities += "\n\n*Start my learning engine to build my knowledge base!*"
        
        return capabilities
    
    def _describe_knowledge(self) -> str:
        """Describe what the AI knows"""
        stats = self._get_stats()
        
        if stats['knowledge_count'] == 0:
            return """I don't have much knowledge yet - I'm still new!

To help me learn, you can:
1. Start my **Learning Engine** to let me explore Wikipedia
2. **Teach me directly** using the Knowledge Base panel
3. **Ask me to search** for specific topics

What would you like me to learn about?"""
        
        # Get top concepts/topics
        top_concepts = []
        if self.memory:
            top_concepts = self.memory.get_top_concepts(10)
        
        response = f"""**My Knowledge Base:**

📚 **Statistics:**
• {stats['knowledge_count']:,} knowledge entries
• {stats['vocab_size']:,} words in vocabulary
• {stats['sources_learned']:,} sources processed
• {stats['concepts_count']:,} concepts identified"""
        
        if top_concepts:
            concept_list = ", ".join([c['name'] for c in top_concepts[:8]])
            response += f"\n\n**Key Concepts I Know:**\n{concept_list}"
        
        # Get recent sources
        if self.memory:
            recent = self.memory.get_recent_sources(5)
            if recent:
                response += "\n\n**Recently Learned From:**"
                for src in recent[:3]:
                    title = src.get('title', src.get('url', 'Unknown'))[:40]
                    response += f"\n• {title}"
        
        response += "\n\nAsk me about any topic, and I'll tell you what I know!"
        return response
    
    def _describe_expertise(self) -> str:
        """Describe areas of expertise based on actual knowledge"""
        if not self.memory:
            return "I need to learn more before I can assess my expertise areas."
        
        top_concepts = self.memory.get_top_concepts(15)
        top_knowledge = self.memory.get_top_knowledge(10)
        
        if not top_concepts:
            return """I haven't developed strong expertise in any area yet.

Start my learning engine or teach me about topics you're interested in, and I'll build expertise over time!"""
        
        response = "**My Areas of Expertise:**\n\nBased on what I've learned, I'm most knowledgeable about:\n"
        
        for i, concept in enumerate(top_concepts[:10], 1):
            confidence = concept.get('confidence', 0) * 100
            mentions = concept.get('mention_count', 0)
            response += f"\n{i}. **{concept['name'].title()}** - {confidence:.0f}% confident ({mentions} references)"
        
        avg_conf = sum(c.get('confidence', 0) for c in top_concepts) / len(top_concepts) * 100
        response += f"\n\n*Average confidence across topics: {avg_conf:.1f}%*"
        
        return response
    
    def _describe_recent_learning(self) -> str:
        """Describe what was recently learned"""
        if not self.memory:
            return "I don't have recent learning history available."
        
        recent_sources = self.memory.get_recent_sources(10)
        recent_words = self.memory.get_recent_words(15)
        
        if not recent_sources and not recent_words:
            return """I haven't learned anything recently.

Would you like to:
• Start my learning engine to explore new topics
• Teach me something specific
• Ask me to search for information"""
        
        response = "**Recent Learning Activity:**\n"
        
        if recent_sources:
            response += "\n📖 **Recently Read:**"
            for src in recent_sources[:5]:
                title = src.get('title', 'Unknown')[:50]
                chunks = src.get('chunks_learned', 0)
                response += f"\n• {title} ({chunks} chunks)"
        
        if recent_words:
            words = [w['word'] for w in recent_words[:10]]
            response += f"\n\n📝 **New Vocabulary:**\n{', '.join(words)}"
        
        return response
    
    def _describe_limitations(self) -> str:
        """Describe what the AI cannot do or doesn't know"""
        stats = self._get_stats()
        
        return f"""**My Current Limitations:**

🚫 **I cannot:**
• Access real-time information without searching
• Remember previous conversations (each chat is fresh)
• Execute code or access external systems
• Process images or audio
• Access websites that block automated access

⚠️ **I have limited knowledge about:**
• Very recent events (need to search)
• Specialized/technical topics I haven't learned
• Personal or private information

📊 **Current Knowledge Gaps:**
• I've only learned from {stats['sources_learned']:,} sources
• My vocabulary is {stats['vocab_size']:,} words (limited compared to human language)
• Average confidence: {stats.get('avg_knowledge_confidence', 0)*100:.1f}%

When I don't know something, I'll tell you honestly and offer to search for the information!"""
    
    def _describe_confidence(self) -> str:
        """Describe confidence levels"""
        stats = self._get_stats()
        
        avg_conf = stats.get('avg_knowledge_confidence', 0) * 100
        
        if self.memory:
            top_knowledge = self.memory.get_top_knowledge(5)
            high_conf = [k for k in top_knowledge if k.get('confidence', 0) > 0.7]
        else:
            high_conf = []
        
        response = f"""**My Confidence Assessment:**

📊 **Overall:** {avg_conf:.1f}% average confidence across all knowledge

**How I measure confidence:**
• Frequency of topic encounters
• Number of sources confirming information
• How often I've successfully used the knowledge

**High Confidence Areas:** {len(high_conf)} topics above 70% confidence
**Knowledge Entries:** {stats['knowledge_count']:,} total

*Note: I'm most confident about topics I've learned from multiple sources and used in conversations.*"""
        
        return response
    
    def _describe_progress(self) -> str:
        """Describe learning progress"""
        stats = self._get_stats()
        
        return f"""**My Learning Progress:**

📈 **Growth Statistics:**
• Vocabulary: {stats['vocab_size']:,} words learned
• Knowledge: {stats['knowledge_count']:,} entries stored
• Sources: {stats['sources_learned']:,} websites read
• Tokens: {stats['total_tokens']:,} processed
• Training: {stats['training_steps']:,} steps completed

**How I Improve:**
1. Each conversation teaches me context
2. Web learning expands my knowledge base
3. Repeated concepts get stronger neural representations
4. Corrections help me fix misconceptions

Keep chatting with me to help me grow smarter! 🌱"""
