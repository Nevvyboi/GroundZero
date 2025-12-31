# 🧠 NeuralMind: Learn Machine Learning from Scratch

> A complete beginner's guide to understanding AI, neural networks, and building your own learning system.

---

## 📚 Table of Contents

1. [Introduction: What is Machine Learning?](#1-introduction-what-is-machine-learning)
2. [The Building Blocks: How Computers "Learn"](#2-the-building-blocks-how-computers-learn)
3. [Neural Networks: Inspired by the Brain](#3-neural-networks-inspired-by-the-brain)
4. [Transformers: The Architecture Behind ChatGPT](#4-transformers-the-architecture-behind-chatgpt)
5. [Understanding NeuralMind's Code](#5-understanding-neualminds-code)
6. [The Reasoning Engine: Logic, Math & Code](#6-the-reasoning-engine-logic-math--code)
7. [Hands-On Exercises](#7-hands-on-exercises)
8. [Glossary of Terms](#8-glossary-of-terms)
9. [Next Steps & Resources](#9-next-steps--resources)

---

## 1. Introduction: What is Machine Learning?

### The Simple Explanation

Imagine teaching a child to recognize cats. You don't give them a rulebook saying "cats have pointy ears, whiskers, and four legs." Instead, you show them hundreds of pictures of cats, and eventually, they just *know* what a cat looks like.

**Machine Learning (ML)** works the same way. Instead of programming explicit rules, we show the computer many examples and let it figure out the patterns itself.

### Three Types of Machine Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING TYPES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SUPERVISED LEARNING                                          │
│     • You provide: Data + Correct Answers                        │
│     • Computer learns: The relationship between them             │
│     • Example: "This email is spam" / "This email is not spam"   │
│                                                                  │
│  2. UNSUPERVISED LEARNING                                        │
│     • You provide: Data only (no answers)                        │
│     • Computer learns: Hidden patterns and groups                │
│     • Example: Grouping customers by shopping behavior           │
│                                                                  │
│  3. REINFORCEMENT LEARNING                                       │
│     • You provide: Rules of a game + Rewards/Penalties           │
│     • Computer learns: Best strategy through trial and error     │
│     • Example: AI learning to play chess                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What About AI?

**Artificial Intelligence (AI)** is the broader goal of making machines "smart." Machine Learning is one way to achieve AI. Think of it like this:

```
AI (Artificial Intelligence)
 └── Machine Learning
      └── Deep Learning (Neural Networks)
           └── Transformers (like GPT, BERT)
                └── Large Language Models (ChatGPT, Claude)
```

NeuralMind uses **Deep Learning** concepts to build a simple AI that can learn and reason.

---

## 2. The Building Blocks: How Computers "Learn"

### 2.1 Data: The Fuel of ML

Everything starts with **data**. In NeuralMind, our data is text from the internet (Wikipedia articles).

```python
# Example: Raw text data
text = "Python is a programming language created by Guido van Rossum."
```

But computers don't understand words—they only understand numbers. So we need to convert text to numbers.

### 2.2 Tokenization: Breaking Text into Pieces

**Tokenization** splits text into smaller units called **tokens**.

```
"Hello world" → ["Hello", "world"]
"I'm learning" → ["I", "'m", "learning"]
```

In NeuralMind's code (`model.py`):

```python
class Tokenizer:
    def tokenize(self, text):
        # Convert to lowercase and split into words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
```

**Why tokenize?**
- Makes text manageable
- Creates a "vocabulary" (list of known words)
- Each word gets a unique number (ID)

```
Vocabulary:
  "python" → 0
  "is" → 1  
  "a" → 2
  "programming" → 3
  "language" → 4
```

### 2.3 Embeddings: Giving Words Meaning

A word's ID (like "python" = 0) doesn't capture its *meaning*. The word "python" is related to "programming" and "code," but the numbers 0 and 3 don't show that.

**Embeddings** solve this by representing each word as a list of numbers (a **vector**) where similar words have similar vectors.

```
"python"      → [0.8, 0.2, 0.9, 0.1, ...]  (256 numbers)
"programming" → [0.7, 0.3, 0.8, 0.2, ...]  (similar!)
"banana"      → [0.1, 0.9, 0.1, 0.8, ...]  (very different)
```

In NeuralMind:

```python
# Create random embeddings (256 dimensions per word)
self.embeddings = np.random.randn(vocab_size, d_model) * 0.02
```

**Key Insight:** Words that appear in similar contexts end up with similar embeddings. "King" and "queen" become close in this number-space because they appear in similar sentences.

### 2.4 The Learning Process

How does the computer actually "learn"? Through these steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE LEARNING LOOP                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. FORWARD PASS                                                │
│      • Feed data through the network                             │
│      • Get a prediction                                          │
│                                                                  │
│   2. CALCULATE ERROR (LOSS)                                      │
│      • Compare prediction to correct answer                      │
│      • "How wrong were we?"                                      │
│                                                                  │
│   3. BACKWARD PASS (BACKPROPAGATION)                            │
│      • Figure out which parts of the network caused the error    │
│      • Calculate gradients (directions to improve)               │
│                                                                  │
│   4. UPDATE WEIGHTS                                              │
│      • Adjust the numbers in the network slightly                │
│      • Move in the direction that reduces error                  │
│                                                                  │
│   5. REPEAT                                                      │
│      • Do this millions of times with lots of data               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** NeuralMind uses a simplified learning approach (updating embeddings directly) rather than full backpropagation, making it easier to understand while still demonstrating the core concepts.

---

## 3. Neural Networks: Inspired by the Brain

### 3.1 What is a Neural Network?

Your brain has ~86 billion **neurons** connected by **synapses**. When you think, electrical signals pass through networks of neurons.

A **neural network** is a simplified computer version:

```
        INPUT LAYER          HIDDEN LAYER         OUTPUT LAYER
        (Your Data)         (Processing)          (Prediction)
        
           ○                    ○
            \                  /|\
           ○──────────────────○─┼─────────────────○
            \                / \|/                /
           ○────────────────○───○───────────────○
            \              / \ /|\              /
           ○──────────────○───○───────────────○
                          (neurons)
```

### 3.2 How a Single Neuron Works

Each neuron does a simple calculation:

```
         INPUTS              WEIGHTS              OUTPUT
         
         x₁ ──── w₁ ───┐
                       │
         x₂ ──── w₂ ───┼──→ SUM ──→ ACTIVATION ──→ output
                       │
         x₃ ──── w₃ ───┘
         
    
    output = activation(x₁*w₁ + x₂*w₂ + x₃*w₃ + bias)
```

**In plain English:**
1. Take inputs (numbers)
2. Multiply each by a **weight** (importance)
3. Add them up
4. Apply an **activation function** (adds non-linearity)
5. Output a number

### 3.3 Activation Functions

Why do we need activation functions? Without them, no matter how many layers we stack, the network can only learn simple linear patterns (straight lines).

**Common activation functions:**

```python
# ReLU (Rectified Linear Unit) - Most popular
def relu(x):
    return max(0, x)  # If negative, output 0; otherwise, output x

# Sigmoid - Squashes output between 0 and 1
def sigmoid(x):
    return 1 / (1 + exp(-x))

# Softmax - Converts numbers to probabilities (sum to 1)
def softmax(x):
    exp_x = exp(x)
    return exp_x / sum(exp_x)
```

In NeuralMind, we use ReLU in our transformer blocks:

```python
def feed_forward(self, x):
    # First layer with ReLU activation
    hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
    # Second layer
    return hidden @ self.W2 + self.b2
```

### 3.4 Layers: Stacking Neurons

A **layer** is a group of neurons that process data together:

- **Input Layer:** Receives your raw data
- **Hidden Layers:** Do the actual learning (more layers = can learn more complex patterns)
- **Output Layer:** Produces the final prediction

```python
# NeuralMind has 6 transformer layers
self.n_layers = 6

# Each layer processes the data and passes it to the next
for layer in self.layers:
    x = layer.forward(x)
```

**Deep Learning** = Neural networks with many hidden layers (hence "deep").

---

## 4. Transformers: The Architecture Behind ChatGPT

### 4.1 The Attention Revolution

Before 2017, most language AI used **Recurrent Neural Networks (RNNs)** that read text one word at a time, left to right. This was slow and struggled with long texts.

Then came the paper **"Attention Is All You Need"** which introduced **Transformers**. The key insight: let the model look at ALL words at once and figure out which ones are important.

### 4.2 What is Attention?

Imagine reading this sentence:
> "The cat sat on the mat because it was tired."

What does "it" refer to? You instantly know it's "the cat" because your brain pays **attention** to the relevant words.

**Self-attention** lets the AI do the same thing:

```
Query: What does "it" refer to?

         The    cat    sat    on    the    mat    because    it    was    tired
Score:   0.1    0.8    0.05   0.01  0.01   0.02   0.01       -     0.05   0.05
                 ↑
            Highest attention!
```

### 4.3 How Attention Works (Simplified)

For each word, we create three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**Step by step:**

1. **Q × K^T:** Compare the query with all keys (dot product)
2. **/ √d:** Scale down to prevent huge numbers
3. **softmax:** Convert to probabilities (0-1, sum to 1)
4. **× V:** Weighted sum of values

In NeuralMind:

```python
class AttentionHead:
    def forward(self, x):
        # Create Q, K, V
        Q = x @ self.W_q  # Query
        K = x @ self.W_k  # Key  
        V = x @ self.W_v  # Value
        
        # Calculate attention scores
        scores = Q @ K.T / np.sqrt(self.head_dim)
        
        # Softmax to get probabilities
        attention_weights = self.softmax(scores)
        
        # Weighted sum of values
        return attention_weights @ V
```

### 4.4 Multi-Head Attention

One attention head might focus on grammar, another on meaning, another on context. **Multi-head attention** runs several attention operations in parallel:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input ──┬──→ Head 1 (grammar) ──┐                             │
│           │                        │                             │
│           ├──→ Head 2 (meaning) ──┼──→ Concatenate ──→ Output   │
│           │                        │                             │
│           ├──→ Head 3 (context) ──┤                             │
│           │                        │                             │
│           └──→ Head 4 (other) ────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

NeuralMind uses 8 attention heads:

```python
self.n_heads = 8

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.heads = [AttentionHead(d_model, d_model // n_heads) 
                      for _ in range(n_heads)]
    
    def forward(self, x):
        # Run all heads and combine results
        head_outputs = [head.forward(x) for head in self.heads]
        return np.concatenate(head_outputs, axis=-1)
```

### 4.5 The Transformer Block

A complete transformer block combines attention with a feed-forward network:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input                                                          │
│     │                                                            │
│     ▼                                                            │
│   ┌─────────────────────┐                                       │
│   │  Multi-Head         │                                       │
│   │  Attention          │◄── "What should I focus on?"          │
│   └─────────────────────┘                                       │
│     │                                                            │
│     ▼                                                            │
│   ┌─────────────────────┐                                       │
│   │  Add & Normalize    │◄── Residual connection + LayerNorm    │
│   └─────────────────────┘                                       │
│     │                                                            │
│     ▼                                                            │
│   ┌─────────────────────┐                                       │
│   │  Feed-Forward       │◄── Process each position              │
│   │  Network            │    independently                       │
│   └─────────────────────┘                                       │
│     │                                                            │
│     ▼                                                            │
│   ┌─────────────────────┐                                       │
│   │  Add & Normalize    │                                       │
│   └─────────────────────┘                                       │
│     │                                                            │
│     ▼                                                            │
│   Output                                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key components:**

1. **Residual Connections (Add):** Add the input to the output. This helps with training deep networks by allowing gradients to flow directly.

2. **Layer Normalization:** Keeps numbers in a reasonable range, making training stable.

```python
class TransformerBlock:
    def forward(self, x):
        # Attention with residual connection
        attention_out = self.attention.forward(x)
        x = self.norm1.forward(x + attention_out)  # Add & Norm
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2.forward(x + ff_out)  # Add & Norm
        
        return x
```

### 4.6 Putting It All Together

NeuralMind's architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURALMIND ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "What is Python?"                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │  Tokenizer  │  "what", "is", "python"                       │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │  Embedding  │  [0.2, 0.8, ...], [0.1, 0.3, ...], ...       │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │ Transformer │  × 6 layers                                   │
│   │   Block     │  (8 attention heads, 1024 FFN dim)            │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │  Knowledge  │  Search for relevant stored information       │
│   │   Memory    │                                               │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   "Python is a programming language..."                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Understanding NeuralMind's Code

Let's walk through the actual code, file by file.

### 5.1 model.py - The Neural Network

#### The Tokenizer Class

```python
class Tokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}  # "python" → 42
        self.id_to_word = {}  # 42 → "python"
        self.word_counts = {} # How often each word appears
        self.next_id = 0
    
    def tokenize(self, text):
        """Split text into words"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def encode(self, text):
        """Convert text to numbers"""
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
        return ids
    
    def decode(self, ids):
        """Convert numbers back to text"""
        return ' '.join(self.id_to_word.get(id, '<unk>') for id in ids)
```

**What's happening:**
- `word_to_id`: Dictionary mapping words to unique numbers
- `encode`: Turns "hello world" into [0, 1]
- `decode`: Turns [0, 1] back into "hello world"

#### The Knowledge Memory

```python
class KnowledgeMemory:
    def __init__(self, max_entries=100000):
        self.memories = []      # Stored knowledge
        self.embeddings = []    # Vector representations
        self.max_entries = max_entries
    
    def store(self, content, embedding, source=""):
        """Save a piece of knowledge"""
        self.memories.append({
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        self.embeddings.append(embedding)
    
    def retrieve(self, query_embedding, top_k=5):
        """Find relevant memories"""
        similarities = []
        for i, mem_emb in enumerate(self.embeddings):
            # Cosine similarity: how similar are two vectors?
            sim = self.cosine_similarity(query_embedding, mem_emb)
            similarities.append((i, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.memories[i] for i, _ in similarities[:top_k]]
    
    def cosine_similarity(self, a, b):
        """Measure similarity between two vectors (0 to 1)"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
```

**What's happening:**
- Store knowledge as text + its vector representation
- When queried, find the most similar stored knowledge
- **Cosine similarity** measures angle between vectors (1 = identical direction, 0 = perpendicular)

#### The Main NeuralMind Class

```python
class NeuralMind:
    def __init__(self, vocab_size=50000, d_model=256, n_heads=8, 
                 n_layers=6, d_ff=1024, max_seq_len=512):
        
        # Model dimensions
        self.d_model = d_model      # Size of embeddings (256)
        self.n_heads = n_heads      # Attention heads (8)
        self.n_layers = n_layers    # Transformer layers (6)
        self.d_ff = d_ff            # Feed-forward size (1024)
        
        # Initialize components
        self.tokenizer = Tokenizer(vocab_size)
        self.memory = KnowledgeMemory()
        self.reasoning = ReasoningEngine()
        
        # Create embeddings (random to start)
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.02
        
        # Create transformer layers
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ]
```

**Key parameters:**
- `d_model=256`: Each word is represented by 256 numbers
- `n_heads=8`: 8 different attention perspectives
- `n_layers=6`: 6 transformer blocks stacked
- `d_ff=1024`: Hidden size of feed-forward network

### 5.2 reasoning.py - Logic, Math & Code Analysis

#### The Logic Engine

```python
class LogicEngine:
    def parse_relations(self, text):
        """Extract relations like 'A > B' from text"""
        # Pattern: A > B, X < Y, etc.
        pattern = r'(\w+)\s*(>|<|>=|<=|==|!=|=)\s*(\w+)'
        matches = re.findall(pattern, text)
        return matches  # [('A', '>', 'B'), ('B', '>', 'C')]
    
    def find_transitive_relation(self, graph, a, b, op):
        """If A > B and B > C, then A > C"""
        # Use graph traversal to find path from A to C
        # This is transitive closure
        ...
```

**What's transitive reasoning?**
```
Given: A > B, B > C
Question: What's the relationship between A and C?

Step 1: A > B (A is greater than B)
Step 2: B > C (B is greater than C)  
Step 3: Therefore, A > C (A is greater than C)

The chain: A > B > C
```

#### The Math Engine

```python
class MathEngine:
    def __init__(self):
        self.functions = {
            'sin': math.sin, 'cos': math.cos,
            'sqrt': math.sqrt, 'log': math.log,
            'factorial': math.factorial, ...
        }
    
    def evaluate_expression(self, expr):
        """Solve math step by step"""
        # Handle parentheses first
        while '(' in expr:
            # Find innermost parentheses
            # Evaluate and replace
            ...
        
        # Then exponents, multiplication/division, addition/subtraction
        # (Order of operations: PEMDAS)
```

**Example step-by-step:**
```
Input: (5 + 3) * 2 - 4 / 2

Step 1: Evaluate (5 + 3) = 8
        Expression: 8 * 2 - 4 / 2

Step 2: Evaluate 8 * 2 = 16  
        Expression: 16 - 4 / 2

Step 3: Evaluate 4 / 2 = 2
        Expression: 16 - 2

Step 4: Evaluate 16 - 2 = 14

Answer: 14
```

#### The Code Analyzer

```python
class CodeAnalyzer:
    def analyze_python(self, code):
        """Check Python code for errors"""
        try:
            # Try to parse as Python
            ast.parse(code)
            return []  # No syntax errors
        except SyntaxError as e:
            return [{
                'type': 'SyntaxError',
                'line': e.lineno,
                'message': str(e.msg),
                'suggestion': 'Check for missing colons, parentheses...'
            }]
```

**How it works:**
- Python's `ast` module can parse code without running it
- If parsing fails, we get a SyntaxError with line number
- We provide helpful suggestions based on error type

### 5.3 web_learner.py - Learning from the Internet

```python
class WebLearner:
    def __init__(self, model, callback=None):
        self.model = model
        self.visited_urls = set()
        self.url_queue = deque()
        
        # Start with Wikipedia articles
        self.seed_urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            ...
        ]
    
    def scrape_page(self, url):
        """Download and extract text from a webpage"""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        
        return self.clean_text(text)
    
    def learn_from_url(self, url):
        """Scrape a page and teach the model"""
        text = self.scrape_page(url)
        stats = self.model.learn_from_text(text, source=url)
        return stats
```

**The learning loop:**
1. Start with seed URLs (Wikipedia articles)
2. Download page content
3. Clean the text (remove HTML, scripts, etc.)
4. Feed to model's `learn_from_text` method
5. Find more links, add to queue
6. Repeat

---

## 6. The Reasoning Engine: Logic, Math & Code

### 6.1 Why Reasoning Matters

Pure neural networks struggle with:
- **Logic:** "If A > B and B > C, what about A and C?"
- **Math:** "What is 15 * 7 + 23?"
- **Code:** "Why doesn't this code work?"

NeuralMind adds a **symbolic reasoning engine** that handles these tasks with explicit rules, not just pattern matching.

### 6.2 Logic Reasoning

```python
# Input: "If A > B and B > C, what's the relationship between A and C?"

# Step 1: Parse relations
relations = [('A', '>', 'B'), ('B', '>', 'C')]

# Step 2: Build a graph
graph = {
    '>': {'A': ['B'], 'B': ['C']},  # A points to B, B points to C
    '<': {'B': ['A'], 'C': ['B']}   # Reverse edges
}

# Step 3: Find path from A to C using '>'
path = find_path(graph, 'A', 'C', '>')  # Returns ['A', 'B', 'C']

# Step 4: Answer
"A > C because A > B > C"
```

### 6.3 Math Solving

The math engine follows **order of operations (PEMDAS)**:

```
P - Parentheses first
E - Exponents (powers)
M/D - Multiplication and Division (left to right)
A/S - Addition and Subtraction (left to right)
```

```python
# Solving: 2^3 + sqrt(16) * 2

# Step 1: Functions first
#   sqrt(16) = 4
#   Expression: 2^3 + 4 * 2

# Step 2: Exponents
#   2^3 = 8
#   Expression: 8 + 4 * 2

# Step 3: Multiplication
#   4 * 2 = 8
#   Expression: 8 + 8

# Step 4: Addition
#   8 + 8 = 16

# Answer: 16
```

### 6.4 Code Analysis

```python
# Input code with bug:
def hello()
    print("world")

# Analysis:
# 1. Try to parse with Python's ast module
# 2. Get SyntaxError: "expected ':'" at line 1
# 3. Suggest: "Add a colon after the function definition"

# Fixed code:
def hello():
    print("world")
```

---

## 7. Hands-On Exercises

### Exercise 1: Understand Tokenization

```python
# Try this code:
text = "Machine learning is fascinating!"

# Manual tokenization
tokens = text.lower().split()
print(tokens)  # ['machine', 'learning', 'is', 'fascinating!']

# Better tokenization (remove punctuation)
import re
tokens = re.findall(r'\b\w+\b', text.lower())
print(tokens)  # ['machine', 'learning', 'is', 'fascinating']
```

**Your task:** Tokenize "I'm learning AI in 2024!" and see what happens with contractions and numbers.

### Exercise 2: Visualize Embeddings

```python
import numpy as np

# Create simple embeddings
words = ["king", "queen", "man", "woman", "apple", "orange"]
embeddings = {
    "king":   np.array([0.9, 0.1, 0.8]),
    "queen":  np.array([0.9, 0.9, 0.8]),
    "man":    np.array([0.8, 0.1, 0.5]),
    "woman":  np.array([0.8, 0.9, 0.5]),
    "apple":  np.array([0.1, 0.5, 0.1]),
    "orange": np.array([0.1, 0.6, 0.1]),
}

# Calculate similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"king-queen: {cosine_similarity(embeddings['king'], embeddings['queen']):.3f}")
print(f"king-apple: {cosine_similarity(embeddings['king'], embeddings['apple']):.3f}")
```

**Your task:** Add more words and see which ones are most similar.

### Exercise 3: Build a Simple Attention

```python
import numpy as np

def simple_attention(query, keys, values):
    """
    query: what we're looking for (1D array)
    keys: what each position contains (2D array)
    values: information at each position (2D array)
    """
    # Calculate scores
    scores = np.dot(keys, query)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores))
    attention_weights = exp_scores / np.sum(exp_scores)
    
    # Weighted sum
    output = np.dot(attention_weights, values)
    
    return output, attention_weights

# Example
query = np.array([1, 0, 0])  # Looking for "subject"
keys = np.array([
    [0.9, 0.1, 0.1],  # "The" 
    [0.8, 0.2, 0.9],  # "cat" (subject!)
    [0.1, 0.8, 0.1],  # "sat"
])
values = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

output, weights = simple_attention(query, keys, values)
print(f"Attention weights: {weights}")
print(f"Output: {output}")
```

**Your task:** Modify the query to focus on different words.

### Exercise 4: Run NeuralMind

```bash
# 1. Install dependencies
pip install flask flask-socketio requests beautifulsoup4 numpy

# 2. Start the server
python run.py

# 3. Open browser to http://localhost:5000

# 4. Try these queries:
#    - "If A > B and B > C, what's A vs C?"
#    - "Calculate: (10 + 5) * 2"
#    - "Debug: def hello() print('hi')"
```

---

## 8. Glossary of Terms

| Term | Definition |
|------|------------|
| **Activation Function** | A function that adds non-linearity (e.g., ReLU, sigmoid) |
| **Attention** | Mechanism that lets models focus on relevant parts of input |
| **Backpropagation** | Algorithm for calculating gradients to update weights |
| **Batch** | A group of examples processed together |
| **Embedding** | Dense vector representation of words or tokens |
| **Epoch** | One complete pass through the training data |
| **Gradient** | Direction and magnitude of change for a parameter |
| **Hidden Layer** | Layers between input and output in a neural network |
| **Inference** | Using a trained model to make predictions |
| **Layer Normalization** | Technique to stabilize training |
| **Learning Rate** | How big of a step to take when updating weights |
| **Loss Function** | Measures how wrong predictions are |
| **Multi-Head Attention** | Multiple attention mechanisms in parallel |
| **Neural Network** | Computation model inspired by biological neurons |
| **Overfitting** | When a model memorizes training data but fails on new data |
| **Parameter** | A learnable value in the model (weights, biases) |
| **Query/Key/Value** | Components of attention mechanism |
| **Residual Connection** | Shortcut that adds input to output |
| **Softmax** | Converts numbers to probabilities summing to 1 |
| **Token** | A unit of text (word, subword, or character) |
| **Transformer** | Architecture using self-attention |
| **Vector** | A list of numbers representing something |
| **Weight** | Learnable parameter that scales inputs |

---

## Congratulations! 🎉

You've learned:
- ✅ What machine learning is
- ✅ How neural networks work
- ✅ What attention and transformers do
- ✅ How to read ML code
- ✅ How NeuralMind combines learning and reasoning

Keep experimenting, break things, and rebuild them. That's how you truly learn!

---

*Built with ❤️ for learning. Questions? Experiment with the code!*
