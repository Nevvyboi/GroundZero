# GroundZero → World Model: Upgrade Roadmap

## Executive Summary

**Current State**: GroundZero is a knowledge-based learning AI with neural transformer, knowledge graphs, and text understanding.

**Target State**: A "World Model" that understands physics, spatial relationships, causality, and can predict outcomes of actions in the real world.

**Timeline**: With focused development, GroundZero can begin transitioning to world model capabilities in **6-12 months**, with full capabilities in **2-3 years**.

---

## What is a World Model?

A World Model is an AI that maintains an **internal representation of how the world works** - not just text patterns, but physics, causality, and consequences.

### Core Difference

| Aspect | Current LLM/GroundZero | World Model |
|--------|------------------------|-------------|
| **Knowledge** | "Paris is capital of France" | "If I push a ball, it will roll" |
| **Reasoning** | Pattern matching on text | Physics simulation |
| **Prediction** | Next word/token | Next state of environment |
| **Action** | Generate text | Plan physical movements |
| **Memory** | Context window | Persistent world state |

### The 2018 Foundation (Ha & Schmidhuber)

World Models decompose into three components:

```
┌─────────────────────────────────────────────────────────────┐
│                     WORLD MODEL ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. VISION (V)              2. MEMORY (M)                  │
│   ┌──────────────┐          ┌──────────────────┐            │
│   │ Variational  │          │  Mixture Density │            │
│   │ Autoencoder  │───────────│  Network (MDN)   │            │
│   │  (VAE)       │          │  + RNN/LSTM      │            │
│   └──────────────┘          └──────────────────┘            │
│          │                           │                       │
│          └───────────┬───────────────┘                       │
│                      │                                       │
│              3. CONTROLLER (C)                              │
│              ┌──────────────┐                               │
│              │   Simple     │                               │
│              │   Linear     │                               │
│              │   Network    │                               │
│              └──────────────┘                               │
│                      │                                       │
│                      ▼                                       │
│              [ACTION OUTPUT]                                │
└─────────────────────────────────────────────────────────────┘
```

**V (Vision Model)**: Compresses observations into compact latent codes
**M (Memory Model)**: Predicts future latent codes from past + actions  
**C (Controller)**: Makes decisions based on V and M

---

## Current Industry Leaders (2024-2025)

| Company | Model | Capabilities |
|---------|-------|--------------|
| **Google DeepMind** | Genie 3 | Interactive 3D worlds from single images, physics-consistent |
| **NVIDIA** | Cosmos | World foundation models for robotics/autonomous vehicles |
| **Meta** | V-JEPA 2 | Video-based world model for robot planning, zero-shot control |
| **OpenAI** | Sora 2 | Video generation with physics understanding |
| **World Labs** | (unnamed) | 3D scene generation with spatial intelligence |
| **Wayve** | GAIA-2 | Driving simulation with controllable scenarios |

### Scale of These Models

- **Parameters**: 1B-70B+
- **Training Data**: Petabytes of video (millions of hours)
- **Compute Cost**: $10M-$100M+ per model
- **GPUs**: 1000s of A100/H100s

---

## GroundZero's Path to World Model Status

### Phase 1: Foundation (Current - 6 months)
**Goal**: Build the prerequisite capabilities

#### 1.1 Multimodal Input Processing
```
Current:  Text → Embeddings → Knowledge Base
Add:      Image → Vision Encoder → Visual Embeddings
Add:      Video → Frame Encoder → Temporal Embeddings
Add:      Audio → Speech Encoder → Audio Embeddings (DONE ✓)
```

**Implementation**: Add vision transformer (ViT) for image processing
- Minimum: ~86M parameters (ViT-Base)
- Training: ImageNet or OpenImages dataset
- GPU: Single RTX 3090/4090 sufficient for fine-tuning

#### 1.2 Spatial Understanding Module
```python
class SpatialReasoner:
    """
    Understands spatial relationships:
    - "above", "below", "inside", "next to"
    - Distance estimation
    - 3D scene reconstruction from 2D
    """
    def process_scene(self, image) -> Scene3D:
        depth = self.depth_estimator(image)
        objects = self.object_detector(image)
        relationships = self.relation_extractor(objects, depth)
        return Scene3D(objects, relationships, depth)
```

**Minimum Requirements**:
- GPU: 8GB VRAM for depth estimation (MiDaS)
- GPU: 4GB VRAM for object detection (YOLO)
- Total: Can run on single 12GB GPU

#### 1.3 Causal Reasoning Enhancement
Upgrade knowledge graph with causal links:
```
Current:  Paris --[capital_of]--> France
Add:      Push --[causes]--> Move
Add:      Heat --[causes]--> Melt
Add:      Drop --[causes]--> Fall
```

**Already have foundation**: `persistent_graph.py` with inference rules

---

### Phase 2: Dynamics Learning (6-12 months)
**Goal**: Learn how things change over time

#### 2.1 Video Prediction Module
```
┌─────────────────────────────────────────────────────────────┐
│                   VIDEO PREDICTION NETWORK                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Frame t-2    Frame t-1    Frame t       Frame t+1 (pred)  │
│      ▼            ▼            ▼               ▲             │
│   ┌────┐       ┌────┐       ┌────┐         ┌────┐          │
│   │Enc │──────→│Enc │──────→│Enc │────────→│Dec │          │
│   └────┘       └────┘       └────┘         └────┘          │
│                                                              │
│   Latent space: z_t-2 → z_t-1 → z_t → z_t+1 (predicted)    │
└─────────────────────────────────────────────────────────────┘
```

**Architecture Options**:
- **SimVP**: Simple Video Prediction (25M params, trainable on single GPU)
- **Video Transformer**: Larger but more capable (100M-1B params)
- **Diffusion Models**: Best quality but computationally expensive

**Minimum viable**:
- Model: SimVP or similar (~25-100M parameters)
- Training: 100K-1M short video clips
- GPU: 24GB VRAM (RTX 3090/4090)
- Time: 1-2 weeks on single GPU

#### 2.2 Physics Encoder
Learn implicit physics from observation:
```python
class PhysicsEncoder:
    """
    Learns physics concepts from video:
    - Gravity (things fall)
    - Momentum (moving things continue)
    - Collision (objects bounce/stop)
    - Fluid dynamics (liquids flow)
    """
    
    def encode_dynamics(self, video_frames):
        # Predict next frame
        predicted = self.predictor(video_frames[:-1])
        actual = video_frames[-1]
        
        # The error signal teaches physics
        loss = self.physics_loss(predicted, actual)
        return loss
```

**Training Data Sources**:
- YouTube physics experiments
- Simulation environments (PyBullet, MuJoCo)
- Synthetic data from game engines (Unity, Unreal)

---

### Phase 3: Action-Conditioned Prediction (12-18 months)
**Goal**: Predict "what happens if I do X"

#### 3.1 Action-Conditioned World Model
```
┌─────────────────────────────────────────────────────────────┐
│                ACTION-CONDITIONED PREDICTION                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Current State (s_t) ──┬──→ Predicted State (s_t+1)        │
│                         │                                    │
│   Action (a_t) ─────────┘                                   │
│                                                              │
│   Example:                                                   │
│   State: [ball on table]                                    │
│   Action: [push ball left]                                  │
│   Predicted: [ball rolling left, off table edge]            │
└─────────────────────────────────────────────────────────────┘
```

**Architecture**: Similar to DreamerV3
- State encoder: ~50M parameters
- Dynamics model: ~100M parameters  
- Value predictor: ~50M parameters
- Total: ~200M parameters

**Training**:
- Environment: OpenAI Gym, Minecraft, custom simulation
- Method: Model-based Reinforcement Learning
- GPU: 1-2 RTX 4090s or 1 A100

#### 3.2 Planning Module
Use world model for planning:
```python
class WorldModelPlanner:
    def plan(self, goal, current_state, max_steps=10):
        """
        Use world model to plan actions to reach goal.
        "Imagine" different action sequences and pick best.
        """
        best_plan = None
        best_score = -inf
        
        for action_sequence in self.sample_actions(max_steps):
            # Simulate in "imagination"
            imagined_states = self.world_model.rollout(
                current_state, action_sequence
            )
            
            # How close to goal?
            score = self.goal_similarity(imagined_states[-1], goal)
            
            if score > best_score:
                best_plan = action_sequence
                best_score = score
        
        return best_plan
```

---

### Phase 4: Full World Model (18-36 months)
**Goal**: Complete physical AI system

#### 4.1 Integrated Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    GROUNDZERO WORLD MODEL                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐       │
│  │   VISION    │  │   LANGUAGE   │  │    AUDIO      │       │
│  │  Encoder    │  │   Encoder    │  │   Encoder     │       │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘       │
│         │                │                   │               │
│         └────────────────┼───────────────────┘               │
│                          ▼                                   │
│                  ┌──────────────┐                           │
│                  │  MULTIMODAL  │                           │
│                  │   FUSION     │                           │
│                  └──────┬───────┘                           │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ KNOWLEDGE│    │ PHYSICS  │    │ CAUSAL   │              │
│  │   GRAPH  │    │  MODEL   │    │ REASONER │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                      │
│       └───────────────┼───────────────┘                     │
│                       ▼                                      │
│               ┌──────────────┐                              │
│               │   PLANNING   │                              │
│               │   & ACTION   │                              │
│               └──────────────┘                              │
│                       │                                      │
│         ┌─────────────┼─────────────┐                       │
│         ▼             ▼             ▼                        │
│     [TEXT]       [MOVEMENT]    [CONTROL]                    │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2 Required Scale

| Component | Parameters | Training Data | GPU Requirement |
|-----------|------------|---------------|-----------------|
| Vision Encoder | 300M-1B | 10M+ images | 8x A100 40GB |
| Language Model | 1B-7B | 1T+ tokens | 8x A100 40GB |
| Physics Model | 500M-2B | 1M+ hours video | 16x A100 40GB |
| Dynamics Model | 500M-2B | 10M+ episodes | 16x A100 40GB |
| **Total** | **3B-12B** | Petabytes | **32-64 A100s** |

---

## Minimum Viable World Model (MVWM)

### Can Start NOW with Current Hardware

**Target**: Simplified world model for specific domain (e.g., 2D physics games)

**Architecture** (fits on single RTX 3090/4090):
```
Components:
├── Vision Encoder (ViT-Small): 22M params
├── Dynamics Predictor (Transformer): 50M params  
├── Physics Decoder (CNN): 10M params
└── Controller (MLP): 5M params
Total: ~87M parameters (~350MB memory)
```

**Training Approach**:
1. Generate synthetic training data from simple physics engine
2. Train vision encoder to compress frames
3. Train dynamics predictor to predict next frame
4. Train controller via model-based RL

**Example Domain**: 2D Ball Physics
```
- Balls with gravity
- Bouncing off walls
- Collisions between balls
- Predict next 10 frames from current state + action
```

**Hardware Needed**:
- GPU: RTX 3090 (24GB) or RTX 4090 (24GB)
- RAM: 64GB
- Storage: 1TB SSD
- Time: 1-2 weeks of training

---

## Upgrade Checklist for GroundZero

### Immediate (0-3 months)
- [ ] Add vision encoder module (image → embeddings)
- [ ] Create multimodal fusion layer
- [ ] Extend knowledge graph with causal relations
- [ ] Build synthetic physics dataset generator
- [ ] Implement basic frame prediction

### Short-term (3-6 months)
- [ ] Train video prediction network
- [ ] Add spatial relationship understanding
- [ ] Implement object permanence (things exist when not seen)
- [ ] Create action-conditioned prediction
- [ ] Build simple planning module

### Medium-term (6-12 months)
- [ ] Scale to 3D environments
- [ ] Add temporal consistency across long horizons
- [ ] Implement counterfactual reasoning ("what if...")
- [ ] Train on diverse video data
- [ ] Integrate with robotic control (simulation first)

### Long-term (12-24 months)
- [ ] Real-world video training
- [ ] Physical robot integration
- [ ] Multi-task generalization
- [ ] Self-supervised continuous learning
- [ ] Full world model capabilities

---

## Key Insights from Industry

### 1. Start with Simulation
> "Training these large models costs millions of dollars in GPU compute resources"
> — NVIDIA Cosmos documentation

**Solution**: Use game engines (Unity, Unreal) to generate synthetic training data. This is free and unlimited.

### 2. Latent Space is Key
> "A world model addresses these limits by learning a compact latent state"
> — World Model research

**Insight**: Don't predict raw pixels. Predict compressed representations. This makes training tractable.

### 3. Self-Supervised Learning Works
> "V-JEPA 2 training involves self-supervised learning from video, which allows us to train on video without requiring additional human annotation"
> — Meta AI

**Insight**: You don't need labeled data. The prediction task itself provides the learning signal.

### 4. Start Small, Scale Later
> "DeepSeek's breakthrough model exemplifies how open-source, cost-efficient world model architectures are leveling the playing field"
> — AI industry analysis

**Insight**: Efficient architectures can compete with brute-force scaling. Focus on architecture first.

---

## Conclusion

**When can GroundZero become a world model?**

| Milestone | Timeline | Hardware Needed |
|-----------|----------|-----------------|
| Basic video prediction | 1-3 months | RTX 3090/4090 |
| 2D physics understanding | 3-6 months | RTX 3090/4090 |
| Action-conditioned prediction | 6-9 months | 2x RTX 4090 |
| 3D world simulation | 9-18 months | 4x RTX 4090 or A100 |
| Full world model | 18-36 months | GPU cluster or cloud |

**The key insight**: You don't need Google's resources to start. Begin with:
1. A specific domain (2D physics, simple games)
2. Synthetic training data
3. Small but well-designed architecture
4. Iterative scaling as you prove concepts

GroundZero already has the foundation:
- ✅ Knowledge representation (knowledge graph)
- ✅ Causal reasoning (inference rules)  
- ✅ Learning infrastructure (training loop)
- ✅ Text understanding (transformer)

**Next step**: Add vision encoder and video prediction. This is achievable on current hardware within 3-6 months.

---

## References

1. Ha, D., & Schmidhuber, J. (2018). "World Models." NeurIPS 2018.
2. Hafner et al. (2025). "DreamerV3: Mastering Diverse Domains through World Models." Nature.
3. Meta AI (2025). "V-JEPA 2: World Model for Video Understanding and Robot Planning."
4. NVIDIA (2025). "Cosmos: World Foundation Model Platform for Physical AI."
5. DeepMind (2025). "Genie 3: Foundation World Models for Interactive Simulation."
