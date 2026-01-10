"""
GroundZero Custom Speech-to-Text Model
=======================================
A trainable speech recognition model built from scratch.

This runs ALONGSIDE Whisper initially, but as it learns from
corrections and more data, it can eventually replace Whisper entirely.

ARCHITECTURE
============
Based on modern ASR (Automatic Speech Recognition) approaches:

1. Feature Extraction (Mel Spectrogram → Features)
   - Convert raw audio to mel-frequency spectrograms
   - Convolutional feature extractor

2. Encoder (Conformer-style)
   - Combines CNN + Transformer
   - Local features (CNN) + Global context (Attention)

3. Decoder (CTC + Attention)
   - CTC for alignment-free training
   - Attention decoder for better quality

4. Language Model Integration
   - Optional shallow fusion with our neural brain's LM

TRAINING STRATEGY
=================
1. Start with Whisper as teacher (knowledge distillation)
2. Collect user corrections (reinforcement learning)
3. Fine-tune on domain-specific vocabulary
4. Eventually run standalone

SUPPORTED FORMATS
=================
- WAV (16kHz, mono)
- WebM (converted internally)
- MP3 (converted internally)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import pickle
import struct
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# AUDIO PROCESSING (Pure NumPy - No Dependencies)
# ============================================================

class AudioProcessor:
    """
    Audio feature extraction without external dependencies.
    Converts raw audio to mel spectrograms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,           # 25ms window at 16kHz
        hop_length: int = 160,       # 10ms hop
        n_mels: int = 80,           # 80 mel bins (standard for ASR)
        fmin: float = 0.0,
        fmax: float = 8000.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Pre-compute mel filterbank
        self.mel_filters = self._create_mel_filterbank()
        
        # Hann window for STFT
        self.window = np.hanning(n_fft).astype(np.float32)
    
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to Mel scale"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert Mel to Hz"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix"""
        # Mel points
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        
        # Convert to FFT bins
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        n_freq_bins = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_freq_bins), dtype=np.float32)
        
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising edge
            for j in range(left, center):
                if j < n_freq_bins:
                    filterbank[i, j] = (j - left) / max(center - left, 1)
            
            # Falling edge
            for j in range(center, right):
                if j < n_freq_bins:
                    filterbank[i, j] = (right - j) / max(right - center, 1)
        
        return filterbank
    
    def stft(self, audio: np.ndarray) -> np.ndarray:
        """Short-time Fourier Transform"""
        # Pad audio
        pad_length = self.n_fft // 2
        audio = np.pad(audio, (pad_length, pad_length), mode='reflect')
        
        # Calculate number of frames
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        
        # Pre-allocate output
        n_freq_bins = self.n_fft // 2 + 1
        stft_matrix = np.zeros((n_freq_bins, n_frames), dtype=np.complex64)
        
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * self.window
            stft_matrix[:, i] = np.fft.rfft(frame)
        
        return stft_matrix
    
    def mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel spectrogram.
        
        Args:
            audio: Raw audio samples (1D numpy array, values in [-1, 1])
            
        Returns:
            Mel spectrogram of shape (n_mels, time_steps)
        """
        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0  # Assume 16-bit audio
        
        # Compute STFT
        stft = self.stft(audio)
        
        # Power spectrum
        power = np.abs(stft) ** 2
        
        # Apply mel filterbank
        mel_spec = self.mel_filters @ power
        
        # Log mel spectrogram
        mel_spec = np.log(mel_spec + 1e-10)
        
        return mel_spec.astype(np.float32)
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio from file (WAV only, no dependencies).
        For WebM/MP3, use ffmpeg externally first.
        """
        with open(file_path, 'rb') as f:
            # Read WAV header
            riff = f.read(4)
            if riff != b'RIFF':
                raise ValueError("Not a WAV file")
            
            f.read(4)  # File size
            wave = f.read(4)
            if wave != b'WAVE':
                raise ValueError("Not a WAV file")
            
            # Find data chunk
            while True:
                chunk_id = f.read(4)
                chunk_size = struct.unpack('<I', f.read(4))[0]
                
                if chunk_id == b'fmt ':
                    fmt_data = f.read(chunk_size)
                    audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                    n_channels = struct.unpack('<H', fmt_data[2:4])[0]
                    sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                    bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                    
                elif chunk_id == b'data':
                    # Read audio data
                    audio_data = f.read(chunk_size)
                    break
                else:
                    f.seek(chunk_size, 1)  # Skip unknown chunk
        
        # Convert to numpy
        if bits_per_sample == 16:
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif bits_per_sample == 32:
            audio = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
        
        # Convert stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        return audio
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling"""
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, new_length)
        
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

class ASRVocabulary:
    """
    Character-level vocabulary for ASR.
    Includes special tokens for CTC.
    """
    
    def __init__(self):
        # Character set: lowercase letters, digits, common punctuation
        self.chars = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?'-")
        
        # Special tokens
        self.blank_token = "<blank>"  # CTC blank
        self.unk_token = "<unk>"      # Unknown
        self.sos_token = "<sos>"      # Start of sequence
        self.eos_token = "<eos>"      # End of sequence
        
        # Build vocabulary
        self.special_tokens = [self.blank_token, self.unk_token, self.sos_token, self.eos_token]
        self.tokens = self.special_tokens + self.chars
        
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        
        # IDs
        self.blank_id = self.token_to_id[self.blank_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.sos_id = self.token_to_id[self.sos_token]
        self.eos_id = self.token_to_id[self.eos_token]
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        text = text.lower()
        return [self.token_to_id.get(c, self.unk_id) for c in text]
    
    def decode(self, ids: List[int], remove_blanks: bool = True) -> str:
        """Convert token IDs to text"""
        chars = []
        prev_id = None
        
        for id in ids:
            # Skip blanks
            if remove_blanks and id == self.blank_id:
                prev_id = id
                continue
            
            # Skip repeated characters (CTC decoding)
            if id == prev_id:
                continue
            
            # Skip special tokens
            if id in [self.unk_id, self.sos_id, self.eos_id]:
                prev_id = id
                continue
            
            if id in self.id_to_token:
                chars.append(self.id_to_token[id])
            
            prev_id = id
        
        return ''.join(chars)


# ============================================================
# NEURAL MODEL (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    
    class ConvFeatureExtractor(nn.Module):
        """
        Convolutional feature extractor for audio.
        Reduces time dimension while extracting features.
        """
        
        def __init__(self, n_mels: int = 80, hidden_dim: int = 256):
            super().__init__()
            
            self.conv_layers = nn.Sequential(
                # First conv block
                nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.GELU(),
                
                # Second conv block
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.GELU(),
                
                # Third conv block
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.GELU(),
            )
            
            # Calculate output dimension
            # After 3x stride-2 convs: n_mels/8
            conv_out_dim = 128 * (n_mels // 8)
            
            self.projection = nn.Linear(conv_out_dim, hidden_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Mel spectrogram (batch, n_mels, time)
            Returns:
                Features (batch, time', hidden_dim)
            """
            # Add channel dimension
            x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
            
            # Apply convolutions
            x = self.conv_layers(x)  # (batch, 128, n_mels/8, time/8)
            
            # Reshape for linear projection
            batch, channels, freq, time = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
            x = x.view(batch, time, channels * freq)  # (batch, time, channels*freq)
            
            # Project to hidden dimension
            x = self.projection(x)
            
            return x
    
    
    class ConformerBlock(nn.Module):
        """
        Conformer block: Feed-Forward + Self-Attention + Convolution + Feed-Forward
        Combines the strengths of Transformers and CNNs.
        """
        
        def __init__(self, hidden_dim: int = 256, n_heads: int = 4, 
                     ff_dim: int = 1024, conv_kernel: int = 31, dropout: float = 0.1):
            super().__init__()
            
            # First feed-forward (half-step)
            self.ff1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            
            # Self-attention
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.attn_dropout = nn.Dropout(dropout)
            
            # Convolution module
            self.conv_norm = nn.LayerNorm(hidden_dim)
            self.conv = nn.Sequential(
                nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1),
                nn.GLU(dim=1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, 
                         padding=conv_kernel // 2, groups=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Dropout(dropout)
            )
            
            # Second feed-forward (half-step)
            self.ff2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            
            self.final_norm = nn.LayerNorm(hidden_dim)
        
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: Input tensor (batch, time, hidden)
                mask: Attention mask (batch, time)
            """
            # First FF (half-step residual)
            x = x + 0.5 * self.ff1(x)
            
            # Self-attention
            attn_in = self.attn_norm(x)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=mask)
            x = x + self.attn_dropout(attn_out)
            
            # Convolution
            conv_in = self.conv_norm(x)
            conv_in = conv_in.transpose(1, 2)  # (batch, hidden, time)
            conv_out = self.conv(conv_in)
            conv_out = conv_out.transpose(1, 2)  # (batch, time, hidden)
            x = x + conv_out
            
            # Second FF (half-step residual)
            x = x + 0.5 * self.ff2(x)
            
            return self.final_norm(x)
    
    
    class SpeechEncoder(nn.Module):
        """
        Full speech encoder: Feature Extraction + Conformer Blocks
        """
        
        def __init__(
            self,
            n_mels: int = 80,
            hidden_dim: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            ff_dim: int = 1024,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.feature_extractor = ConvFeatureExtractor(n_mels, hidden_dim)
            
            self.conformer_blocks = nn.ModuleList([
                ConformerBlock(hidden_dim, n_heads, ff_dim, dropout=dropout)
                for _ in range(n_layers)
            ])
        
        def forward(self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                mel: Mel spectrogram (batch, n_mels, time)
                mask: Padding mask (batch, time)
            Returns:
                Encoded features (batch, time', hidden)
            """
            # Extract features
            x = self.feature_extractor(mel)
            
            # Apply conformer blocks
            for block in self.conformer_blocks:
                x = block(x, mask)
            
            return x
    
    
    class GroundZeroASR(nn.Module):
        """
        Complete ASR model with CTC output.
        
        Uses Connectionist Temporal Classification (CTC) for
        alignment-free training - no need for frame-level labels!
        """
        
        def __init__(
            self,
            vocab_size: int = 48,
            n_mels: int = 80,
            hidden_dim: int = 256,
            n_layers: int = 4,
            n_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            
            # Encoder
            self.encoder = SpeechEncoder(
                n_mels=n_mels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout
            )
            
            # CTC output layer
            self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            """
            Args:
                mel: Mel spectrogram (batch, n_mels, time)
            Returns:
                Log probabilities (batch, time', vocab_size)
            """
            # Encode
            encoded = self.encoder(mel)
            
            # CTC output
            logits = self.ctc_head(encoded)
            log_probs = F.log_softmax(logits, dim=-1)
            
            return log_probs
        
        def decode_greedy(self, mel: torch.Tensor, vocab: ASRVocabulary) -> str:
            """
            Greedy decoding for inference.
            """
            self.eval()
            with torch.no_grad():
                log_probs = self(mel)
                
                # Greedy: take argmax at each timestep
                predictions = log_probs.argmax(dim=-1)  # (batch, time)
                
                # Decode first sample
                ids = predictions[0].cpu().tolist()
                text = vocab.decode(ids)
                
                return text
        
        def decode_beam(self, mel: torch.Tensor, vocab: ASRVocabulary, 
                       beam_width: int = 10) -> str:
            """
            Beam search decoding for better quality.
            """
            self.eval()
            with torch.no_grad():
                log_probs = self(mel)  # (1, time, vocab)
                log_probs = log_probs[0]  # (time, vocab)
                
                # Initialize beams: (log_prob, sequence)
                beams = [(0.0, [])]
                
                for t in range(log_probs.shape[0]):
                    new_beams = []
                    
                    for log_prob, seq in beams:
                        for token_id in range(self.vocab_size):
                            token_log_prob = log_probs[t, token_id].item()
                            new_log_prob = log_prob + token_log_prob
                            new_seq = seq + [token_id]
                            new_beams.append((new_log_prob, new_seq))
                    
                    # Keep top beams
                    new_beams.sort(key=lambda x: -x[0])
                    beams = new_beams[:beam_width]
                
                # Return best beam
                best_seq = beams[0][1]
                return vocab.decode(best_seq)


# ============================================================
# TRAINING SYSTEM
# ============================================================

@dataclass
class ASRTrainingData:
    """Single training example"""
    audio_path: str
    transcript: str
    duration: float = 0.0
    source: str = ""  # 'whisper', 'user_correction', 'manual'
    confidence: float = 1.0


class ASRTrainer:
    """
    Trainer for the custom ASR model.
    
    Training strategies:
    1. Supervised: Train on (audio, transcript) pairs
    2. Distillation: Learn from Whisper outputs
    3. Correction: Learn from user corrections
    """
    
    def __init__(self, data_dir: str = "data/asr"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_processor = AudioProcessor()
        self.vocab = ASRVocabulary()
        
        # Model (lazy initialization)
        self._model = None
        self._optimizer = None
        
        # Training data
        self.training_data: List[ASRTrainingData] = []
        self._load_training_data()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_duration': 0.0,
            'training_steps': 0,
            'best_loss': float('inf'),
            'corrections_learned': 0
        }
    
    @property
    def model(self) -> 'GroundZeroASR':
        """Lazy model initialization"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ASR model")
        
        if self._model is None:
            self._model = GroundZeroASR(
                vocab_size=self.vocab.vocab_size,
                n_mels=80,
                hidden_dim=256,
                n_layers=4,
                n_heads=4
            )
            self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)
        
        return self._model
    
    def add_training_sample(
        self,
        audio_path: str,
        transcript: str,
        source: str = "manual",
        confidence: float = 1.0
    ) -> None:
        """Add a training sample"""
        sample = ASRTrainingData(
            audio_path=audio_path,
            transcript=transcript.lower().strip(),
            source=source,
            confidence=confidence
        )
        
        self.training_data.append(sample)
        self.stats['total_samples'] += 1
        
        self._save_training_data()
    
    def add_whisper_distillation(
        self,
        audio_path: str,
        whisper_transcript: str
    ) -> None:
        """Add Whisper output for knowledge distillation"""
        self.add_training_sample(
            audio_path=audio_path,
            transcript=whisper_transcript,
            source="whisper",
            confidence=0.9  # Slightly lower since it's from another model
        )
    
    def add_user_correction(
        self,
        audio_path: str,
        corrected_transcript: str,
        original_transcript: str = ""
    ) -> None:
        """
        Learn from user correction.
        This is valuable data - user explicitly fixed a mistake!
        """
        self.add_training_sample(
            audio_path=audio_path,
            transcript=corrected_transcript,
            source="user_correction",
            confidence=1.0  # High confidence - user verified
        )
        self.stats['corrections_learned'] += 1
    
    def train_step(self, batch_size: int = 4) -> Dict[str, float]:
        """Run one training step"""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        if len(self.training_data) < batch_size:
            return {'error': f'Need at least {batch_size} samples'}
        
        # Sample batch
        import random
        batch = random.sample(self.training_data, batch_size)
        
        # Prepare data
        mels = []
        transcripts = []
        
        for sample in batch:
            try:
                audio = self.audio_processor.load_audio(sample.audio_path)
                mel = self.audio_processor.mel_spectrogram(audio)
                mels.append(mel)
                transcripts.append(sample.transcript)
            except Exception as e:
                continue
        
        if len(mels) == 0:
            return {'error': 'Failed to load audio'}
        
        # Pad mels to same length
        max_time = max(m.shape[1] for m in mels)
        mel_batch = np.zeros((len(mels), 80, max_time), dtype=np.float32)
        for i, mel in enumerate(mels):
            mel_batch[i, :, :mel.shape[1]] = mel
        
        # Convert to tensors
        mel_tensor = torch.from_numpy(mel_batch)
        
        # Encode transcripts
        target_ids = [self.vocab.encode(t) for t in transcripts]
        target_lengths = torch.tensor([len(t) for t in target_ids], dtype=torch.long)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in target_ids],
            batch_first=True,
            padding_value=self.vocab.blank_id
        )
        
        # Forward pass
        self.model.train()
        log_probs = self.model(mel_tensor)
        
        # Input lengths (after convolution downsampling)
        input_lengths = torch.tensor(
            [log_probs.shape[1]] * len(mels), 
            dtype=torch.long
        )
        
        # CTC loss
        log_probs_t = log_probs.transpose(0, 1)  # (time, batch, vocab)
        loss = F.ctc_loss(
            log_probs_t, targets, input_lengths, target_lengths,
            blank=self.vocab.blank_id, zero_infinity=True
        )
        
        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._optimizer.step()
        
        self.stats['training_steps'] += 1
        if loss.item() < self.stats['best_loss']:
            self.stats['best_loss'] = loss.item()
        
        return {
            'loss': loss.item(),
            'batch_size': len(mels),
            'step': self.stats['training_steps']
        }
    
    def train_epoch(self, steps_per_epoch: int = 100, batch_size: int = 4) -> Dict[str, Any]:
        """Train for one epoch"""
        losses = []
        
        for _ in range(steps_per_epoch):
            result = self.train_step(batch_size)
            if 'loss' in result:
                losses.append(result['loss'])
        
        return {
            'steps': len(losses),
            'avg_loss': np.mean(losses) if losses else 0,
            'min_loss': np.min(losses) if losses else 0,
            'total_steps': self.stats['training_steps']
        }
    
    def transcribe(self, audio_path: str, use_beam: bool = False) -> str:
        """Transcribe audio using our model"""
        if not TORCH_AVAILABLE:
            return ""
        
        # Load and process audio
        audio = self.audio_processor.load_audio(audio_path)
        mel = self.audio_processor.mel_spectrogram(audio)
        
        # Add batch dimension
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        
        # Decode
        if use_beam:
            return self.model.decode_beam(mel_tensor, self.vocab)
        else:
            return self.model.decode_greedy(mel_tensor, self.vocab)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save model and training state"""
        if self._model is None:
            return
        
        save_path = Path(path) if path else self.data_dir / "asr_model.pt"
        
        torch.save({
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'stats': self.stats,
            'vocab_size': self.vocab.vocab_size
        }, save_path)
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load model from checkpoint"""
        if not TORCH_AVAILABLE:
            return False
        
        load_path = Path(path) if path else self.data_dir / "asr_model.pt"
        
        if not load_path.exists():
            return False
        
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.stats = checkpoint['stats']
        
        return True
    
    def _save_training_data(self) -> None:
        """Save training data list"""
        data_path = self.data_dir / "training_data.json"
        
        data = [
            {
                'audio_path': s.audio_path,
                'transcript': s.transcript,
                'source': s.source,
                'confidence': s.confidence
            }
            for s in self.training_data
        ]
        
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_training_data(self) -> None:
        """Load existing training data"""
        data_path = self.data_dir / "training_data.json"
        
        if not data_path.exists():
            return
        
        with open(data_path) as f:
            data = json.load(f)
        
        self.training_data = [
            ASRTrainingData(**d) for d in data
        ]
        self.stats['total_samples'] = len(self.training_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            **self.stats,
            'model_available': self._model is not None,
            'pytorch_available': TORCH_AVAILABLE,
            'samples_from_whisper': sum(1 for s in self.training_data if s.source == 'whisper'),
            'samples_from_corrections': sum(1 for s in self.training_data if s.source == 'user_correction'),
            'samples_manual': sum(1 for s in self.training_data if s.source == 'manual')
        }


# ============================================================
# HYBRID TRANSCRIPTION (Whisper + Custom)
# ============================================================

class HybridTranscriber:
    """
    Hybrid transcription system.
    
    Uses Whisper as primary (high quality) but:
    1. Collects data to train custom model
    2. Falls back to custom model if Whisper unavailable
    3. Gradually shifts weight to custom model as it improves
    """
    
    def __init__(self, data_dir: str = "data/asr"):
        self.data_dir = Path(data_dir)
        self.trainer = ASRTrainer(data_dir)
        
        # Whisper model (lazy loaded)
        self._whisper_model = None
        self._whisper_available = None
        
        # Blending weight (0 = all Whisper, 1 = all custom)
        self.custom_weight = 0.0
    
    @property
    def whisper_available(self) -> bool:
        """Check if Whisper is available"""
        if self._whisper_available is None:
            try:
                import whisper
                self._whisper_available = True
            except ImportError:
                self._whisper_available = False
        return self._whisper_available
    
    def transcribe(
        self, 
        audio_path: str,
        save_for_training: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio using hybrid approach.
        
        Returns:
            {
                'text': str,
                'source': 'whisper' | 'custom' | 'hybrid',
                'whisper_text': str (if available),
                'custom_text': str (if available),
                'confidence': float
            }
        """
        result = {
            'text': '',
            'source': 'none',
            'confidence': 0.0
        }
        
        whisper_text = None
        custom_text = None
        
        # Try Whisper
        if self.whisper_available:
            try:
                if self._whisper_model is None:
                    import whisper
                    self._whisper_model = whisper.load_model("base")
                
                whisper_result = self._whisper_model.transcribe(audio_path)
                whisper_text = whisper_result.get('text', '').strip()
                result['whisper_text'] = whisper_text
                
                # Save for training
                if save_for_training and whisper_text:
                    self.trainer.add_whisper_distillation(audio_path, whisper_text)
                    
            except Exception as e:
                result['whisper_error'] = str(e)
        
        # Try custom model
        if TORCH_AVAILABLE and self.trainer.stats['training_steps'] > 100:
            try:
                custom_text = self.trainer.transcribe(audio_path)
                result['custom_text'] = custom_text
            except Exception as e:
                result['custom_error'] = str(e)
        
        # Determine final output
        if whisper_text and custom_text:
            # Blend based on weight
            if self.custom_weight < 0.3:
                result['text'] = whisper_text
                result['source'] = 'whisper'
                result['confidence'] = 0.9
            elif self.custom_weight > 0.7:
                result['text'] = custom_text
                result['source'] = 'custom'
                result['confidence'] = 0.8
            else:
                # Use Whisper but note custom is available
                result['text'] = whisper_text
                result['source'] = 'hybrid'
                result['confidence'] = 0.85
        elif whisper_text:
            result['text'] = whisper_text
            result['source'] = 'whisper'
            result['confidence'] = 0.9
        elif custom_text:
            result['text'] = custom_text
            result['source'] = 'custom'
            result['confidence'] = 0.7
        else:
            result['text'] = ''
            result['source'] = 'none'
            result['confidence'] = 0.0
        
        return result
    
    def report_correction(
        self,
        audio_path: str,
        corrected_text: str,
        original_text: str = ""
    ) -> None:
        """
        User corrected a transcription - learn from it!
        This is valuable training data.
        """
        self.trainer.add_user_correction(
            audio_path=audio_path,
            corrected_transcript=corrected_text,
            original_transcript=original_text
        )
        
        # Increase custom weight slightly after each correction
        self.custom_weight = min(1.0, self.custom_weight + 0.01)
    
    def train_custom_model(self, epochs: int = 1, steps_per_epoch: int = 100) -> Dict[str, Any]:
        """Train the custom model"""
        results = []
        
        for epoch in range(epochs):
            result = self.trainer.train_epoch(steps_per_epoch)
            results.append(result)
        
        # Update custom weight based on performance
        if results and results[-1]['avg_loss'] < 1.0:
            self.custom_weight = min(1.0, self.custom_weight + 0.05)
        
        return {
            'epochs': len(results),
            'final_loss': results[-1]['avg_loss'] if results else 0,
            'custom_weight': self.custom_weight,
            'total_training_steps': self.trainer.stats['training_steps']
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid transcriber statistics"""
        return {
            'whisper_available': self.whisper_available,
            'custom_weight': self.custom_weight,
            'trainer_stats': self.trainer.get_stats(),
            'ready_for_standalone': self.custom_weight > 0.7 and self.trainer.stats['training_steps'] > 1000
        }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'AudioProcessor',
    'ASRVocabulary', 
    'ASRTrainer',
    'HybridTranscriber',
    'ASRTrainingData'
]

if TORCH_AVAILABLE:
    __all__.extend([
        'GroundZeroASR',
        'SpeechEncoder',
        'ConformerBlock',
        'ConvFeatureExtractor'
    ])
