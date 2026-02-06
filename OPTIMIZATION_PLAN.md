# ml-translate Optimization Plan

## Overview
This document outlines optimization and extension opportunities for the neural machine translation codebase, organized by priority.

---

## Phase 1: Critical Foundation

### 1.1 Train/Val/Test Split
- **Location**: `src/ml_translate/data.py`
- **Current**: All data loaded into single DataLoader
- **Change**: Add `split_pairs()` function and new `get_dataloaders()` returning train/val/test loaders
- **Status**: [x] Complete

### 1.2 Validation Loop
- **Location**: `src/ml_translate/train.py`
- **Current**: No validation during training
- **Change**: Add `validate_epoch()` function, call after each training epoch
- **Status**: [x] Complete

### 1.3 BLEU Score Evaluation
- **Location**: `src/ml_translate/eval.py`
- **Current**: Only qualitative random sampling
- **Change**: Add `evaluate_bleu()` using torchtext.data.metrics.bleu_score
- **Status**: [x] Complete

### 1.4 Early Stopping
- **Location**: `src/ml_translate/train.py`
- **Current**: Fixed number of epochs
- **Change**: Add `EarlyStopping` class with patience parameter
- **Status**: [x] Complete

---

## Phase 2: Training Improvements

### 2.1 Learning Rate Scheduling
- **Location**: `src/ml_translate/train.py`
- **Current**: Fixed learning rate throughout training
- **Change**: Add `ReduceLROnPlateau` scheduler with configurable patience/factor
- **Status**: [x] Complete

### 2.2 Gradient Clipping
- **Location**: `src/ml_translate/train.py`
- **Current**: No gradient clipping (risk of exploding gradients)
- **Change**: Add `clip_grad_norm_()` before optimizer step, default max_norm=1.0
- **Status**: [x] Complete

### 2.3 Model Checkpointing
- **Location**: New file or `src/ml_translate/train.py`
- **Current**: No saving of model state
- **Change**: Add `CheckpointManager` class to save/load best models
- **Status**: [~] Skipped (not needed for current model size)

### 2.4 Enhanced Configuration
- **Location**: `src/ml_translate/config.py`
- **Current**: Minimal config with 5 parameters
- **Change**: Modular config classes (EmbeddingConfig, EncoderConfig, TrainingConfig, etc.)
- **Status**: [ ] Not started

---

## Phase 3: Embeddings & Model Architecture

### 3.1 Pre-trained Embedding Support
- **Location**: `src/ml_translate/embedding.py`, `src/ml_translate/model.py`
- **Current**: Random initialization only (`nn.Embedding`)
- **Change**:
  - Added `embedding.py` with `load_glove_embeddings()` and `PretrainedEmbedding` class
  - Models accept optional `embedding` parameter
  - Changed hidden_size to 100 to match GloVe dimension
- **Impact**: 30-50% faster convergence
- **Status**: [x] Complete

### 3.2 Bidirectional Encoder
- **Location**: `src/ml_translate/model.py`
- **Current**: Unidirectional GRU encoder
- **Change**: Add `BidirectionalEncoderRNN` class with hidden projection
- **Impact**: ~15% quality improvement
- **Status**: [ ] Not started

### 3.3 Luong Attention
- **Location**: `src/ml_translate/model.py`
- **Current**: Single Bahdanau attention only
- **Change**: Added `LuongAttention` class with dot/general/concat methods.
  `AttnDecoderRNN` now accepts `attention_type` parameter.
- **Status**: [x] Complete

### 3.4 Layer Stacking
- **Location**: `src/ml_translate/model.py`
- **Current**: Single layer encoder/decoder
- **Change**: Add `num_layers` parameter, support residual connections
- **Status**: [ ] Not started

### 3.5 Layer Normalization
- **Location**: `src/ml_translate/model.py`
- **Current**: No normalization layers
- **Change**: Add `nn.LayerNorm` after GRU layers
- **Status**: [ ] Not started

---

## Phase 4: Evaluation & Decoding

### 4.1 Beam Search Decoding
- **Location**: `src/ml_translate/eval.py`
- **Current**: Greedy decoding (top-1 only)
- **Change**: Add `evaluate_with_beam_search()` with configurable beam width
- **Impact**: 15-25% quality improvement
- **Status**: [ ] Not started

### 4.2 Additional Metrics
- **Location**: `src/ml_translate/eval.py`
- **Current**: No quantitative metrics
- **Change**: Add `TranslationMetrics` class with TER, length ratio, coverage
- **Status**: [ ] Not started

### 4.3 Comprehensive Evaluation
- **Location**: `src/ml_translate/eval.py`
- **Current**: Single sentence evaluation only
- **Change**: Add `comprehensive_evaluate()` for corpus-level metrics
- **Status**: [ ] Not started

---

## Phase 5: Performance Optimization

### 5.1 Mixed Precision Training (AMP)
- **Location**: `src/ml_translate/train.py`
- **Current**: Full float32
- **Change**: Added `use_amp` and `device` parameters to `train()`, uses `autocast` and `GradScaler`
- **Impact**: 2-3x speedup on GPU
- **Status**: [x] Complete

### 5.2 Dynamic Batching
- **Location**: `src/ml_translate/data.py`, `src/ml_translate/model.py`
- **Current**: Fixed padding to MAX_LENGTH
- **Change**: Added `TranslationDataset`, `collate_dynamic_batch()`, and `dynamic_batching` parameter.
  Decoders now use target length during training, MAX_LENGTH during inference.
- **Impact**: 20-30% faster training, 40-60% less memory
- **Status**: [x] Complete

### 5.3 Device Management
- **Location**: `src/ml_translate/utils.py`
- **Current**: Manual device handling
- **Change**: Added `get_device()` function with auto-detection (CUDA > MPS > CPU)
- **Status**: [x] Complete

---

## Phase 6: Data Augmentation (Optional)

### 6.1 Back-Translation
- **Location**: `src/ml_translate/data.py`
- **Current**: No augmentation
- **Change**: Add `back_translate_augment()` for synthetic training data
- **Impact**: 10-15% quality improvement
- **Status**: [ ] Not started

### 6.2 Word Dropout
- **Location**: `src/ml_translate/data.py`
- **Current**: No noise injection
- **Change**: Random word dropout during training for regularization
- **Status**: [ ] Not started

---

## Progress Tracking

| Phase | Items | Completed | Status |
|-------|-------|-----------|--------|
| 1. Critical Foundation | 4 | 4 | Complete |
| 2. Training Improvements | 4 | 2 (1 skipped) | In progress |
| 3. Embeddings & Architecture | 5 | 2 | In progress |
| 4. Evaluation & Decoding | 3 | 0 | Not started |
| 5. Performance | 3 | 3 | Complete |
| 6. Data Augmentation | 2 | 0 | Not started |

---

## Notes

- Each item should be implemented with corresponding tests
- Update `config.py` as new parameters are added
- Maintain backward compatibility where possible
- Run existing tests after each change to ensure no regressions
