# Optimizations 

Before fine-tuning an LLM, several optimizations can be applied to improve efficiency, reduce memory usage, and speed up training. Here are key optimizations:

1. Quantization – Reduces model precision (e.g., FP16, INT8) to lower memory usage and speed up computations with minimal accuracy loss.
2. Pruning – Removes less important model weights or neurons to make the model smaller and faster.
3. Knowledge Distillation – Trains a smaller model (student) using a larger model (teacher) to retain performance while reducing size.
4. LoRA (Low-Rank Adaptation) – Optimizes only specific low-rank layers instead of full model weights, reducing memory and compute requirements.
5. Adapters (Prefix-Tuning, Prompt-Tuning, etc.) – Adds small trainable layers to a frozen model, enabling efficient domain adaptation.
6. Efficient Memory Techniques (PagedAttention, FlashAttention) – Optimizes memory access patterns for faster inference and training.
7. Gradient Checkpointing – Saves memory by recomputing intermediate activations during backpropagation instead of storing them.


### LLM Optimization Methods and Feasibility on M1

| Optimization Method         | Feasibility |
|----------------------------|-------------|
| **Quantization (4-bit, 8-bit)** | ✅ Feasible|
| **LoRA (Low-Rank Adaptation)** | ✅ Feasible|
| **Adapters (Prefix/Prompt Tuning)** | ✅ Feasible|
| **Gradient Checkpointing** | ✅ Feasible|
| **PagedAttention (Memory Optimization)** | ⚠️ Only for inference (Not useful for training) |
| **Pruning** | ❌ Not feasible (Requires high compute) |
| **Knowledge Distillation** | ❌ Not feasible (Needs a larger teacher model) |

These are the **major and widely used** LLM optimization methods. Many others exist, but these are the most effective for improving efficiency, reducing memory usage, and optimizing training and inference.



