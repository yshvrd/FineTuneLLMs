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


## Quantization 

Quantization is the process of reducing the precision of numerical values in a model to make it more efficient. In deep learning, it typically refers to reducing the bit-width of floating-point numbers (e.g., 32-bit floating point, FP32) to lower-bit representations like 16-bit floating point (FP16), 8-bit integer (INT8), or even 4-bit integer (INT4).

In simple words, it is a way to reduce model and computational complexity by reducing the weights from a higher memory format to a lower memory format.

### Why is Quantization Used?

1. Reduces Model Size – Lower-bit representations take up less memory, making models easier to store and deploy on edge devices.

2. Improves Inference Speed – Operations on lower-bit values require fewer computational resources, leading to faster inference.

3. Lowers Power and Memory Consumption – Useful for mobile and embedded devices where memory and power efficiency is crucial.

### Types of Quantization

1. Post-Training Quantization (PTQ) – The model is trained in full precision (FP32) and then converted to lower precision after training.

2. Quantization-Aware Training (QAT) – The model is trained while simulating lower-bit computations to adapt to the reduced precision.



## LoRA (Low Rank Adaptation)

LoRA (Low-Rank Adaptation) is a technique used to fine-tune large language models (LLMs) efficiently by freezing most of the model's weights and only training a small set of additional parameters. Fine-tuning a large model like LLaMA 3.2 1B directly is expensive in terms of memory, computation, and storage. LoRA solves this by only updating a small, low-rank matrix instead of the entire weight matrix.


In simpler terms, LoRA only updates the required weights in a matrix. Instead of modifying the huge weight matrices, it adds small matrices that learn the changes. This reduces memory usage and speeds up training while still adapting the model to new tasks.


