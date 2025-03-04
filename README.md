# Fine-Tuning LLMs

This repository is a collection of everything I’ve learned while exploring LLMs. I’ve tried to document my journey as thoroughly as possible, covering key concepts, experiments, challenges, and insights along the way.

### About This Repository

1. **platform-Specific Notes:** The techniques, code, and setups in this repo are based on my personal machine — a MacBook Air M1 (2020, 8GB RAM, no dedicated GPU).

2. **General Compatibility:** Most methods should work across different platforms (Windows, Linux, macOS), but certain dependencies or performance factors may vary, configurations might be Mac specific.

3. **What’s Covered:**
- Step-by-step fine-tuning of LLaMA 3.2 1B for domain-specific applications
- Challenges faced while running LLMs on low-end hardware
- Performance tweaks and optimizations for limited RAM setups


4. This repository is a work in progress, documenting my personal experiences and experiments with LLM fine-tuning. Everything here is anecdotal, based on what I’ve tried and learned firsthand.

5. Each folder contains a README.md with a general introduction to the topic and an _InDepth.md with detailed explanations. Some files may be nested within subfolders for better organization, and everything is named and structured logically.


### Repository Structure 

**00_WhatIsLLM** - General Introduction To LLM's

**01_BaseModel** - How to load and inference a base model

**02_Optimizations** - Parameter Efficient Fine-Tuning (LoRA, QLoRA) and Quantization Techniques




### Topic Specific Index 

- [Intro to LLM](./00_WhatIsLLM/README.md#whats-a-llm), [LLM InDepth](./00_WhatIsLLM/LLM_InDepth.md), [Intro to Fine-Tuning](./00_WhatIsLLM/README.md#fine-tuning-llm), [Fine-Tuning InDepth](./00_WhatIsLLM/FineTuning_InDepth.md)

- [Intro to Models](./01_BaseModel/README.md#base-model), [Models InDepth](./01_BaseModel/Models_InDepth.md), [Llama-3.2-1B InDepth](./01_BaseModel/Llama-3.2-1B_InDepth.md)

- [Intro to Optimizations]()

- [Quantization InDepth]()

- [Efficient Model Formats](), [Distilled Models]()

- [LoRA In_Depth](), [CLoRA InDepth]()

