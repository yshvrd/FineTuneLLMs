# Base Model 

Not all LLMs are based on a base model, but most fine-tuned models are. This repo serves as a brief introduction to LLMs and their fine tuning. 

A base model is a large, pretrained model trained on diverse datasets to learn general language patterns. It serves as the foundation for fine-tuning on specific tasks or domains.

LLaMA 3.2 1B is a model trained by Meta on a broad dataset. We will be using it as our base model. It is not specialized but provides a strong foundation for fine-tuning. We can train it on domain-specific data to make it more useful for our specific application.



### General Project Setup 

I am using an M1 MacBook Air (2020, 8GB), and these instructions are based on that. For other operating systems, the steps should be somewhat similar, but please verify accordingly.

- install huggingface-cli and login (refer to official documentation)
```zsh
brew install huggingface-cli
huggingface-cli login
```

- use a python virtual environment 
```zsh
python3 -m venv .venv 
```

- load the .venv
```zsh
source .venv/bin/activate
```

- install all the dependencies 
```zsh
pip install -r requirements.txt
```



### Download and run the Llama model 

There are two broad ways to download and run a model :  

**Note that you need to signup on huggingface and then request this model in order to download it.**


- **Method-1 :** use it directly, in which case all of the setup and dependencies is taken care of and you can run it with minimal code. (first run will take some time as it needs to download the model)

```python
import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu" # mps, auto 
)

result = pipe("Hello, who or what are you ?")
print(result)

```
(average output time - 6seconds on CPU and 8seconds on MPS)


- **Method-2 :** download the original model weights, which requires manual setup and is slower for some cases

download the model using -

```zsh 
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir Llama-3.2-1B
```

and then run it with python -

```python
# CPU version

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
    )

prompt = "Hello, who or what are you ?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_length=50)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Model Response:", response)
```


```python 
# GPU (mps) version

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model on MPS (Apple Metal GPU)
device = torch.device("mps")

model_path = "Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Encode input
prompt = "Hello, who or what are you?"
inputs = tokenizer(prompt, return_tensors="pt")

# Move input tensors to MPS
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate and decode response
output_ids = model.generate(**inputs, max_length=50)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Model Response:", response)


```
(average output time - 6 seconds on CPU and 10 seconds on GPU)



### Key Takeaways - 

- Base models like LLaMA 3.2 1B are pretrained on general data and need fine-tuning for specific tasks.

- Not all LLMs are derived from a base model, some are trained entirely from scratch, though this is rare due to cost.

- Using pipeline() → Minimal Setup and Faster Execution

- Manually Downloading & Running Weights → More Control, may be a little slower 

- MPS is slow for this use case, possibly because Transformers and Torch are not as well-optimized for Apple Metal as they are for NVIDIA CUDA. Additionally, MPS was observed to significantly increase memory pressure, while the CPU handled it more efficiently.

