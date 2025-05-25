import os

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_API_KEY'))



from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import torch
activations = {}
def hook_fn(module, input, output):
    activations['mlp_output'] = output.detach()
# Register hook to the MLP of the first transformer layer
model.model.layers[0].mlp.register_forward_hook(hook_fn)
# Tokenize input
prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt")
# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
# Access activations
mlp_activations = activations['mlp_output']
print(mlp_activations.shape)  # Shape: [batch_size, sequence_length, intermediate_size]

import matplotlib.pyplot as plt
neuron_idx = 0  # Example: Inspect the first neuron
plt.plot(mlp_activations[0, :, neuron_idx].cpu().numpy())
plt.title(f"Activation of Neuron {neuron_idx} in MLP Layer")
plt.xlabel("Token Position")
plt.ylabel("Activation Value")
plt.show()