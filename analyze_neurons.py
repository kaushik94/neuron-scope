from transformers import AutoModel, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("The dog is on the mat.", return_tensors="pt")
outputs = model(**inputs)
hidden_states = outputs.hidden_states
activations = hidden_states[2].detach().numpy().squeeze()
# sns.heatmap(activations[2].reshape(1, -1))  # Visualize activations for "cat"
# sns.heatmap(activations, xticklabels=False, yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# plt.show()

# Create and save the heatmap
plt.figure(figsize=(10, 2))  # Optional: Set figure size for better readability
sns.heatmap(activations[2].reshape(1, -1), cmap='viridis')  # Visualize activations for "cat"
plt.title("Neuron Activations for 'dog' in Layer 2")  # Optional: Add title
plt.xlabel("Neuron Index")  # Optional: Label axes
plt.ylabel("Token: dog")
plt.savefig("neuron_activations_dog_layer2.png", dpi=300, bbox_inches="tight")  # Save the plot
plt.close()  # Close the plot to free memory
# plt.show()