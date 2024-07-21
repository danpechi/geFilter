import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_dataset
import pandas as pd

from utils.data import CustomDataset

from utils.grad_norm import precompute_gradient_norms, compute_gradient_norm


def train_on_selected_samples(selected_samples, model, optimizer):
    batch_size = 8
    for i in range(0, len(selected_samples), batch_size):
        batch_samples = selected_samples[i:i + batch_size]
        batch_input_ids = torch.cat([x[2] for x in batch_samples], dim=0)
        batch_attention_mask = torch.cat([x[3] for x in batch_samples], dim=0)

        # Forward pass and optimization
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Batch {i // batch_size}, Loss: {loss.item()}')


def train():
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.train()
    model.cuda()  # Move model to GPU if available

    # Load the dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_texts = dataset['train']['text']

    # Prepare the custom dataset and dataloader
    train_dataset = CustomDataset(train_texts, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Precompute gradient norms for all samples
    gradient_norms = precompute_gradient_norms(dataloader, model)

    # Sort samples by gradient norm and select the top-k samples
    top_k = 1000  # Select top-k samples with the highest gradient norms
    gradient_norms.sort(reverse=True, key=lambda x: x[0])
    selected_samples = gradient_norms[:top_k]

    # Train on selected samples
    train_on_selected_samples(selected_samples, model, optimizer)

    return selected_samples

selected_samples = train()

# Analyze selected samples
print("Analyzing selected samples based on gradient norm...")

data = [(grad_norm, text) for grad_norm, text, _, _ in selected_samples]
df = pd.DataFrame(data, columns=['Grad Norm', 'Text'])
print(df)

# Save DataFrame to a CSV file
df.to_csv('selected_samples.csv', index=False)