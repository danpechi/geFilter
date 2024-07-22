import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Config
from datasets import load_dataset
import pandas as pd


from utils.data import CustomDataset

from utils.grad_norm import precompute_gradient_norms, compute_gradient_norm
from utils.conditional_entropy import conditional_entropy_selection


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


def train_obs():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model.train()
    model.cuda()

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_texts = dataset['train']['text']

    train_dataset = CustomDataset(train_texts, tokenizer)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    selection_fn = conditional_entropy_selection()

    all_selected_samples = []

    for epoch in range(1):  # Number of epochs
        for i, (texts, input_ids, attention_mask) in enumerate(dataloader):
            if input_ids.size(1) == 0:
                continue
            if input_ids.size(0) < 2:
                continue

            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            target = input_ids.cuda()

            selected_indices, metrics_to_log, _ = selection_fn(
                selected_batch_size=8,  # Number of samples to select
                data=input_ids,
                target=target,
                global_index=torch.arange(i * dataloader.batch_size, (i + 1) * dataloader.batch_size),
                large_model=model,
                num_mc=5,
                num_classes=model.config.vocab_size
            )

            selected_texts = [texts[idx] for idx in selected_indices]
            selected_input_ids = input_ids[selected_indices]
            selected_attention_mask = attention_mask[selected_indices]

            batch_input_ids = selected_input_ids.cuda()
            batch_attention_mask = selected_attention_mask.cuda()

            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_selected_samples.extend([(idx.item(), texts[idx]) for idx in selected_indices])
            del input_ids_chunk, attention_mask_chunk, target_chunk, selected_input_ids, selected_attention_mask
            torch.cuda.empty_cache()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    return all_selected_samples



selected_samples = train_obs()

print("Analyzing selected samples based on conditional entropy...")
data = [(idx, text) for idx, text in selected_samples]
df = pd.DataFrame(data, columns=['Index', 'Text'])
print(df)

csv_filename = 'selected_samples.csv'
df.to_csv(csv_filename, index=False)