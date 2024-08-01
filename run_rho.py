import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd


# Custom Dataset class to process text data
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                   return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return text, input_ids, attention_mask


# Function to compute loss
def compute_loss(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    return loss


# Train small GPT model on holdout set
def train_small_gpt(holdout_loader, tokenizer):
    small_model = GPT2LMHeadModel.from_pretrained('gpt2')
    small_model.train()
    small_model.cuda()

    optimizer = AdamW(small_model.parameters(), lr=5e-5)
    for epoch in range(1):  # Number of epochs for small model
        for texts, input_ids, attention_mask in holdout_loader:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            loss = compute_loss(small_model, input_ids, attention_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return small_model


# Function to compute irreducible holdout loss
def compute_irreducible_loss(small_model, holdout_texts, tokenizer):
    small_model.eval()
    irreducible_loss = []
    with torch.no_grad():
        for text in holdout_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            loss = compute_loss(small_model, input_ids, attention_mask)
            irreducible_loss.append(loss.item())
    return torch.tensor(irreducible_loss)


# Function to get top x indices
def top_x_indices(tensor, x, largest=True):
    """Select the top x indices from the tensor"""
    x = min(x, tensor.size(0))  # Ensure x does not exceed tensor size
    indices = torch.topk(tensor, x, largest=largest).indices  # Top x indices
    not_selected = torch.topk(tensor, tensor.size(0) - x, largest=not largest).indices  # Remaining indices
    return indices, not_selected


# RHO-LOSS batch selection and training function
def train_rho_loss():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.train()
    model.cuda()

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    texts = dataset['train']['text']

    # Split the dataset into training and holdout sets
    train_size = int(0.8 * len(texts))
    holdout_size = len(texts) - train_size
    train_texts, holdout_texts = random_split(texts, [train_size, holdout_size])

    # Create DataLoaders for training and holdout sets
    train_dataset = CustomDataset(train_texts, tokenizer)
    holdout_dataset = CustomDataset(holdout_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=True)

    # Train the small GPT model on the holdout set
    small_model = train_small_gpt(holdout_loader, tokenizer)

    # Compute irreducible holdout loss for all samples in the holdout set
    irreducible_loss = compute_irreducible_loss(small_model, holdout_texts, tokenizer)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    all_selected_samples = []

    for epoch in range(1):  # Number of epochs
        for i, (texts, input_ids, attention_mask) in enumerate(train_loader):
            if input_ids.size(1) == 0:
                continue

            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            target = input_ids.cuda()

            # Step 1: Randomly select a large batch
            large_batch_size = 64
            large_batch_indices = torch.randperm(len(train_dataset))[:large_batch_size]
            large_batch = [train_dataset[idx] for idx in large_batch_indices]
            large_texts, large_input_ids, large_attention_mask = zip(*large_batch)
            large_input_ids = torch.stack(large_input_ids).cuda()
            large_attention_mask = torch.stack(large_attention_mask).cuda()

            # Step 2: Compute training loss for each sample in the large batch
            losses = []
            for j in range(large_batch_size):
                loss = compute_loss(model, large_input_ids[j].unsqueeze(0), large_attention_mask[j].unsqueeze(0))
                losses.append(loss.item())
            losses = torch.tensor(losses)

            # Step 3: Compute RHO-LOSS for each sample
            rho_loss = losses - irreducible_loss[large_batch_indices]

            # Step 4: Select top-nb samples based on RHO-LOSS
            selected_batch_size = 8
            selected_indices = top_x_indices(rho_loss, selected_batch_size, largest=True)[0]
            selected_input_ids = large_input_ids[selected_indices]
            selected_attention_mask = large_attention_mask[selected_indices]
            selected_texts = [large_texts[idx] for idx in selected_indices]

            # Step 5: Perform mini-batch gradient descent
            outputs = model(selected_input_ids, attention_mask=selected_attention_mask, labels=selected_input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_selected_samples.extend(
                [(large_batch_indices[idx].item(), selected_texts[idx]) for idx in selected_indices])

            torch.cuda.empty_cache()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    # Save selected samples to CSV
    print("Analyzing selected samples based on RHO-LOSS...")
    data = [(idx, text) for idx, text in all_selected_samples]
    df = pd.DataFrame(data, columns=['Index', 'Text'])
    print(df)

    csv_filename = 'selected_samples.csv'
    df.to_csv(csv_filename, index=False)


# Run the training function
train_rho_loss()