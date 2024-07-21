def compute_gradient_norm(model, input_ids, attention_mask):
    model.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def precompute_gradient_norms(dataloader, model):
    gradient_norms = []
    total_samples = len(dataloader)
    for i, (text, input_ids, attention_mask) in enumerate(dataloader):
        if input_ids.size(1) == 0:  # Skip empty input_ids
            continue

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        grad_norm = compute_gradient_norm(model, input_ids, attention_mask)
        gradient_norms.append((grad_norm, text, input_ids, attention_mask))

        if i % 100 == 0:
            percent_complete = (i / total_samples) * 100
            print(f"Progress: {percent_complete:.2f}%")
    return gradient_norms
