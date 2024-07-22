import torch
import torch.nn.functional as F

def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def compute_conditional_entropy(log_probs):
    """Compute the conditional entropy given log probabilities"""
    probs = torch.exp(log_probs)
    conditional_entropy = -torch.sum(probs * log_probs, dim=-1)
    return conditional_entropy.mean(dim=0)

def top_x_indices(tensor, x, largest=True):
    x = min(x, tensor.size(0))  # Ensure x does not exceed tensor size
    indices = torch.topk(tensor, x, largest=largest).indices
    not_selected = torch.topk(tensor, tensor.size(0) - x, largest=not largest).indices
    return indices, not_selected

class conditional_entropy_selection:
    bald = True

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        num_mc=5,
        num_classes=10,
        *args,
        **kwargs,
    ):
        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
              output = large_model(data)
              logits = output.logits  # Shape: [batch_size, seq_len, vocab_size]
              avg_logits = logits.mean(dim=1)  # Average over sequence length
              predictions[i] = avg_logits
              log_probs[i] = F.log_softmax(avg_logits, dim=-1)
            conditional_entropy = compute_conditional_entropy(log_probs.transpose(0, 1))
            selected_minibatch, not_selected_minibatch = top_x_indices(
                conditional_entropy, selected_batch_size, largest=True
            )

        metrics_to_log = {"detailed_only_keys": []}

        return selected_minibatch, metrics_to_log, None