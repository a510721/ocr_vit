import torch
import torch.nn as nn
import math

class CustomCTCLoss(torch.nn.Module):
    # T x B x H => Softmax on dimension 2
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, logits, labels,
            prediction_sizes, target_sizes):
        EPS = 1e-7
        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)

    def sanitize(self, loss):
        EPS = 1e-7
        if abs(loss.item() - float('inf'))< EPS:
            return torch.zeros_like(loss)
        if math.isnan( loss.item() ):
            return torch.zeros_like(loss)
        return loss

    def debug(self, loss, logits, labels,
            prediction_sizes, target_sizes):
        if math.isnan(loss.item()):
            print("Loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained. But why?")
        return loss

if __name__ == '__main__':
    # Target are to be padded
    T = 50  # Input sequence length
    C = 20  # Number of classes (including blank)
    N = 16  # Batch size
    S = 30  # Target sequence length of longest target in batch (padding length)
    S_min = 10 # Minimum target length, for demonstration purposes

    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T,N,C).log_softmax(2).detach().requires_grad_()
    target = torch.randint(low=1, high=C,size=(N,S))
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(input, target, input_lengths, target_lengths)
    loss.backward()
