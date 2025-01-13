import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, lag, hidden, horizon, dim=1, context=6):
        super(MLP, self).__init__()
        self.lag  = lag
        self.hidden = hidden
        self.horizon = horizon
        self.dim = dim
        self.context = context
        self.fc = nn.Sequential(
            nn.Linear(lag * dim + context, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * dim)
        )

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        batch_size = x.shape[0]
        input = x.view(batch_size, self.lag * self.dim) # (B, lag*dim)
        if  context is not None:
            input = torch.cat((input, context), dim=1) # (B, lag*dim+context)
        output = self.fc(input) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output