from turtle import forward
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, T=0.5, eps=1e-8):
        super().__init__()
        self.T = T
        self.eps = eps  # epsilon value added to avoid numerical instability

    def forward(self, x, y):
        representations = x
        label = y
        T = self.T
        n = label.shape[0]  # batch size

        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix.cuda()

        # Clip similarity matrix to avoid numerical instability
        similarity_matrix = torch.clamp(
            similarity_matrix, min=-1.0 + self.eps, max=1.0 - self.eps)

        # Create mask for matching labels
        mask = torch.ones_like(similarity_matrix) * \
            (label.expand(n, n).eq(label.expand(n, n).t()))
        mask = mask.cuda()

        # Create inverse mask and diagonal mask
        inverse_mask = torch.ones_like(mask) - mask
        diagonal_mask = torch.ones(n, n) - torch.eye(n, n)
        diagonal_mask = diagonal_mask.cuda()

        # Calculate similarity matrix with temperature scaling
        similarity_matrix = torch.exp(similarity_matrix / T)

        # Prevent overflow in exponential calculation
        similarity_matrix = torch.clamp(similarity_matrix, max=1e10)

        similarity_matrix = similarity_matrix * diagonal_mask

        # Split into similar and dissimilar components
        similar_pairs = mask * similarity_matrix
        dissimilar_pairs = similarity_matrix - similar_pairs

        # Calculate sum of dissimilar pairs
        dissimilar_sum = torch.sum(dissimilar_pairs, dim=1)

        # Expand for matrix computation
        dissimilar_sum_expanded = dissimilar_sum.repeat(n, 1).T
        similarity_sum = similar_pairs + dissimilar_sum_expanded

        # Divide, adding epsilon to prevent division by zero
        loss = torch.div(similar_pairs, similarity_sum + self.eps)

        # Add inverse mask and identity matrix
        loss = inverse_mask + loss + torch.eye(n, n).cuda()

        # Prevent NaN in log calculation by ensuring values > 0
        loss = torch.clamp(loss, min=self.eps)

        # Calculate final loss
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss

# loss_func = ContrastiveLoss()
