import math
import torch.nn.functional as F
import torch


def kl_divergence(p, q, mask=None):
    """
    Compute the KL Divergence between two distributions with an optional mask.

    Args:
        p (Tensor): Logits from the model. Shape: [batch_size, fea_dim]
        q (Tensor): Logits from the prototype. Shape: [batch_size, fea_dim] or [fea_dim]
        mask (Tensor, optional): Mask indicating which samples to include. Shape: [batch_size]

    Returns:
        Tensor: Scalar representing the masked KL Divergence loss.
    """
    # If q lacks a batch dimension, expand it to match p
    if q.dim() == p.dim() - 1:
        q = q.unsqueeze(0).expand_as(p)

    # Apply log_softmax to p and softmax to q to get probability distributions
    p_log_softmax = F.log_softmax(p, dim=-1)  # Shape: [batch_size, fea_dim]
    q_softmax = F.softmax(q, dim=-1)          # Shape: [batch_size, fea_dim]

    # Compute KL Divergence for each sample without reduction
    kl_div = F.kl_div(p_log_softmax, q_softmax, reduction='none').sum(dim=-1)  # Shape: [batch_size]

    if mask is not None:
        # Apply the mask to include only relevant samples
        kl_div = kl_div * mask  # Shape: [batch_size]
        # Compute the mean KL Divergence, avoiding division by zero
        kl_div = kl_div.sum() / (mask.sum() + 1e-8)
    else:
        # If no mask is provided, compute the mean over the batch
        kl_div = kl_div.mean()

    return kl_div

def orthogonal_loss(A, B, mask=None):
    """
    Compute the Orthogonal Loss between two sets of vectors with an optional mask.

    Args:
        A (Tensor): Feature vectors. Shape: [batch_size, fea_dim]
        B (Tensor): Prototype vectors. Shape: [batch_size, fea_dim]
        mask (Tensor, optional): Mask indicating which samples to include. Shape: [batch_size]

    Returns:
        Tensor: Scalar representing the masked Orthogonal Loss.
    """
    A = F.normalize(A, p=2, dim=1)  # Shape: [batch_size, fea_dim]
    B = F.normalize(B, p=2, dim=1)  # Shape: [batch_size, fea_dim]
    # Calculate the dot product between corresponding vectors in A and B
    dot_product = torch.sum(A * B, dim=1)  # Shape: [batch_size]

    # Calculate the square of the dot products
    dot_product_sq = dot_product ** 2      # Shape: [batch_size]
    if mask is not None:
        # Apply the mask to include only relevant samples
        dot_product_sq = dot_product_sq * mask  # Shape: [batch_size]
        # Compute the mean Orthogonal Loss, avoiding division by zero
        loss = dot_product_sq.sum() / (mask.sum() + 1e-8)
    else:
        # If no mask is provided, compute the mean over the batch
        loss = dot_product_sq.mean()

    return loss
    # return 0.0


def l2_loss_fn(p, q, mask=None):
    """
    Compute the L2 loss between two distributions with an optional mask.

    Args:
        p (Tensor): Logits from the model. Shape: [batch_size, fea_dim]
        q (Tensor): Logits from the prototype. Shape: [batch_size, fea_dim] or [fea_dim]
        mask (Tensor, optional): Mask indicating which samples to include. Shape: [batch_size]

    Returns:
        Tensor: Scalar representing the masked L2 loss.
    """
    # If q lacks a batch dimension, expand it to match p
    if q.dim() == p.dim() - 1:
        q = q.unsqueeze(0).expand_as(p)
    p = F.normalize(p, p=2, dim=1)
    q = F.normalize(q, p=2, dim=1)
    # Apply softmax to p and q to get probability distributions
    # p_softmax = F.softmax(p, dim=-1)  # Shape: [batch_size, fea_dim]
    # q_softmax = F.softmax(q, dim=-1)  # Shape: [batch_size, fea_dim]
    
    # Compute squared differences for each sample
    # l2 = (p_softmax - q_softmax).pow(2).sum(dim=-1)  # Shape: [batch_size]
    l2 = (p - q).pow(2).sum(dim=-1)  # Shape: [batch_size]
    
    if mask is not None:
        # Apply the mask to include only relevant samples
        l2 = l2 * mask  # Shape: [batch_size]
        # Compute the mean L2 loss, avoiding division by zero
        l2 = l2.sum() / (mask.sum() + 1e-8)
    else:
        # If no mask is provided, compute the mean over the batch
        l2 = l2.mean()
    
    return l2

def InfoNCE_loss_fn(anchor, positive, negative, label):
    temperature = 0.07
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)   
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = -torch.log(torch.exp(pos_dist / temperature) / (torch.exp(pos_dist / temperature) + torch.exp(neg_dist / temperature)))
    return loss


def infoNCE_align_loss(anchor, positive, temperature=0.07):

    anchor = F.normalize(anchor, p=2, dim=1)  # [batch_size, fea_dim]
    positive = F.normalize(positive, p=2, dim=1)  # [batch_size, fea_dim]
    
    batch_size = anchor.size(0)
    
    # anchor @ positive.T: [batch_size, batch_size]
    sim_matrix = torch.matmul(anchor, positive.transpose(-1, -2)) / temperature
    
    pos_sim = torch.diag(sim_matrix)  # [batch_size]
    
    exp_sim = torch.exp(sim_matrix)  # [batch_size, batch_size]
    sum_exp_sim = torch.sum(exp_sim, dim=1)  # [batch_size]
    
    loss = -torch.log(torch.exp(pos_sim) / (sum_exp_sim + 1e-8))
    
    return loss.mean()
