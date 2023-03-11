import torch 
import torch.nn.functional as F    

def supConLoss(features, labels):
    features = F.normalize(features, p =2, dim=-1)
    assert features.shape[0] == labels.shape[0]
    dot_product_features = torch.mm(features, features.T)
    exp_dot_tempered = (
            torch.exp(dot_product_features - torch.max(dot_product_features, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
    labels = labels.contiguous().view(-1, 1)
    mask_combined = torch.eq(labels, labels.T).float()
    cardinality_per_samples = torch.sum(mask_combined, dim=1)
    log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
    supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
    supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
    return supervised_contrastive_loss
    