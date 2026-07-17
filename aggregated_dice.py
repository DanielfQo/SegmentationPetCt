# aggregated_dice.py
import torch

def aggregated_dice(all_preds, all_labels, num_fg=2):
    """
    Dice agregado: sum(2*|A∩B|) / sum(|A|+|B|) sobre TODO el conjunto, por clase.
    all_preds: lista de tensores [D, H, W]
    all_labels: lista de tensores [D, H, W]
    """
    inter = torch.zeros(num_fg)
    denom = torch.zeros(num_fg)
    for p, l in zip(all_preds, all_labels):
        for c in range(1, num_fg + 1):
            pc, lc = (p == c), (l == c)
            inter[c-1] += (pc & lc).sum()
            denom[c-1] += pc.sum() + lc.sum()
    return (2 * inter / denom.clamp(min=1)).tolist()   # [tumor, ganglios]
