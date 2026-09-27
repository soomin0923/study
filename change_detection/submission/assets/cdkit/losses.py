"""학습 손실: CE + 클래스별 1화소 허용 F 손실(산식 근사) + 분류 헤드 BCE."""
import torch
import torch.nn.functional as F

IGNORE = 255


def tolerant_f_loss(prob, gt, valid=None, eps=1e-6):
    """prob, gt: (B,1,H,W). 정답 양성 쌍에서만 1-F 평균 (쌍당 같은 가중치 = 산식 구조)."""
    if valid is not None:
        prob, gt = prob * valid, gt * valid
    gd, pd = F.max_pool2d(gt, 3, 1, 1), F.max_pool2d(prob, 3, 1, 1)
    prec = (prob * gd).sum((2, 3)) / (prob.sum((2, 3)) + eps)
    rec = (gt * pd).sum((2, 3)) / (gt.sum((2, 3)) + eps)
    f = 2 * prec * rec / (prec + rec + eps)
    pos = gt.sum((2, 3)) > 0
    return (1 - f)[pos].mean() if pos.any() else prob.sum() * 0


def cd_loss(seg_logits, cls_logit, target, w_f=1.0, w_cls=0.5, class_weights=None):
    """target: (B,H,W) long, 0 배경 / 1 증축 / 2 벌목 / IGNORE."""
    ce = F.cross_entropy(seg_logits, target, weight=class_weights, ignore_index=IGNORE)
    valid = (target != IGNORE).float().unsqueeze(1)
    prob = torch.softmax(seg_logits, dim=1)
    parts, total = {"ce": ce.item()}, ce
    for k, name in ((1, "f_bld"), (2, "f_tree")):
        gt = (target == k).float().unsqueeze(1)
        lf = tolerant_f_loss(prob[:, k:k + 1], gt, valid)
        total = total + 0.5 * w_f * lf
        parts[name] = lf.item()
    if cls_logit is not None:
        has = torch.stack([(target == 1).flatten(1).any(1), (target == 2).flatten(1).any(1)], 1).float()
        certain = (valid.flatten(1).amin(1) > 0).unsqueeze(1) | (has > 0)
        if certain.any():
            lc = F.binary_cross_entropy_with_logits(cls_logit[certain], has[certain])
            total = total + w_cls * lc
            parts["cls"] = lc.item()
    return total, parts
