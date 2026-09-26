"""학습 손실. 평가 산식(패치별 1화소 허용 F-score)을 직접 근사합니다."""
import torch
import torch.nn.functional as F

IGNORE = 255


def tolerant_f_loss(prob, gt, valid=None, eps=1e-6):
    """prob, gt: (B,1,H,W). 산식의 dilate1 을 3x3 max-pool 로 근사.
    정답 양성 패치에서만 1 - F 를 평균합니다 (패치당 같은 가중치 = 산식과 같은 구조)."""
    if valid is not None:
        prob, gt = prob * valid, gt * valid
    gd = F.max_pool2d(gt, 3, 1, 1)
    pd = F.max_pool2d(prob, 3, 1, 1)
    prec = (prob * gd).sum((2, 3)) / (prob.sum((2, 3)) + eps)
    rec = (gt * pd).sum((2, 3)) / (gt.sum((2, 3)) + eps)
    f = 2 * prec * rec / (prec + rec + eps)
    pos = gt.sum((2, 3)) > 0
    return (1 - f)[pos].mean() if pos.any() else prob.sum() * 0


def debris_loss(seg_logits, cls_logit, target, w_f=1.0, w_cls=0.5):
    """target: (B,H,W) long, 0/1, IGNORE=무시.
    반환: (총손실, 항목별 dict)"""
    ce = F.cross_entropy(seg_logits, target, ignore_index=IGNORE)
    valid = (target != IGNORE).float().unsqueeze(1)
    gt = (target == 1).float().unsqueeze(1)
    prob = torch.softmax(seg_logits, dim=1)[:, 1:2]
    lf = tolerant_f_loss(prob, gt, valid)
    total = ce + w_f * lf
    parts = {"ce": ce.item(), "f": lf.item()}
    if cls_logit is not None:
        has_pos = gt.flatten(1).amax(1) > 0
        # 양성 화소가 없는데 무시 영역이 있는 패치는 존재 여부가 불확실하므로 분류 손실에서 뺍니다
        certain = has_pos | (valid.flatten(1).amin(1) > 0)
        if certain.any():
            lc = F.binary_cross_entropy_with_logits(cls_logit[certain], has_pos[certain].float())
            total = total + w_cls * lc
            parts["cls"] = lc.item()
    return total, parts
