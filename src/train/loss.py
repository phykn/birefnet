import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


def masked_mean(value, valid):
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


def erode_valid(valid, radius):
    return 1.0 - F.max_pool2d(1.0 - valid, 2 * radius + 1, 1, radius)


def resize_classes(target, size):
    if target.shape[-2:] == size:
        return target
    resized = F.interpolate(target[:, None].float(), size, mode="nearest-exact")
    return resized[:, 0].long()


def make_onehot(target, classes):
    safe = target.masked_fill(target == 255, 0)
    return F.one_hot(safe, classes).permute(0, 3, 1, 2).float()


class TrainLoss:
    def __init__(self, cfg):
        self.classes = cfg.model.classes
        self.loss = cfg.loss

    def __call__(self, out, batch, teacher=None, teacher_weight=0.0):
        target = batch["mask"]
        label = batch.get("label")
        valid = (target[:, None] != 255).float()
        if label is not None:
            valid = valid * (label[:, None] == 255)
        cut = batch.get("cut")
        ce = dice = 0.0
        last = len(out.logits) - 1
        for index, pred in enumerate(out.logits):
            pred_ce, pred_dice, boundary = self.compute_mask_loss(
                pred, target, valid, cut, index == last, teacher, teacher_weight
            )
            ce = ce + pred_ce / len(out.logits)
            dice = dice + pred_dice / len(out.logits)
        c = self.loss
        mask_loss = c.ce * ce + c.dice * dice
        aux = boundary.new_zeros(())
        if out.gdt is not None and c.aux:
            aux = self.compute_guidance_loss(out.gdt, valid)
        loss = c.boundary * boundary + c.aux * aux
        parts = {"ce": ce, "dice": dice, "boundary": boundary, "aux": aux}
        if label is None:
            loss = loss + mask_loss
        else:
            parts["label_ce"] = self.compute_label_loss(out.logits, label)
            loss = loss + c.label_weight * parts["label_ce"] + c.mask_weight * mask_loss
        parts["loss"] = loss
        return parts, loss

    def compute_mask_loss(self, pred, target, valid, cut, last, teacher, weight):
        pred = pred.float()
        size = pred.shape[-2:]
        blur = self.loss.label_blur * size[-1] / target.shape[-1]
        target = resize_classes(target, size)
        if valid.shape[-2:] != size:
            valid = F.interpolate(valid, size, mode="nearest-exact")
        onehot = make_onehot(target, self.classes)
        soft = onehot
        if blur > 0:
            side = 2 * int(round(3 * blur)) + 1
            window, sigma = (side, side), (blur, blur)
            mass = gaussian_blur2d(valid, window, sigma, border_type="constant")
            soft = gaussian_blur2d(soft * valid, window, sigma, border_type="constant")
            soft = soft / mass.clamp_min(1e-6)
        if teacher is not None and weight > 0:
            if teacher.shape[-2:] != size:
                teacher = F.interpolate(teacher, size, mode="area")
            soft = soft * (1.0 - weight) + teacher.detach().float() * weight
        pred = pred.masked_fill(valid == 0, 0)
        log_prob = pred.log_softmax(1)
        q = self.loss.gce_q
        if q > 0:
            ce = (soft * (1 - (q * log_prob).exp())).sum(1, keepdim=True) / q
        else:
            ce = -(soft * log_prob).sum(1, keepdim=True)
        prob = log_prob.exp()
        overlap = (prob * soft * valid).sum((0, 2, 3))
        total = ((prob + soft) * valid).sum((0, 2, 3))
        dice = (1.0 - (2 * overlap + 1e-6) / (total + 1e-6)).mean()
        boundary = pred.new_zeros(())
        if last and self.loss.boundary > 0:
            r = self.loss.boundary_width
            k = 2 * r + 1
            padded = F.pad(onehot, (r,) * 4, mode="replicate")
            edge = F.max_pool2d(padded, k, 1) + F.max_pool2d(-padded, k, 1) > 0
            active = edge.amax(1, keepdim=True).float() * erode_valid(valid, r)
            if cut is not None:
                cut = F.interpolate(cut.float(), size, mode="nearest-exact")
                active = active * (1 - F.max_pool2d(cut, k, 1, r))
            boundary = masked_mean(ce, active)
        return masked_mean(ce, valid), dice, boundary

    def compute_label_loss(self, logits, label):
        smooth = self.loss.label_smooth
        total = 0.0
        for pred in logits:
            pred = pred.float()
            target = resize_classes(label, pred.shape[-2:])
            painted = target != 255
            onehot = make_onehot(target, self.classes)
            soft = onehot * (1 - smooth) + smooth / self.classes
            ce = -(soft * pred.log_softmax(1)).sum(1)
            floor = 0.005 * target.numel()
            total = total + (ce * painted).sum() / max(float(painted.sum()), floor)
        return total / len(logits)

    def compute_guidance_loss(self, gdt, valid):
        preds, labels = gdt
        total = 0.0
        for pred, label in zip(preds, labels):
            size = label.shape[-2:]
            if pred.shape[-2:] != size:
                pred = F.interpolate(pred, size, mode="bilinear", align_corners=False)
            mask = F.interpolate(valid, size, mode="nearest-exact")
            mask = erode_valid(mask, radius=2)
            pred = pred.masked_fill(mask == 0, 0)
            label = label.detach().masked_fill(mask == 0, 0)
            bce = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
            total = total + masked_mean(bce, mask)
        return total / len(preds)
