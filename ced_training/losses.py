"""Float32 losses; frozen vocabulary remains in the readout gradient path."""
import torch
import torch.nn.functional as F


def readout_losses(student_logits, teacher_logits, labels, temperature=1.0):
    ce = F.cross_entropy(student_logits.float(), labels)
    log_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_teacher = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    kd = F.kl_div(log_student, log_teacher, reduction="batchmean", log_target=True) * temperature**2
    return ce, kd


def normalized_mse(prediction, target, energy):
    # Fixed training-only target energy, not a batch-dependent denominator.
    return (prediction.float() - target.detach().float()).square().mean() / max(float(energy), 1e-8)


def gate_loss(logits, labels):
    # No class reweighting: keep the posterior interpretation for calibration.
    return F.binary_cross_entropy_with_logits(logits.float().reshape(-1), labels.float().reshape(-1))
