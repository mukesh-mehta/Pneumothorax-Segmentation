def dice_coeff(output, target, smooth=0, eps=1e-7):
    return (2 * sum(output * target) + smooth) / (
            sum(output) + sum(target) + smooth + eps)
