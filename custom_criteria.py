"""
custorm criteria
"""
import torch


def ssim(x, y, win_size=3, aggregate=True):
    """ Official implementation
    def SSIM(self, x, y):
        C1 = 0.01 ** 2 # why not use L=255
        C2 = 0.03 ** 2 # why not use L=255
        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')
        # if this implementatin equvalent to the SSIM paper?
        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2 
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    """
    avepooling2d = torch.nn.AvgPool2d(win_size, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    sigma_x = avepooling2d(x ** 2) - mu_x ** 2
    sigma_y = avepooling2d(y ** 2) - mu_y ** 2
    sigma_xy = avepooling2d(x * y) - mu_x * mu_y
    k1_square = 0.01 ** 2
    k2_square = 0.03 ** 2
    L_square = 1
    SSIM_n = (2 * mu_x * mu_y + k1_square * L_square) * (
        2 * sigma_xy + k2_square * L_square
    )
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + k1_square * L_square) * (
        sigma_x + sigma_y + k2_square * L_square
    )
    SSIM = SSIM_n / SSIM_d

    if aggregate:
        return torch.mean(SSIM)
    else:
        return SSIM


def dssim(x, y, win_size=11, aggregate=True):
    """
    SSIM Loss
    # TODO: add full docstrings
    """
    SSIM = ssim(x, y, win_size=win_size, aggregate=False)
    loss = torch.clamp((1 - SSIM) / 2, 0, 1)
    if aggregate:
        return torch.mean(loss)
    else:
        return loss
