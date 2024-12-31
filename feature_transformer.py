import torch
import math

def transform_CAT(content_feature, style_feature, alpha=0.5):
    """
    A WCT function can be used directly between encoder and decoder
    """
    cf = content_feature.squeeze(0)  # .double()
    sf = style_feature.squeeze(0)  # .double()
    c, ch, cw = cf.shape
    s, sh, sw = sf.shape
    cf = cf.reshape(c, -1)
    sf = sf.reshape(c, -1)
    c_mean = torch.mean(cf, 1, keepdim=True)
    s_mean = torch.mean(sf, 1, keepdim=True)

    c_std = torch.sum((cf - c_mean) ** 2, 1, keepdim=True) ** 0.5
    s_std = torch.sum((sf - s_mean) ** 2, 1, keepdim=True) ** 0.5

    cf_m = cf - c_mean
    sf_m = sf - s_mean
    c_cov = torch.mm(cf_m, cf_m.t()).div(ch*cw - 1)
    s_cov = torch.mm(sf_m, sf_m.t()).div(sh*sw - 1)
    diag_W = (s_std / c_std) ** 0.5
    diag_W[:,:] = 0
    for i in range(diag_W.shape[0]):
        diag_W[i] = abs(s_cov[i,i])**(0.5)/(abs(c_cov[i,i])+1e-5)**0.5#*

    diag_W = torch.diag(diag_W.squeeze(-1))
    final_cf = torch.mm(diag_W, cf_m)#
    final_cf = final_cf+s_mean
    colored_feature = final_cf.reshape(c, ch, cw).unsqueeze(0).float()
    return colored_feature

