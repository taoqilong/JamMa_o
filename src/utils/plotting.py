import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch
from matplotlib import cm
import matplotlib.patheffects as path_effects
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #
def make_matching_figure_color(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k'
    text_ = fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)
    text_.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        return fig


def make_evaluation_figure_color(data, b_id, alpha='dynamic', path=None, dpi=150):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    R_errs = data['R_errs'][b_id][0]
    t_errs = data['t_errs'][b_id][0]

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    runtime = data['runtime']
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R_errs: {R_errs:.1f}',
        f't_errs: {t_errs:.1f}',
        f'runtime: {runtime:.1f}',
    ]

    # make the figure
    figure = make_matching_figure_color(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return figure


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    color = vis[v.astype(np.int32), u.astype(np.int32)]
    return color

def draw_kp(img, kps, colors):
    for i, kp in enumerate(kps):
        img = cv2.circle(img, (int(kp[1]), int(kp[0])), 3, colors[i].tolist(), -1)
    return img


def vis_matches(image0, image1, kp0, kp1):
    lh, lw = image0.shape[:2]
    rh, rw = image1.shape[:2]
    mask1 = np.logical_and.reduce(np.array((kp0[:,1]>=0, kp0[:,1]<lw, kp0[:,0]>=0, kp0[:,0]<lh)))
    mask2 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<rw, kp1[:,0]>=0, kp1[:,0]<rh)))

    mask = np.logical_and.reduce(np.array((mask1, mask2)))
    kp0 = kp0[mask]
    kp1 = kp1[mask]

    color = kp_color(kp0[:,1], kp0[:,0], (lh, lw))

    image0 = draw_kp(image0, kp0, color)
    image1 = draw_kp(image1, kp1, color)

    pad_width = 5
    zero_image = np.zeros([lh, pad_width, 3])
    vis = np.concatenate([image0, zero_image, image1], axis=1)

    return vis


def make_evaluation_figure_wheel(data, b_id=0, path=None, topk=10000):
    b_mask = data['m_bids'] == b_id

    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.permute(1, 2, 0).detach().cpu().numpy() * 255, img1.permute(1, 2, 0).detach().cpu().numpy() * 255
    img0, img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img0, img1 = img0.round().astype(np.int32), img1.round().astype(np.int32)
    img0 = np.ascontiguousarray(img0)
    img1 = np.ascontiguousarray(img1)

    kpts0 = data['mkpts0_f'][b_mask]
    kpts1 = data['mkpts1_f'][b_mask]
    mconf = data['mconf_f'][b_mask]

    num = len(mconf) if len(mconf) < topk else topk
    idx = torch.topk(mconf, num, 0).indices
    kpts0 = kpts0[idx]
    kpts1 = kpts1[idx]

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id][[0, 1]]
        kpts1 = kpts1 / data['scale1'][b_id][[0, 1]]

    # make the figure
    kpts_wh_0 = torch.flip(kpts0, [1]).cpu().numpy()
    kpts_wh_1 = torch.flip(kpts1, [1]).cpu().numpy()
    figure = vis_matches(img0, img1, kpts_wh_0, kpts_wh_1)
    cv2.imwrite(path, figure)
    return figure


def make_confidence_figure(data, b_id=0, path=None, dpi=150, topk=10000):
    img0 = data['imagec_0'][b_id]
    img1 = data['imagec_1'][b_id]
    img0 = img0 * (imagenet_std[:, None, None].to(img0.device)) + (imagenet_mean[:, None, None].to(img0.device))
    img1 = img1 * (imagenet_std[:, None, None].to(img1.device)) + (imagenet_mean[:, None, None].to(img1.device))
    img0, img1 = img0.detach().permute(1, 2, 0).cpu().numpy(), img1.detach().permute(1, 2, 0).cpu().numpy()
    img0, img1 = np.clip(img0, 0.0, 1.0), np.clip(img1, 0.0, 1.0)

    num = len(data['mconf_f']) if len(data['mconf_f']) < topk else topk
    idx = torch.topk(data['mconf_f'], num, 0).indices
    kpts0 = data['mkpts0_f'][idx].detach().cpu().numpy()
    kpts1 = data['mkpts1_f'][idx].detach().cpu().numpy()
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
    score = data['mconf_f'][idx].cpu().numpy()
    color = cm.jet(score)

    text = [
        f'#Matches {len(kpts0)}',
    ]
    # make the figure
    fig = make_matching_figure_color(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return fig


def make_matching_figures(data, mode='evaluation', path=None, dpi=150):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['confidence', 'evaluation', 'wheel']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['imagec_0'].size(0)):
        if mode == 'confidence':
            fig = make_confidence_figure(data, b_id, dpi=dpi, path=path)
        elif mode == 'evaluation':
            fig = make_evaluation_figure_color(data, b_id, dpi=dpi, path=path)
        elif mode == 'wheel':
            fig = make_evaluation_figure_wheel(data, b_id, path=path)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
