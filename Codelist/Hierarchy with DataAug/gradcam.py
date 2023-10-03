"""https://github.com/vickyliin/gradcam_plus_plus-pytorch"""
import cv2
import numpy as np
import torch
import sys

from matplotlib import pyplot as plt

sys.path.append('E:/Work/BirdCLEF2017/')

layer_finders = {}
def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    # heatmap = mask.squeeze().type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha
    result = heatmap+img
    result = result.div(result.max()).squeeze()

    return heatmap, result

@register_layer_finder('CHRF')
def find_layer(arch, target_layer_name):
    """Find layer to calculate GradCAM
    Args:
        arch: XLHT models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'feature_embedding'
            target_layer_name = 'main_flow'
            target_layer_name = 'main_classifier'
            target_layer_name = 'transition_layer0'
            target_layer_name = 'transition_layer1'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    """
    if target_layer_name == 'feature_embedding':
        target_layer = arch.feature_embedding
    elif target_layer_name == 'main_flow':
        target_layer = arch.main_flow
    elif target_layer_name == 'main_classifier':
        target_layer = arch.main_classifier
    elif target_layer_name == 'transition_layer0':
        target_layer = arch.transition_layer[0]
    elif target_layer_name == 'transition_layer1':
        target_layer = arch.transition_layer[1]
    else:
        raise ValueError('unknown layer : {}'.format(target_layer_name))
    """
    if target_layer_name == 'feature_embedding':
        target_layer = arch.feature_embedding
    elif target_layer_name == 'class_branch':
        target_layer = arch.class_branch
    elif target_layer_name == 'genus_branch':
        target_layer = arch.genus_branch
    elif target_layer_name == 'family_branch':
        target_layer = arch.family_branch
    elif target_layer_name == 'order_branch':
        target_layer = arch.order_branch
    elif target_layer_name == 'class_neck':
        target_layer = arch.class_neck
    elif target_layer_name == 'genus_neck':
        target_layer = arch.genus_neck
    elif target_layer_name == 'family_neck':
        target_layer = arch.family_neck
    elif target_layer_name == 'order_neck':
        target_layer = arch.order_neck
    elif target_layer_name == 'class_classifyHead':
        target_layer = arch.class_classifyHead
    elif target_layer_name == 'genus_classifyHead':
        target_layer = arch.genus_classifyHead
    elif target_layer_name == 'family_classifyHead':
        target_layer = arch.family_classifyHead
    elif target_layer_name == 'order_classifyHead':
        target_layer = arch.order_classifyHead
    else:
        raise ValueError('unknown layer : {}'.format(target_layer_name))
    return target_layer

class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 1, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, hier=0, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        #logit, tm = self.model_arch(input)
        _, _, logit, _, _ = self.model_arch(input)
        logit = logit[hier]
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)

        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, hier=0, class_idx=None, retain_graph=False):
        return self.forward(input, hier, class_idx, retain_graph)

if __name__ == '__main__':
    from model_mod import *
    # initialize a model, model_dict and gradcam
    #model = XLHT(hierarchical_class=[100, 47, 18], use_attention=True)
    model = CHRF(hierarchy={'class': 150, 'genus': 122, 'family': 42, 'order': 14}, use_attention=True)
    checkpoint = torch.load(sys.path[-1] + 'Results/Hierarchy with DataAug/HASound_aug/ckpt/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    from utils import get_feature
    from config import *
    select_wave = ['E:/Work/BirdCLEF2017/SortedData/Song_22050\\1_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN26985.wav',
                 'E:/Work/BirdCLEF2017/SortedData/Song_22050\\3_LIFECLEF2017_BIRD_XC_WAV_RN43962.wav',
                 'E:/Work/BirdCLEF2017/SortedData/Song_22050\\1_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN12994.wav',
                 'E:/Work/BirdCLEF2017/SortedData/Song_22050\\5_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN11309.wav',
                 'E:/Work/BirdCLEF2017/SortedData/Song_22050\\1_LIFECLEF2017_BIRD_XC_WAV_RN39357.wav']
    select_img = ['E:/Work/BirdCLEF2017/SortedData/img\\1_LIFECLEF2015_BIRDAMAZON_XC_WAV_RN26985-0.png',
                   'E:/Work/BirdCLEF2017/SortedData/img\\3_LIFECLEF2017_BIRD_XC_WAV_RN43962-9.png',
                   'E:/Work/BirdCLEF2017/SortedData/img\\1_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN12994-0.png',
                   'E:/Work/BirdCLEF2017/SortedData/img\\5_LIFECLEF2014_BIRDAMAZON_XC_WAV_RN11309-0.png',
                   'E:/Work/BirdCLEF2017/SortedData/img\\1_LIFECLEF2017_BIRD_XC_WAV_RN39357-0.png']
    class_index=[[135, 27, 13, 9],[9, 27, 13, 9],[1, 47, 13, 9],[17, 111, 19, 9],[15, 119, 24, 13]]
    import librosa
    wave_list = []
    for i, path in enumerate(select_wave):
        wave_data, _ = librosa.load(path, 22050)
        if i == 1:
            wave_data = wave_data[45*22050: 50*22050]
        else:
            wave_data = wave_data[0: 5 * 22050]
        wave_list.append(wave_data)

    i = 4
    h = 0
    spec = get_feature(wave_list[i], 22050, frame_len=FRAME_LEN, win_step=1/4, n_mels=N_MELS)

    # spec1 = spec[0:64, ]
    # spec2 = spec[64:128, ]
    # spec3 = spec[128:192, ]
    # spec = np.stack((spec1, spec2, spec3))
    spec = np.expand_dims(spec, 0)  # (1,3,64,235)
    spec = torch.Tensor(spec)


    layer_name = ['class_branch','genus_branch','family_branch','order_branch']
    gradcam = GradCAM.from_config(model_type='CHRF', arch=model, layer_name=layer_name[h])

    mask, logit = gradcam(spec, hier=h, class_idx=class_index[i][h])
    # make heatmap from mask and synthesize saliency map using heatmap and img
    img_path = select_img[i]
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    from torchvision import transforms
    data_transform = transforms.Compose([transforms.Resize((N_MELS, 431)),
                                         transforms.ToTensor(),
                                         ])
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    heatmap, cam_result = visualize_cam(mask, img)
    heatmap = heatmap.permute(1, 2, 0)
    cam_result = cam_result.permute(1, 2, 0)
    plt.imshow(cam_result)
    plt.show()