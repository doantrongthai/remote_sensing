import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAMSegmentation:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        target_module = dict(self.model.named_modules()).get(self.target_layer_name)
        if target_module is None:
            for name, mod in self.model.named_modules():
                if isinstance(mod, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                    target_module = mod
                    break
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.hook_handles.append(target_module.register_forward_hook(forward_hook))
        self.hook_handles.append(target_module.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def generate_cam(self, input_tensor, target_mask=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        
        if target_mask is None:
            pred_water = (probs[:, 1] > 0.5)
            if pred_water.sum() > 0:
                target_score = (logits[:, 1] * pred_water.float()).sum()
            else:
                target_score = logits[:, 1].sum()
        else:
            target_score = (logits[:, 1] * target_mask.float()).sum()
            
        target_score.backward(retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze().detach().cpu().numpy(), logits.detach()

def sanity_check_randomized_weights(model_class, model_kwargs, trained_model, input_tensor, target_layer_name):
    cam_generator = GradCAMSegmentation(trained_model, target_layer_name)
    cam_trained, _ = cam_generator.generate_cam(input_tensor)
    cam_generator.remove_hooks()
    
    random_model = model_class(**model_kwargs).to(input_tensor.device)
    cam_rand_gen = GradCAMSegmentation(random_model, target_layer_name)
    cam_random, _ = cam_rand_gen.generate_cam(input_tensor)
    cam_rand_gen.remove_hooks()
    
    diff = np.mean(np.abs(cam_trained - cam_random))
    is_sensitive = diff > 0.05
    return {
        'trained_cam': cam_trained,
        'random_cam': cam_random,
        'mean_absolute_difference': float(diff),
        'passes_randomization_sanity_check': bool(is_sensitive)
    }

def save_cam_figure(cam, original_image, ground_truth, prediction, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    if original_image.ndim == 3 and original_image.shape[0] in [2, 13, 15]:
        rgb = original_image[:3] if original_image.shape[0] >= 3 else np.stack([original_image[0], original_image[1], (original_image[0]+original_image[1])/2])
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
    else:
        rgb = original_image
        
    axes[0].imshow(rgb)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(ground_truth, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    axes[3].imshow(rgb)
    axes[3].imshow(cam, cmap='jet', alpha=0.5)
    axes[3].set_title("Grad-CAM Explanation")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
