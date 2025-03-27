import torch
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
)
from model import Model

# Set to GPU or CPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

feat_model_name = 'x3d_l'
feat_model = torch.hub.load('facebookresearch/pytorchvideo', feat_model_name, pretrained=True)
feat_model = feat_model.eval()
feat_model = feat_model.to(device)
del feat_model.blocks[-1]

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# Get transform parameters based on model
transform_params = model_transform_params[feat_model_name]

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            Lambda(lambda x: x/255.0),
            Normalize(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCrop((transform_params["crop_size"],transform_params["crop_size"])),
            Lambda(lambda x: x.permute((1, 0, 2, 3)))
        ]
    ),
)

model = Model().to(device)
model.load_state_dict(torch.load('saved_models/888tiny.pkl', map_location=device))
model.eval()

if __name__ == '__main__':
    # Input shape: 16x3x224x224
    # 16: số frame (không phải 32 như Uniformer32)
    # 3x224x224: Channel x Width x Height: channel=3; width và height tuỳ ý, cứ giữ nguyên khi đọc từ video
    t = torch.rand((16, 3, 224, 320)).to(device)
    
    t_transform = transform({'video' : t})['video']
    with torch.no_grad():
        feature = feat_model(t_transform.unsqueeze(0))
        logits, _ = model(feature)
        output = torch.sigmoid(logits)
        print(f'Anomaly probability: {output}')