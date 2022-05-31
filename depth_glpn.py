import torch
from data.datasets import get_dataset
import matplotlib.pyplot as plt
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation


data = get_dataset('coco', 128, 'test')

image, label, bbox = data[5000]


# load models
feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# extract features
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

# predict depth
with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth

print(predicted_depth.shape)

# interpolate to original size
prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=pixel_values.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
             )
prediction = prediction.squeeze().cpu().numpy()

print(prediction.shape)


# visualize
_, axs = plt.subplots(1, 2)
axs[0].imshow(image.permute(1,2,0) *  0.5 + 0.5)
axs[1].imshow(prediction, cmap='jet')
plt.show()