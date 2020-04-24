# Modules
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
from torchvision import transforms

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Pretrained DeepLabV3
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()
model.to(device)

# Preprocess image
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Start camera capture
capture = cv2.VideoCapture(0)
while(True):

    # Capture video frame
    _, frame = capture.read()

    # Convert frame to tensor
    frame_tensor = preprocess(frame).unsqueeze(0).to(device)

    # Predict image segmentation
    with torch.no_grad():
        output = model(frame_tensor)['out'][0].argmax(0)

    # Group classes into human or background    
    output[output != 15] = 0
    output[output == 15] = 1

    # Resize output to frame shape
    output = output.byte().cpu().numpy()
    output = np.stack((output, output, output), -1)
    output = cv2.resize(output, frame.shape[1::-1]).astype(bool)

    # Create human and background masks
    human = (frame * output).astype(float)
    background = frame * np.invert(output)

    # Apply transparent overlay to human class
    overlay = output * np.array([[255, 0, 0]])
    human = 0.66 * human + 0.33 * overlay

    # Display frame with overlay
    cv2.imshow('frame', human.astype('uint8') + background.astype('uint8'))

    # Exit with q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera capture
capture.release()
cv2.destroyAllWindows()
