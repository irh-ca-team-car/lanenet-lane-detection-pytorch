import time
import os
import sys

import torch
from .dataloader.transformers import Rescale
from .model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from .model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = "src/apollo_python/apollo_python/lanenet/data/tusimple_test_image/3.jpg"
    print(img_path)
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(
        "src/apollo_python/apollo_python/lanenet/"+model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(
        outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255

    left = np.logical_and(np.logical_and(
        instance_pred[0, :, :] > 225, instance_pred[1, :, :] < 225), instance_pred[2, :, :] > 225).astype(np.uint8) * 255
    right = np.logical_and(np.logical_and(
        instance_pred[0, :, :] < 225, instance_pred[1, :, :] < 225), instance_pred[2, :, :] > 225).astype(np.uint8) * 255

    def extract(left):
        contours, _ = cv2.findContours(
            left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        line = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                line = cnt

        #line = np.array(sorted(line, key=lambda x: dist(1,1,x[0], x[1])))
        line = torch.from_numpy(line)
        line = line.squeeze(1)
        line = pd.DataFrame({"x": line[:, 0], "y": line[:, 1]})
        line = line.sort_values("y")
        skip = int(len(line)/10)
        line_filtered = line.iloc[::skip]

        pts = []
        for index, pt in line_filtered.iterrows():
            pts_in_row = line[line["y"] == pt["y"]]
            pts.append([int(pts_in_row["x"].mean()), pt["y"]])
        return pts
    # print(line)
    left_pts = extract(left)
    right_pts = extract(right)
    left[:] = 0
    right[:] = 0
    for x, y in left_pts:
        left[y, x] = 255
    for x, y in right_pts:
        right[y, x] = 255

    distance = 3.7

    class mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.height = torch.nn.Parameter(torch.tensor(1.9))
            self.hfov = torch.nn.Parameter(torch.tensor(150.))
            self.vfov = torch.nn.Parameter(torch.tensor(70.))

            self.ln1 = torch.nn.Linear(2, 2)
            self.ln2 = torch.nn.Linear(2, 2)
            self.ln3 = torch.nn.Linear(2, 2)
            pass

        def layers(self, x):
            x = self.ln1(x)
            x = torch.nn.functional.sigmoid(x)
            x = self.ln2(x)
            x = torch.nn.functional.sigmoid(x)
            x = self.ln3(x)
            x = torch.nn.functional.sigmoid(x)
            return x

    params = mdl()

    def _2dTo3d(_2d, img):
        import math
        h, w = img.shape
        height = torch.tensor(1.9)  # params.height#1.9m to ground

        y = _2d[1]
        half_height = h/4.0
        val = (y - half_height)/h
        if val < 0:
            val = 0

        angle = (params.vfov/2.0)*val
        distance = height / (torch.tan(torch.deg2rad(angle)) + 0.1e-7)

        x = _2d[0]
        half_width = w/2.0
        val = (x - half_width)/w
        angle = (params.hfov/2.0)*val
        _3dx = (distance) * torch.sin(torch.deg2rad(angle))
        _3dy = (distance) * torch.cos(torch.deg2rad(angle))
        _3dz = -height

        return torch.cat([_3dx.view(1), _3dy.view(1), _3dz.view(1)], 0)
    def ajust(left,right):
        for a,b in zip(left,right):
            center = (a[0] + b[0])/2
            a[0] = center - distance/2
            b[0] = center + distance/2
    def loss(left, right) -> torch.Tensor:
        #error = torch.cat([(torch.sqrt(torch.pow(a[0]-b[0],2)+torch.pow(a[1]-b[1],2))).view(1) for a,b in zip(left,right)],0)
        error = torch.cat([(a[0]-b[0]).view(1)
                          for a, b in zip(left, right)], 0)
        zeros = torch.ones(error.shape) * distance
        return torch.nn.functional.mse_loss(error, zeros)
    optim = torch.optim.Adamax(params.parameters(), lr=2e-2)

    for i in range(1):
        def a():
            optim.zero_grad()

            left_3d = [_2dTo3d(pt, left) for pt in left_pts]
            right_3d = [_2dTo3d(pt, right) for pt in right_pts]
            loss_ = loss(left_3d, right_3d)
            loss_.backward()
            return loss_

        optim.step(a)
        left_3d = [_2dTo3d(pt, left) for pt in left_pts]
        right_3d = [_2dTo3d(pt, right) for pt in right_pts]
        loss_ = loss(left_3d, right_3d)
        print(i, loss_, params.height.data.item(),
              params.hfov.data.item(), params.vfov.data.item())

    ajust(left_3d,right_3d)
    left_3d = pd.DataFrame(left_3d)
    right_3d = pd.DataFrame(right_3d)

    import matplotlib
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt

    plt.scatter(left_3d[0], left_3d[1])
    plt.scatter(right_3d[0], right_3d[1])
    plt.show()
    #binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255
    # cv2.imwrite(os.path.join('test_output', 'input.bmp'), input)
    # cv2.imwrite(os.path.join('test_output', 'instance_output.bmp'), instance_pred.transpose((1, 2, 0)))
    # cv2.imwrite(os.path.join('test_output', 'instance_output_left.bmp'), left)
    # cv2.imwrite(os.path.join('test_output', 'instance_output_right.bmp'), right)
    # cv2.imwrite(os.path.join('test_output', 'binary_output.bmp'), binary_pred)


if __name__ == "__main__":
    test()
