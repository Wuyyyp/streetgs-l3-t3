import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def draw_cost(cost,vis = False,method=cv2.COLORMAP_HOT):
    cost_n = cost
    cost_n[cost_n > 0.35] = (cost_n[cost_n >0.35] - 0.35) / (2-0.35) * 0.1 + 0.9
    cost_n[cost_n <= 0.35] = (cost_n[cost_n <= 0.35]) / (0.35) * 0.7
    cost_n = (cost_n * 255).astype("uint8")
    # cost_n = adjust_gamma(cost_n, gamma=1.5)
    colored = cv2.applyColorMap(cost_n, method)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_pred(pred,vis=False,method=cv2.COLORMAP_HOT):
    pred = (pred * 255).astype("uint8")
    colored = cv2.applyColorMap(pred, method)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

# def draw_binary(pred,vis=False):
#     pred = (pred * 255).astype("uint8")
#     colored = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
#     colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
#     if vis:
#         plt.imshow(colored)
#         plt.show()
#     return colored

def draw_depth(depmap,vis = False,method=cv2.COLORMAP_MAGMA):
    depmap_n = 255 - ((depmap - depmap.min()) / (depmap.max() - depmap.min()) * 255).astype("uint8")
    colored = cv2.applyColorMap(depmap_n, method)  # bone OCEAN
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if vis:
        plt.imshow(colored)
        plt.show()
    return colored