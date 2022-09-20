import matplotlib.pyplot as plt
import os
import numpy as np
import h5py as h5

"""此代码读取真值与预测值的 dose numpy 并生成相应的png图像，是剂量矩阵横断面的可视化。"""

x_dim = 1024
y_dim = 1024
pred_path = "/Users/mr.chai/Desktop/499339_pred1024.npy"
true_path = "/Users/mr.chai/Desktop/499339_truedose.npy"
output_path = "/Users/mr.chai/Desktop/499399_fig3"
if not os.path.exists(output_path):
    os.mkdir(output_path)
p1title = "true"
p2title = "pred"
p3title = "minus"



true = np.load(true_path)
print("true dose shape [{:}]".format(true.shape))
print("true dose max is [{:}]".format(np.max(true)))
z_dim = true.shape[2]

pred = np.load(pred_path)
pred = pred.reshape((x_dim, y_dim, true.shape[2]))
print("pred dose shape [{:}]".format(pred.shape))
print("pred dose max is [{:}]".format(np.max(pred)))

minus = true - pred
print("minus dose shape [{:}]".format(minus.shape))
print("minus dose max is [{:}]".format(np.max(minus)))

for z in range(z_dim//10):
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(true[256:768, 256:768, 10*z + 5], cmap="jet", vmin= 0, vmax=80)
    # plt.imshow(true[128:384, 128:384, 10*z + 5], cmap="jet", vmin= 0, vmax=80)

    ax1.set_title(p1title, font="Arial", fontsize = 25, pad = 20, fontweight = "bold")

    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(pred[256:768, 256:768, 10*z + 5], cmap="jet", vmin= 0, vmax=80)
    # plt.imshow(pred[128:384, 128:384, 10*z + 5], cmap="jet", vmin= 0, vmax=80)

    ax2.set_title(p2title, font="Arial", fontsize = 25, pad = 20, fontweight = "bold")


    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(minus[256:768, 256:768, 10*z + 5], cmap="Blues", vmin= 0, vmax=20)
    # plt.imshow(minus[128:384, 128:384, 10*z + 5], cmap="Blues", vmin= 0, vmax=20)

    ax3.set_title(p3title, font="Arial", fontsize = 25, pad = 20, fontweight = "bold")

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Gy', y=0, ha='left', rotation='-40')

    # plt.show()
    plt.savefig(output_path + os.sep + str(z) + ".jpg", dpi = 1000, bbox_inches = 'tight')
    print("[{:}] slice finished!".format(z))
