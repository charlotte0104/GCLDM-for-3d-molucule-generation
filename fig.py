import torch
import matplotlib.pyplot as plt

# 创建一个包含 100000 个元素的张量，假设使用标准正态分布
tensor = torch.load('/tmp/pycharm_project_80/data_for_distribution/data_Cv.pth')
tensor2 = torch.load('/tmp/pycharm_project_80/data_for_distribution/sample_Cv.pth')
tensor2=tensor2.to('cpu')
# 绘制概率分布的直方图
#30,110for alpha       -2,10for mu   -45,0forCv
plt.xlim(-45,0)
plt.hist(tensor.numpy(), bins=100, density=True, alpha=0.6, color='b')
plt.hist(tensor2.detach().numpy(), bins=50, density=True, alpha=0.6, color='r')
# 设置图表标题和标签
# plt.title('Probability Distribution Histogram')
plt.xlabel('Cv')
# plt.ylabel('Density')
plt.tight_layout()
# 显示图表
plt.show()