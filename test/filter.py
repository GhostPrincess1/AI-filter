import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import json

# 生成数据（替换为你的数据）
#np.random.seed(42)
#data = 258*np.random.normal(size=100) + 0# np.linspace(0, 10, 100)
data = [0,2, 1, 0, 3,4,5,6,9,100,250]
for i in range(len(data)):
    data[i] = int(data[i])
print(data)
x = np.arange(len(data))

# 计算B样条曲线
tck, u = splprep([x, data], s=10000,k=2)
new_x, new_data = splev(np.linspace(0, 1, len(data)), tck)

for i in range(len(new_data)):
    new_data[i] = int(new_data[i])

print(new_data)
# 绘制原始和平滑后的曲线
plt.plot(x, data, label='Original Data')
plt.plot(new_x, new_data, label='B-spline Smoothed Data', linewidth=2)
plt.legend()
plt.show()


