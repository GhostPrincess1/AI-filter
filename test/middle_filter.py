import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 生成示例数据
x = np.linspace(0, 10, num=100)
data = np.random.normal(size=(100, 2))

# 计算 B 样条曲线的参数表示
tck, u = splprep(data.T, s=100)

# 计算平滑曲线上的点
new_x = np.linspace(0, 1, len(data))
new_data = np.column_stack(splev(new_x, tck))

# 绘制原始数据和平滑曲线
plt.scatter(data[:, 0], data[:, 1], label='原始数据', alpha=0.5)
plt.plot(new_data[:, 0], new_data[:, 1], label='平滑曲线', color='red')
plt.legend()
plt.show()
