import torch
load_path = 'results/'

build_time_stlcgpp, robustness_time_stlcgpp= torch.load(load_path + 'STLCGpp_GPU_Batched.pt')
build_time_lb4tl, robustness_time_lb4tl = torch.load(load_path + 'LB4TL_GPU_Batched.pt')

import matplotlib.pyplot as plt
T = 30
x_ticks = list(range(5*(1+1), 5*(30+1+1), 5)) + list(range(5*(35+1) , 5*(50+1+1), 25)) 

plt.plot(x_ticks, build_time_stlcgpp[::-1], label='STLCGpp Formula Building Time')
plt.plot(x_ticks, build_time_lb4tl[::-1], label='LB4TL Formula Building Time')
plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Formula Building Time Comparison')
plt.legend()
plt.grid()
plt.show()

plt.plot(x_ticks, robustness_time_stlcgpp[::-1], label='STLCGpp Robustness Time')
plt.plot(x_ticks, robustness_time_lb4tl[::-1], label='LB4TL Robustness Time')
plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Robustness Time Comparison')
plt.legend()
plt.grid()
plt.show()

