import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import time
import sys
import pathlib
import random


sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


from networks.neural_net_generator import generate_network
from formula_factory import FormulaFactory

device = "cpu" if torch.cuda.is_available() else "cpu"


def model(s, a):

    dt = 0.05
    L = 1

    v = 2.5 * torch.tanh(0.5 * a[:, 0]) + 2.5
    gam = (torch.pi / 4) * torch.tanh(a[:, 1])

    f1 = s[:, 0] + (L / torch.tan(gam)) * (torch.sin(s[:, 2] + (v / L) * torch.tan(gam) * dt) - torch.sin(s[:, 2]))
    f2 = s[:, 1] + (L / torch.tan(gam)) * (-torch.cos(s[:, 2] + (v / L) * torch.tan(gam) * dt) + torch.cos(s[:, 2]))
    f3 = s[:, 2] + (v / L) * torch.tan(gam) * dt

    s_next = torch.stack([f1, f2, f3], dim = -1)

    return s_next

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 1
seed_everything(seed)

controller_hidden_size = 30
controller_net = nn.Sequential(
    nn.Linear(4, controller_hidden_size),
    nn.ReLU(),
    nn.Linear(controller_hidden_size, 2)
).to(device)

num_epochs = 100000
bs = 3
T = 40

def run_trajectory(initial_state, env_model, controller_net, T, bs):
    trajectory = []
    trajectory.append(initial_state)
    state = initial_state
    for t in range(0, T):
        Time = torch.zeros([bs, 1], dtype=torch.float32) + t
        Time = Time.to(device)
        sa = torch.cat([state, Time], dim=1)
        state = env_model( state, controller_net(sa) )
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)    

def generate_formula(args):
    
    FF = FormulaFactory(args)

    M = torch.zeros((12, 2))  # 12 rows, 2 columns
    c = torch.zeros((12, 1))  # 12 rows, 1 column

    M[0, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[0, 0] = torch.tensor([1], dtype=torch.float32)

    M[1, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[1, 0] = torch.tensor([-4], dtype=torch.float32)

    M[2, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[2, 0] = torch.tensor([-5], dtype=torch.float32)

    M[3, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[3, 0] = torch.tensor([2], dtype=torch.float32)

    M[4, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[4, 0] = torch.tensor([-3], dtype=torch.float32)

    M[5, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[5, 0] = torch.tensor([4], dtype=torch.float32)

    M[6, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[6, 0] = torch.tensor([0], dtype=torch.float32)

    M[7, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[7, 0] = torch.tensor([1], dtype=torch.float32)

    M[8, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[8, 0] = torch.tensor([-5], dtype=torch.float32)

    M[9, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[9, 0] = torch.tensor([6], dtype=torch.float32)

    M[10, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[10, 0] = torch.tensor([-3], dtype=torch.float32)

    M[11, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[11, 0] = torch.tensor([4], dtype=torch.float32)


    p1 = FF.LinearPredicate(M[0,:], c[0,0])
    p2 = FF.LinearPredicate(M[1,:], c[1,0])
    p3 = FF.LinearPredicate(M[2,:], c[2,0])
    p4 = FF.LinearPredicate(M[3,:], c[3,0])
    p5 = FF.LinearPredicate(M[4,:], c[4,0])
    p6 = FF.LinearPredicate(M[5,:], c[5,0])
    p7 = FF.LinearPredicate(M[6,:], c[6,0])
    p8 = FF.LinearPredicate(M[7,:], c[7,0])
    p9 = FF.LinearPredicate(M[8,:], c[8,0])
    p10 = FF.LinearPredicate(M[9,:], c[9,0])
    p11 = FF.LinearPredicate(M[10,:], c[10,0])
    p12 = FF.LinearPredicate(M[11,:], c[11,0])


    or1  = FF.Or([p1,p2,p3,p4])
    and2 = FF.And([p5,p6,p7,p8])
    and1 = FF.And([p9,p10,p11,p12])


    ordered = FF.Ordered(and1 , and2 , 0 , T)

    SFE = FF.G(  or1,   0,   T  )


    my_formula  =  FF.And( [ ordered , SFE ] )
    
    return my_formula

args = {'T': T+1, 'd_state': 2, 'Batch': bs, 'device': device, 'detailed_str_mode': False, 'approximation_beta': 10}
specification = generate_formula(args)

robustness_net_exact = generate_network(specification, approximate=False, beta=1.0).to(device)
robustness_net_exact.eval()

robustness_net_approximate = generate_network(specification, approximate=True, beta=10.0).to(device)
robustness_net_approximate.eval()

# Define the optimizer for f_network
optimizer = optim.Adam(controller_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

max_time_seconds = 3600
# Record the start time
start_time = time.time()
theta_set = torch.linspace(-6*torch.pi/8, -4*torch.pi/8, 10000)
theta_candidate_set = [-6*torch.pi/8, -5*torch.pi/8, -4*torch.pi/8]
assert len(theta_candidate_set) == 3, "theta_candidate_set should have 3 elements."
for epoch in range(num_epochs):
    theta = torch.tensor(theta_candidate_set)
    # theta = theta[torch.randint(0, len(theta_set), (bs,))]
        
    x_init, y_init = torch.zeros(bs) + 6, torch.zeros(bs) + 8
    init_state = torch.stack([x_init, y_init, theta], dim=1).to(device)
    
    trajectory = run_trajectory(init_state, model, controller_net, T, bs)
    objective_value = robustness_net_approximate(trajectory[:, :, :-1]).mean(dim=0)
    
    # Backward pass and optimization to maximize the objective function
    optimizer.zero_grad()
    (-objective_value).backward()
    optimizer.step()
    scheduler.step()

    # Print the objective value during training
    print(f'Epoch {epoch + 1}, Objective Value: {objective_value.item()}')
    
    if epoch % 10 == 0:
        
        bs_test = 100
        theta_random = torch.tensor(theta_set[torch.randint(0, len(theta_set), (bs_test,))])
        x_init_test, y_init_test = torch.zeros(bs_test) + 6, torch.zeros(bs_test) + 8
        init_state_test = torch.stack([x_init_test, y_init_test, theta_random], dim=1).to(device)
        
        with torch.no_grad():
            trajectory_test = run_trajectory(init_state_test, model, controller_net, T, bs_test)
            exact_robustness = robustness_net_exact(trajectory_test[:, :, :-1]).min(dim=0).values
            print(f'Epoch {epoch + 1}, Exact Robustness: {exact_robustness.item()} vs Approximate Robustness: {objective_value.item()}')
        if exact_robustness > 0:
            print("Found a robust solution. Breaking out of the loop.")
            break
    
    elapsed_time = time.time() - start_time
    if elapsed_time > max_time_seconds:
        print("Time limit exceeded the threshold. Breaking out of the loop.")
        break

elapsed_time = time.time() - start_time
print("Training completed, with time = ", elapsed_time, " seconds, epochs = ", epoch + 1)