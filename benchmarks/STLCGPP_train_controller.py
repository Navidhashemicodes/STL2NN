import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import time
import sys
import pathlib
import random
import stlcgpp.formula as stlcg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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

def generate_CG(T):
    def compute_box_dist_x(x):
        return torch.abs(x[..., 0] - 5.5)

    def compute_box_dist_y(x):
        return torch.abs(x[..., 1] - 3.5)

    def compute_box_dist_x2(x):
        return torch.abs(x[..., 0] - 3.5)

    def compute_box_dist_y2(x):
        return torch.abs(x[..., 1] - 0.5)

    def compute_box_dist_x3(x):
        return torch.abs(x[..., 0] - 2.5)

    def compute_box_dist_y3(x):
        return torch.abs(x[..., 1] - 3.5)

    def goal_1():
        dx = stlcg.Predicate("box_dist_x", predicate_function = compute_box_dist_x)
        dy = stlcg.Predicate("box_dist_y", predicate_function = compute_box_dist_y)
    
        within = stlcg.And(dx <= 0.5, dy <= 0.5)
    
        return within

    def goal_2():
        dx = stlcg.Predicate("box_dist_x2", predicate_function = compute_box_dist_x2)
        dy = stlcg.Predicate("box_dist_y2", predicate_function = compute_box_dist_y2)
    
        within = stlcg.And(dx <= 0.5, dy <= 0.5)
        return within

    
    def safe():
        dx = stlcg.Predicate("box_dist_x3", predicate_function = compute_box_dist_x3)
        dy = stlcg.Predicate("box_dist_y3", predicate_function = compute_box_dist_y3)
    
        outside = stlcg.Or(dx > 1.5, dy > 1.5)
        return outside


    def always_safe(T):
        safe_formula = safe()
        return stlcg.Always(safe_formula, interval = [0,T])

    def eventually_goal1_then_eventually_goal2(T):
        goal2 = goal_2()
        goal1 = goal_1()
    
        for i in range(0, T):
        
            eventually_goal2 = stlcg.Eventually(goal2, interval=[i+1, T])
            eventually_goal1 = stlcg.Eventually(goal1, interval=[i, i])
            if i == 0:
                subformulas = stlcg.And(eventually_goal1  , eventually_goal2)
            elif i > 0:
                subformula = stlcg.And(eventually_goal1, eventually_goal2)
                subformulas = stlcg.Or(subformulas,subformula)
    
        return subformulas

    def build_formula(T):
        formula1 = eventually_goal1_then_eventually_goal2(T)
        formula2 = always_safe(T)
        formula = stlcg.And(formula1, formula2)
        return formula.robustness

    return build_formula(T)

specification = generate_CG(T)

robustness_func_exact = torch.vmap(specification)
approx_method = "logsumexp"  # or "softmax"
robustness_func_approximate = torch.vmap(lambda x: specification(x, approx_method=approx_method, temperature=10.0))

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
    objective_value = robustness_func_approximate(trajectory[:, :, :-1]).mean(dim=0)
    
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
            exact_robustness = robustness_func_exact(trajectory_test[:, :, :-1]).min(dim=0).values
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