import torch
import time
import sys
import pathlib
import stlcgpp.formula as stlcg
import numpy as np
from tqdm.auto import tqdm

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

device = torch.device("cpu")
Times = []
from functools import partial


BUILD_FORMULA_TIMES = []
ROBUSTNESS_TIMES = []

import time

for i in tqdm(range(0,15)):
   
    T = 5*(i+1)
    bs = 1
    Epochs = 1000
    
    begin_time = time.perf_counter()
    formula = build_formula(T)
    end_time = time.perf_counter()
    BUILD_FORMULA_TIMES.append((end_time - begin_time))
    

        
    begin_time = time.perf_counter()
    for j in range(0,Epochs):
        trajectory = torch.randn( bs, T+1, 2).to(device)
        objective_value = torch.vmap(formula)(trajectory)
    end_time = time.perf_counter()
    
    ROBUSTNESS_TIMES.append((end_time - begin_time) / Epochs)
    print(f"Formula building time: {BUILD_FORMULA_TIMES[-1]:.4f} seconds")
    print(f"Robustness time: {ROBUSTNESS_TIMES[-1]:.4f} seconds")
    
    

import matplotlib.pyplot as plt

plt.plot(BUILD_FORMULA_TIMES, label='Formula Building Time')
plt.show()
plt.plot(ROBUSTNESS_TIMES, label='Robustness Time')
plt.show()

import os
save_path = 'results/'
os.makedirs(save_path, exist_ok=True)
torch.save([BUILD_FORMULA_TIMES, ROBUSTNESS_TIMES], save_path + 'STLCGpp_CPU_Epochs.pt')