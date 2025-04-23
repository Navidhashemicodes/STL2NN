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

from networks.neural_net_generator import generate_network
from formula_factory import FormulaFactory

device = "cpu" if torch.cuda.is_available() else "cpu"

bs = 100
T = 40

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

args = {'T': T+1, 'd_state': 2, 'Batch': bs, 'device': device, 'detailed_str_mode': False, 'approximation_beta': 1}
specification = generate_formula(args)

robustness_net_exact = generate_network(specification, approximate=False, beta=1.0).to(device)
robustness_net_exact.eval()


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

specification_cg = generate_CG(T)

robustness_func_exact = torch.vmap(specification_cg)


random_trajectory = torch.randn((bs, T+1, 2), device=device, requires_grad=True)

r1 = robustness_func_exact(random_trajectory)
r2 = robustness_net_exact(random_trajectory)
r3 = specification.evaluate(random_trajectory)[0]
print(torch.abs(r1 - r2).mean())
print(torch.abs(r3 - r2).mean())

