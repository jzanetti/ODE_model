import matplotlib.pyplot as plt
import torch

from model import SIRS

device = torch.device(f"cuda:0")

cfg = {
    "fixed": {"num_agents": 200, "contact rate": 0.5},
    "learnable": {"initial_infections_percentage": 20},
}

# "SIRS" stands for Susceptible, Infected, Recovered, and Susceptible again
my_model = SIRS(cfg, device)

new_infections = []
for t in range(100):
    new_infection, _ = my_model.step(t, cfg)
    new_infections.append(new_infection.item())

plt.plot(new_infections)
plt.savefig("test.png")
plt.close()
