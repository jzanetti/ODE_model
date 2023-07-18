import torch
import torch.nn as nn
import yaml


class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()

        self.params = params
        self.device = device
        # self.num_agents = self.params["num_agents"]  # Population


class SIRS(ODE):
    def __init__(self, params, device):
        super().__init__(params, device)

    def init_compartments(self, initial_infections_percentage, num_agents):
        """let's get initial conditions"""
        initial_conditions = torch.empty((2)).to(self.device)
        no_infected = (initial_infections_percentage / 100) * num_agents  # 1.0 is ILI
        initial_conditions[1] = no_infected
        initial_conditions[0] = num_agents - no_infected
        print("initial infected", no_infected)

        self.state = initial_conditions

    def step(self, t, cfg):
        """
        Computes ODE states via equations
            state is the array of state value (S,I)
        """
        params = {"beta": cfg["fixed"]["contact rate"]}
        # set from expertise
        params["D"] = 3.5
        params["L"] = 2000
        num_agents = cfg["fixed"]["num_agents"]

        if t == 0:
            self.init_compartments(cfg["learnable"]["initial_infections_percentage"], num_agents)
        dS = (num_agents - self.state[0] - self.state[1]) / params["L"] - params[
            "beta"
        ] * self.state[0] * self.state[1] / num_agents
        dSI = params["beta"] * self.state[0] * self.state[1] / num_agents
        dI = dSI - self.state[1] / params["D"]

        # concat and reshape to make it rows as obs, cols as states
        self.dstate = torch.stack([dS, dI], 0)

        NEW_INFECTIONS_TODAY = dSI
        # ILI is percentage of outpatients with influenza-like illness
        # ILI = params['lambda'] * dSI / self.num_agents
        # this is what Shaman and Pei do https://github.com/SenPei-CU/Multi-Pathogen_ILI_Forecast/blob/master/code/SIRS_AH.m
        ILI = dSI / num_agents * 100  # multiply 100 because it is percentage

        # update state
        self.state = self.state + self.dstate
        return NEW_INFECTIONS_TODAY, ILI
