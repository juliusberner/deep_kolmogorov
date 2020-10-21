import math
import time
import torch
from torch import nn

EPSILON = 1e-08
NORMLAYERS = {
    "layernorm": torch.nn.LayerNorm,
    "batchnorm": nn.BatchNorm1d,
    "none": nn.Identity,
}


class BaseNet(torch.nn.Module):
    """
    Base class for different networks.
    """

    def __init__(self, dim_in, config):
        super().__init__()
        self.dim_in = dim_in
        self.config = config
        self.params_groups = [{"params": self.parameters()}]
        self.active_groups = []

    def unfreeze_only_active(self):
        for group in self.params_groups:
            for param in group["params"]:
                if group in self.active_groups:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def update_active_groups(self, iteration):
        idx = iteration // self.config["unfreeze_patience"]
        if idx < len(self.params_groups):
            if self.config["unfreeze"] == "single":
                self.active_groups = [self.params_groups[idx]]
            elif self.config["unfreeze"] == "sequential":
                self.active_groups = self.params_groups[: idx + 1]
            else:
                self.active_groups = self.params_groups
        else:
            self.active_groups = self.params_groups

    def decay_lr(self, iteration):
        if not (iteration + 1) % self.config["lr_decay_patience"]:
            for params_group in self.active_groups:
                if params_group["lr"] > self.config["min_lr"]:
                    params_group["lr"] *= self.config["lr_decay"]

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class LevelNet(nn.Module):
    """
    Network module for a single level.
    """

    def __init__(self, dim_in, dim, level, norm_layer):
        super().__init__()
        self.level = level
        self.dense_layers = nn.ModuleList([nn.Linear(dim_in, dim, bias=False)])
        self.dense_layers += [
            nn.Linear(dim, dim, bias=False) for _ in range(2 ** level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim, 1))
        self.norm_layers = nn.ModuleList(
            [NORMLAYERS[norm_layer](dim, eps=EPSILON) for _ in range(2 ** level)]
        )
        self.act = nn.ReLU()

    def forward(self, tensor, res_tensors=None):
        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):
            tensor = self.norm_layers[i](tensor)
            tensor = self.act(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(BaseNet):
    """
    Multilevel net.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)
        dim = self.config["factor"] * self.dim_in
        self.nets = nn.ModuleList(
            [
                LevelNet(self.dim_in, dim, level, config["norm_layer"])
                for level in range(self.config["levels"])
            ]
        )
        self.params_groups = [{"params": net.parameters()} for net in self.nets]

    def forward(self, tensor):
        res_tensors = None
        for net in self.nets[::-1]:
            res_tensors = net(tensor, res_tensors)
        return res_tensors[-1]


class MultilevelNetNoRes(MultilevelNet):
    """
    Multilevel net without residual connections.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)

    def forward(self, tensor):
        output = self.nets[0](tensor)[-1]
        for net in self.nets[1:]:
            output += net(tensor)[-1]
        return output


class Feedforward(BaseNet):
    """
    Feedforward net.
    """

    def __init__(self, dim_in, config):
        super().__init__(dim_in, config)
        dim = self.config["factor"] * self.dim_in
        self.net = LevelNet(self.dim_in, dim, config["levels"], config["norm_layer"])

    def forward(self, tensor):
        return self.net(tensor)[-1]


NETS = {net.__name__: net for net in BaseNet.get_subclasses()}


class KolmogorovNet(torch.nn.Module):
    """
    DL Kolmogorov model.
    """

    def __init__(self, net, pde):
        super().__init__()
        self.net = net
        self.pde = pde

    def forward(self, batch, train=True):
        with torch.no_grad():
            if train:
                y = self.pde.sde(batch)
            else:
                y = self.pde.solution(batch)
            tensor = self.pde.normalize_and_flatten(batch)
        y_pred = self.net.forward(tensor)
        return {"pde": y, "net": y_pred}


class Metrics:
    """
    Returns the metrics for our trainer.
    """

    names = ["mse", "L2^2", "mae", "L1"]

    def __init__(self):
        self.best = {name: 1.0e10 for name in self.names}
        self.last_improve = {name: 0 for name in self.names}
        self.t = 0.0
        self.steps = 0.0
        self._running = {metric: 0 for metric in self.names}
        self._count = 0
        self._current_t = time.time()

    def store(self, output, return_loss=None):
        abs_error = (output["pde"] - output["net"]).abs()
        magnitude = output["pde"].abs() + 1
        rel_error = abs_error / magnitude
        loss = {
            "mse": (abs_error ** 2).mean(),
            "L2^2": (rel_error ** 2).mean(),
            "mae": abs_error.mean(),
            "L1": rel_error.mean(),
        }
        for name in self.names:
            self._running[name] += loss[name].item()
        self._count += 1
        if return_loss:
            return loss[return_loss]

    def zero(self):
        self._running = {metric: 0 for metric in self._running}
        self._count = 0
        self._current_t = time.time()

    def finalize(self):
        current_t = time.time() - self._current_t
        self.t += current_t
        self.steps += self._count
        current = {
            "time": current_t,
            "steps": self._count,
            "overall time": self.t,
            "overall steps": self.steps,
        }
        current.update(
            {name: metr / self._count for name, metr in self._running.items()}
        )
        for name in self.names:
            if current[name] < self.best[name]:
                self.best[name] = current[name]
                self.last_improve[name] = 0
            else:
                self.last_improve[name] += 1
        current["L2"] = math.sqrt(current["L2^2"])
        return {
            "current": current,
            "best": self.best,
            "last improve": self.last_improve,
        }
