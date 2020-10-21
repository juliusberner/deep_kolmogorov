import math
from abc import ABC, abstractmethod
from operator import mul
from functools import reduce
import torch
from torch.utils.data import Dataset, DataLoader


class Hypercube:
    """
    Hypercube for sampling of the input data.
    """

    def __init__(self, interval, dims=(1,)):
        self.interval = interval
        self.dims = dims

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, value):
        if not len(value) == 2:
            raise ValueError(f"interval {value} must be of the form [a, b]")
        self.__interval = value

    @property
    def dims(self):
        return self.__dims

    @dims.setter
    def dims(self, value):
        if not isinstance(value, tuple):
            raise TypeError(f"dims {value} must be a tuple")
        self.__dims = value

    @property
    def mean(self):
        return sum(self.__interval) / 2

    @property
    def std(self):
        return (self.__interval[1] - self.__interval[0]) / math.sqrt(12)

    @property
    def dim_flat(self):
        return reduce(mul, self.__dims)

    def sample(self, batch_size):
        return torch.FloatTensor(batch_size, *self.__dims).uniform_(*self.__interval)

    def __repr__(self):
        return f'hypercube {self.__interval}^({"x".join(map(str,self.__dims))})'


class Data(Dataset):
    """
    Uniformly distributed input data as a PyTorch (infinite) dataset.
    """

    def __init__(self, hypercubes, batch_size, n_batches):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.hypercubes = hypercubes

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return {
            key: cube.sample(self.batch_size) for key, cube in self.hypercubes.items()
        }


class Pde(ABC):
    """
    Base class for different parametrized PDEs.
    """

    def __init__(self, hypercubes):
        super().__init__()
        self.hypercubes = hypercubes

    @property
    def hypercubes(self):
        return self.__hypercubes

    @hypercubes.setter
    def hypercubes(self, value):
        if not (
            isinstance(value, dict)
            and all(isinstance(cube, Hypercube) for cube in value.values())
        ):
            raise TypeError(f"{value} must be a dictionary consisting of hypercubes")
        if not all(param in value for param in self.params):
            raise ValueError(f"{value} must have keys {self.params}.")
        if not self._check_dims(value):
            raise ValueError("hypercube dimensions are not matching.")
        self.__hypercubes = value

    @property
    def dim_flat(self):
        return sum([cube.dim_flat for cube in self.__hypercubes.values()])

    def dataloader(self, batch_size, n_batches):
        return DataLoader(
            Data(self.__hypercubes, batch_size, n_batches), batch_size=None
        )

    def normalize_and_flatten(self, batch):
        batch = [
            (batch[param] - self.__hypercubes[param].mean) / self.hypercubes[param].std
            for param in self.params
        ]
        return torch.cat([tensor.flatten(start_dim=1) for tensor in batch], dim=1)

    @property
    @abstractmethod
    def params(self):
        pass

    @staticmethod
    @abstractmethod
    def _check_dims(hypercubes):
        pass

    @staticmethod
    @abstractmethod
    def sde(batch):
        pass

    @staticmethod
    @abstractmethod
    def solution(batch):
        pass

    def __repr__(self):
        return f"Parametrized {self.__class__.__name__} PDE with hypercubes {self.__hypercubes}"

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


HYPERCUBES = {
    f"basket_{d_basket}d": {
        "t": Hypercube(interval=[0.0, 1.0]),
        "x": Hypercube(interval=[9.0, 10.0], dims=(d_basket,)),
        "sigma": Hypercube(
            interval=[0.1, 0.6], dims=(d_basket, d_basket, d_basket + 1)
        ),
        "mu": Hypercube(interval=[0.1, 0.6], dims=(d_basket, d_basket + 1)),
        "K": Hypercube(interval=[10.0, 12.0]),
    }
    for d_basket in range(1, 6)
}


class Basket(Pde):
    params = ("t", "x", "sigma", "mu", "K")

    def __init__(self, hypercubes=HYPERCUBES["basket_3d"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        d = hypercubes["x"].dims[0]
        return all(
            [
                hypercubes["t"].dims == (1,),
                hypercubes["x"].dims == (d,),
                hypercubes["sigma"].dims == (d, d, d + 1),
                hypercubes["mu"].dims == (d, d + 1),
                hypercubes["K"].dims == (1,),
            ]
        )

    @staticmethod
    def sde(batch, steps=25):
        """
        Outputs batched realizations of the SDE.
        """
        batch_size, d = batch["x"].shape
        steplen = (batch["t"] / steps).flatten()
        std = torch.sqrt(steplen)
        outputs = batch["x"].clone()
        for _ in range(steps):
            dw = (
                torch.randn(
                    d, batch_size, dtype=batch["x"].dtype, device=batch["x"].device
                )
                * std
            )
            sigma_x = (
                torch.einsum("iklj, il -> ikj", batch["sigma"][:, :, :, :d], outputs)
                + batch["sigma"][:, :, :, d]
            )
            mu_x = (
                torch.einsum("ikj, ij -> ik", batch["mu"][:, :, :d], outputs)
                + batch["mu"][:, :, d]
            )
            outputs += torch.einsum("ij, i -> ij", mu_x, steplen) + torch.einsum(
                "ijk, ki -> ij", sigma_x, dw
            )
        return torch.nn.ReLU()(batch["K"] - outputs.mean(dim=1, keepdims=True))

    @staticmethod
    def solution(batch, steps=25, mc_rounds=1048576):
        """
        Outputs the MC approximated solution.
        """
        ys = []
        for t, x, sigma, mu, K in zip(
            batch["t"], batch["x"], batch["sigma"], batch["mu"], batch["K"]
        ):
            mu_t = mu[:, :-1].T
            steplen = t / steps
            std = torch.sqrt(steplen)
            outputs = x.expand(mc_rounds, -1).clone()
            for _ in range(steps):
                dw = (
                    torch.randn(mc_rounds, len(x), dtype=x.dtype, device=x.device) * std
                )
                sigma_x = (
                    torch.einsum("ijk, lj -> lik", sigma[:, :, :-1], outputs)
                    + sigma[:, :, -1]
                )
                mu_x = outputs @ mu_t + mu[:, -1]
                outputs += mu_x * steplen + torch.einsum("ijk, ik -> ij", sigma_x, dw)
            y = (torch.nn.ReLU()(K - outputs.mean(dim=1, keepdims=True))).mean(
                dim=0, keepdims=True
            )
            ys.append(y)
        return torch.cat(ys, dim=0)


def n_dist(x):
    """
    Cumulative distribution function of the standard normal distribution.
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def n_density(x):
    """
    Density function of the standard normal distribution.
    """
    return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2.0 * math.pi)


HYPERCUBES["black_scholes"] = {
    "t": Hypercube(interval=[0.0, 1.0]),
    "x": Hypercube(interval=[9.0, 10.0]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "K": Hypercube(interval=[10.0, 12.0]),
}


class BlackScholes(Pde):
    params = ("t", "x", "sigma", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    @staticmethod
    def sde(batch):
        """
        Outputs batched realizations of the SDE.
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        sde = batch["x"] * torch.exp(
            -0.5 * batch["t"] * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return torch.nn.ReLU()(batch["K"] - sde)

    @staticmethod
    def solution(batch):
        """
        Outputs the exact solution.
        """
        sigma_sqrtt = batch["sigma"] * torch.sqrt(batch["t"])
        _d = (
            -(
                torch.log(batch["x"] / batch["K"])
                + 0.5 * batch["t"] * batch["sigma"] ** 2
            )
            / sigma_sqrtt
        )
        return batch["K"] * n_dist(_d + sigma_sqrtt) - batch["x"] * n_dist(_d)

    @staticmethod
    def vega_solution(batch):
        """
        Outputs the exact vega.
        """
        sqrtt = torch.sqrt(batch["t"])
        _d = (
            torch.log(batch["x"] / batch["K"]) + 0.5 * batch["t"] * batch["sigma"] ** 2
        ) / (batch["sigma"] * sqrtt)
        return batch["x"] * sqrtt * n_density(_d)


HYPERCUBES.update(
    {
        f"heat_para_{d_heat_para}d": {
            "t": Hypercube(interval=[0.0, 1.0]),
            "x": Hypercube(interval=[0.5, 1.5], dims=(d_heat_para,)),
            "sigma": Hypercube(interval=[0.0, 1.0], dims=(d_heat_para, d_heat_para)),
        }
        for d_heat_para in range(1, 16)
    }
)


class HeatParaboloid(Pde):
    params = ("t", "x", "sigma")

    def __init__(self, hypercubes=HYPERCUBES["heat_para_10d"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        d = hypercubes["x"].dims[0]
        return all(
            [
                hypercubes["t"].dims == (1,),
                hypercubes["x"].dims == (d,),
                hypercubes["sigma"].dims == (d, d),
            ]
        )

    @staticmethod
    def sde(batch):
        """
        Outputs batched realizations of the SDE.
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        x_sigma_dw = batch["x"] + torch.einsum("ijk, ik -> ij", batch["sigma"], dw)
        return (x_sigma_dw ** 2).sum(-1, keepdims=True)

    @staticmethod
    def solution(batch):
        """
        Outputs the exact solution.
        """
        t_trace = batch["t"] * torch.einsum(
            "ijk, ijk -> i", batch["sigma"], batch["sigma"]
        ).unsqueeze(1)
        return (batch["x"] ** 2).sum(-1, keepdims=True) + t_trace


HYPERCUBES.update(
    {
        f"heat_gaussian_{d_heat_gauss}d": {
            "t": Hypercube(interval=[0.0, 1.0]),
            "x": Hypercube(interval=[-0.1, 0.1], dims=(d_heat_gauss,)),
            "sigma": Hypercube(interval=[0, 0.1]),
        }
        for d_heat_gauss in range(10, 200, 10)
    }
)


class HeatGaussian(Pde):
    params = ("t", "x", "sigma")

    def __init__(self, hypercubes=HYPERCUBES["heat_gaussian_150d"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        d = hypercubes["x"].dims[0]
        return all(
            [
                hypercubes["t"].dims == (1,),
                hypercubes["x"].dims == (d,),
                hypercubes["sigma"].dims == (1,),
            ]
        )

    @staticmethod
    def sde(batch):
        """
        Outputs batched realizations of the SDE.
        """
        dw = torch.sqrt(batch["t"]) * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        x_sigma_dw = batch["x"] + batch["sigma"] * dw
        return torch.exp(-(x_sigma_dw ** 2).sum(-1, keepdims=True))

    @staticmethod
    def solution(batch):
        """
        Outputs the exact solution.
        """
        denom = 1 + 2 * batch["t"] * batch["sigma"] ** 2
        norm_x_sq = (batch["x"] ** 2).sum(-1, keepdims=True)
        return torch.exp(-norm_x_sq / denom) / denom ** (batch["x"].shape[-1] / 2)


PDES = {pde.__name__: pde for pde in Pde.get_subclasses()}
