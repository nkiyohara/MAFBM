import numpy as np
import torch
from sde.motions.motion import Motion
from sde.jax.solvers.sde import sdeint_fixed_ts


def render(z, size, sigma=0.05):
    grid = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    x = np.exp(-0.5 * ((grid - z[:, None, None]) ** 2).sum(0) / sigma**2)
    return x[..., None].astype(np.float32)


class Dynamics:
    def f(self, t, y):
        return -0.2 * y

    def g(self, t, y):
        return torch.ones_like(y) * 0.1


class Ball:
    def __init__(self, train, sequence_length, hurst=0.5, size=64, dt=0.1):
        self.train = train
        self.sequence_length = sequence_length
        self.hurst = hurst
        self.size = size
        self.dt = dt
        self.dynamics = Dynamics()

        if not train:
            self.data = [self.generate() for _ in range(len(self))]

    def __len__(self):
        if self.train:
            return 6000
        else:
            return 300

    def generate(self):
        ts_ = np.arange(self.sequence_length) * self.dt
        ts = torch.from_numpy(ts_)
        motion = Motion(
            "fractional_brownian",
            (2,),
            ts[0],
            ts[-1],
            self.sequence_length,
            hurst=self.hurst,
        )
        zs = sdeint_fixed_ts(self.dynamics, motion, torch.zeros((2,)), ts)
        x = np.array([render(z, self.size) for z in zs.cpu().numpy()])
        return x

    def __getitem__(self, index):
        if self.train:
            return self.generate()
        else:
            return self.data[index]
