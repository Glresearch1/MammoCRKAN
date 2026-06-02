"""Kolmogorov-Arnold Network layers used by GMIC."""

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    """A linear layer augmented with learnable B-spline bases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[torch.nn.Module] = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid_step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * grid_step
            + grid_range[0]
        )
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline,
                )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected input with shape [batch, {self.in_features}], "
                f"got {tuple(x.shape)}."
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline bases for an input tensor."""

        self._check_input(x)
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)

        for order in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(order + 1)])
                / (self.grid[:, order:-1] - self.grid[:, : -(order + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, order + 1 :] - x)
                / (self.grid[:, order + 1 :] - self.grid[:, 1:(-order)])
                * bases[:, :, 1:]
            )

        expected_shape = (x.size(0), self.in_features, self.grid_size + self.spline_order)
        if bases.size() != expected_shape:
            raise RuntimeError(f"Unexpected B-spline shape: {tuple(bases.size())}")
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute coefficients of the curve that interpolates the given points."""

        self._check_input(x)
        expected_y_shape = (x.size(0), self.in_features, self.out_features)
        if y.size() != expected_y_shape:
            raise ValueError(f"Expected y shape {expected_y_shape}, got {tuple(y.size())}.")

        spline_basis = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(spline_basis, targets).solution
        result = solution.permute(2, 0, 1)

        expected_result_shape = (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if result.size() != expected_result_shape:
            raise RuntimeError(f"Unexpected coefficient shape: {tuple(result.size())}")
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if not self.enable_standalone_scale_spline:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        self._check_input(x)
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0,
                batch - 1,
                self.grid_size + 1,
                dtype=torch.int64,
                device=x.device,
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        l1_weights = self.spline_weight.abs().mean(-1)
        activation_loss = l1_weights.sum()
        probabilities = l1_weights / activation_loss.clamp_min(1e-12)
        entropy_loss = -torch.sum(probabilities * probabilities.clamp_min(1e-12).log())
        return regularize_activation * activation_loss + regularize_entropy * entropy_loss


class KAN(torch.nn.Module):
    """A stack of KANLinear layers."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type[torch.nn.Module] = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least input and output sizes.")

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList(
            [
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                for in_features, out_features in zip(layers_hidden, layers_hidden[1:])
            ]
        )

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
