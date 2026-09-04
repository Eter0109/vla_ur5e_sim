"""Configurable primitive geometry for the robosuite Lift object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from xml.etree.ElementTree import Element

import numpy as np

PrimitiveShape = Literal["box", "cylinder", "sphere"]


def _attribute(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


@dataclass(frozen=True)
class PrimitiveObjectConfig:
    """Physical and visual configuration for a single graspable primitive.

    ``dimensions_m`` always uses full XYZ dimensions in metres. For a cylinder,
    X and Y must match and represent its diameter. For a sphere, all three
    values must match and represent its diameter.
    """

    shape: PrimitiveShape = "box"
    dimensions_m: tuple[float, float, float] = (0.05, 0.05, 0.05)
    rgba: tuple[float, float, float, float] = (0.85, 0.12, 0.08, 1.0)
    density_kg_m3: float = 400.0
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        if self.shape not in ("box", "cylinder", "sphere"):
            raise ValueError(f"Unsupported primitive shape: {self.shape!r}.")
        if len(self.dimensions_m) != 3 or any(value <= 0 for value in self.dimensions_m):
            raise ValueError("dimensions_m must contain three positive full dimensions.")
        if self.shape == "cylinder" and not np.isclose(
            self.dimensions_m[0], self.dimensions_m[1], rtol=0.0, atol=1e-9
        ):
            raise ValueError("Cylinder X and Y dimensions must match its diameter.")
        if self.shape == "sphere" and not np.allclose(
            self.dimensions_m, self.dimensions_m[0], rtol=0.0, atol=1e-9
        ):
            raise ValueError("Sphere dimensions must be equal in X, Y, and Z.")
        if len(self.rgba) != 4 or any(value < 0 or value > 1 for value in self.rgba):
            raise ValueError("rgba must contain four values in [0, 1].")
        if self.density_kg_m3 <= 0:
            raise ValueError("density_kg_m3 must be positive.")
        if len(self.friction) != 3 or any(value < 0 for value in self.friction):
            raise ValueError("friction must contain three non-negative values.")

    @property
    def half_extents_m(self) -> tuple[float, float, float]:
        return tuple(value / 2.0 for value in self.dimensions_m)

    @property
    def mujoco_size(self) -> tuple[float, ...]:
        half_x, _half_y, half_z = self.half_extents_m
        if self.shape == "box":
            return self.half_extents_m
        if self.shape == "cylinder":
            return (half_x, half_z)
        return (half_x,)

    def apply_to_robosuite_object(self, obj: Any) -> None:
        """Mutate robosuite's generated cube before MuJoCo compilation.

        Lift internally depends on the object being named ``cube``. Retaining
        that object while changing its geoms keeps rewards, contacts, placement,
        and privileged observation names stable across primitive shapes.
        """

        # Lift's placement sampler queries BoxObject offsets from ``size``.
        # Keep full XYZ half extents here even though MuJoCo encodes cylinders
        # and spheres with shorter size tuples; they remain correct bounds.
        if hasattr(obj, "size"):
            obj.size = list(self.half_extents_m)
        getter = getattr(obj, "get_obj", None)
        root = getter() if callable(getter) else None
        if root is None:
            root = getattr(obj, "worldbody", None)
        if root is None:
            root = getattr(obj, "root", None)
        if root is None:
            raise RuntimeError("Unable to find the generated object's MJCF tree.")
        self.apply_to_xml(root)

    def apply_to_xml(self, root: Element) -> int:
        """Apply geometry attributes below ``root`` and return changed geoms."""

        geoms = list(root.findall(".//geom"))
        for geom in geoms:
            geom.set("type", self.shape)
            geom.set("size", _attribute(self.mujoco_size))
            geom.set("rgba", _attribute(self.rgba))
            # Remove a generated material so that rgba is deterministic.
            geom.attrib.pop("material", None)
            is_collision = (
                geom.get("contype", "1") != "0"
                and geom.get("conaffinity", "1") != "0"
            )
            if is_collision:
                geom.set("density", f"{self.density_kg_m3:.9g}")
                geom.set("friction", _attribute(self.friction))
        if not geoms:
            raise RuntimeError("The Lift object contains no MJCF geoms to configure.")
        return len(geoms)
