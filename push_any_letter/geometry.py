"""Convert font contours (including holes) into non-overlapping triangles."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import numpy as np
import mapbox_earcut
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from shapely import affinity
from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import unary_union


class OutlinePen(BasePen):
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.rings = []
        self.points = []

    def _moveTo(self, p):
        self.points = [p]

    def _lineTo(self, p):
        self.points.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = np.asarray(self.points[-1])
        p1, p2, p3 = map(np.asarray, (p1, p2, p3))
        for t in np.linspace(0, 1, 13)[1:]:
            self.points.append(tuple((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3))

    def _qCurveToOne(self, p1, p2):
        p0 = np.asarray(self.points[-1])
        p1, p2 = map(np.asarray, (p1, p2))
        for t in np.linspace(0, 1, 13)[1:]:
            self.points.append(tuple((1-t)**2*p0 + 2*(1-t)*t*p1 + t*t*p2))

    def _closePath(self):
        if len(self.points) >= 3:
            self.rings.append(self.points)
        self.points = []

    _endPath = _closePath


def default_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        str(Path(pygame.__file__).parent / pygame.font.get_default_font()),
    ]
    return next(p for p in candidates if Path(p).is_file())


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [p for g in geometry.geoms for p in polygons(g)] if hasattr(geometry, "geoms") else []


@dataclass
class Glyph:
    letter: str
    geometry: object
    triangles: list
    font_path: str


@lru_cache(maxsize=256)
def make_glyph(letter, font_path=None, size=156.0):
    if len(letter) != 1 or letter.isspace():
        raise ValueError("Choose one visible character.")
    font_path = str(Path(font_path or default_font()).expanduser().resolve())
    with TTFont(font_path, fontNumber=0) as font:
        cmap = font.getBestCmap()
        name = cmap.get(ord(letter))
        if name is None:
            raise ValueError(f"Font does not contain {letter!r}; choose another --font.")
        glyph_set = font.getGlyphSet()
        pen = OutlinePen(glyph_set)
        glyph_set[name].draw(pen)
    # Even/odd nesting keeps counters empty, regardless of contour direction.
    geom = GeometryCollection()
    for ring in pen.rings:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        geom = geom.symmetric_difference(poly)
    if geom.is_empty or geom.area <= 0:
        raise ValueError("Character has no usable outline.")
    minx, miny, maxx, maxy = geom.bounds
    scale = size / max(maxx-minx, maxy-miny)
    geom = affinity.scale(geom, xfact=scale, yfact=-scale, origin=(0, 0))
    geom = geom.simplify(0.3, preserve_topology=True)
    center = geom.centroid
    geom = affinity.translate(geom, -center.x, -center.y)
    triangles = []
    for poly in polygons(geom):
        rings = [np.asarray(poly.exterior.coords[:-1])]
        rings += [np.asarray(r.coords[:-1]) for r in poly.interiors]
        vertices = np.vstack(rings).astype(np.float64)
        ends = np.cumsum([len(r) for r in rings], dtype=np.uint32)
        indices = mapbox_earcut.triangulate_float64(vertices, ends).reshape(-1, 3)
        triangles.extend(vertices[idx].tolist() for idx in indices if Polygon(vertices[idx]).area > 1e-8)
    mesh = unary_union([Polygon(t) for t in triangles])
    if mesh.symmetric_difference(geom).area > geom.area * 1e-5:
        raise ValueError("Could not triangulate this glyph accurately.")
    return Glyph(letter, geom, triangles, font_path)


def transform(geometry, pose):
    return affinity.translate(affinity.rotate(geometry, float(pose[2]), origin=(0, 0), use_radians=True), float(pose[0]), float(pose[1]))
