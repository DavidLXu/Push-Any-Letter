"""Pygame rendering shares the exact outline used by collision geometry."""
import math
import pygame
from .geometry import polygons, transform

PAPER = (241, 244, 239)
INK = (26, 40, 47)
ORANGE = (237, 126, 67)
TEAL = (43, 154, 143)


def draw_geometry(surface, geom, fill, outline=None, width=2):
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for poly in polygons(geom):
        points = list(poly.exterior.coords)
        pygame.draw.polygon(layer, fill, points)
        for ring in poly.interiors:
            pygame.draw.polygon(layer, (0, 0, 0, 0), list(ring.coords))
        if outline:
            pygame.draw.lines(layer, outline, True, points, width)
            for ring in poly.interiors:
                pygame.draw.lines(layer, outline, True, list(ring.coords), width)
    surface.blit(layer, (0, 0))


def draw_arena(env, trail=None):
    surface = pygame.Surface((640, 640))
    surface.fill(PAPER)
    for x in range(30, 630, 24):
        for y in range(30, 630, 24):
            pygame.draw.circle(surface, (216, 224, 216), (x, y), 1)
    pygame.draw.rect(surface, (184, 198, 188), (10, 10, 620, 620), 2, border_radius=14)
    draw_geometry(surface, env.goal_geometry, (*TEAL, 34), TEAL, 2)
    gx, gy, ga = env.goal_pose
    pygame.draw.circle(surface, TEAL, (int(gx), int(gy)), 3)
    pygame.draw.line(surface, TEAL, (gx,gy), (gx+22*math.cos(ga), gy+22*math.sin(ga)), 2)
    if trail and len(trail) > 1:
        pygame.draw.lines(surface, (179, 199, 207), False, trail, 2)
    pose = (*env.block.position, env.block.angle)
    shadow = transform(env.glyph.geometry, (pose[0]+3, pose[1]+5, pose[2]))
    draw_geometry(surface, shadow, (24, 37, 45, 24))
    draw_geometry(surface, transform(env.glyph.geometry, pose), ORANGE, (158, 75, 39), 2)
    x, y = env.agent.position
    pygame.draw.circle(surface, (24,37,45,25), (round(x+2), round(y+3)), 18)
    pygame.draw.circle(surface, INK, (round(x), round(y)), env.pusher_radius)
    pygame.draw.circle(surface, (255,255,255), (round(x), round(y)), env.pusher_radius-4, 2)
    pygame.draw.circle(surface, (216,227,231), (round(x), round(y)), 3)
    tx, ty = (int(round(float(v))) for v in env.last_action)
    pygame.draw.line(surface, (123,139,145), (tx-5,ty), (tx+5,ty), 1)
    pygame.draw.line(surface, (123,139,145), (tx,ty-5), (tx,ty+5), 1)
    return surface
