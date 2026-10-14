import math


def segment_pixels(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    step_x = 1 if x1 >= x0 else -1
    step_y = 1 if y1 >= y0 else -1

    x, y = x0, y0
    error = 0
    pixels = [(x, y)]

    while (x, y) != (x1, y1):
        if 2 * error < dx - dy:
            error += dy
            x += step_x
        else:
            error -= dx
            y += step_y
        pixels.append((x, y))

    return pixels


def to_spherical(x, y, plane_distance=0.0):
    radius = math.hypot(math.hypot(x, y), plane_distance)
    if radius == 0.0:
        return 0.0, 0.0, 0.0
    theta = math.degrees(math.acos(plane_distance / radius))
    phi = math.degrees(math.atan2(y, x))
    return radius, theta, phi


def to_cartesian(radius, theta, phi):
    theta = math.radians(theta)
    phi = math.radians(phi)
    return (radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta))
