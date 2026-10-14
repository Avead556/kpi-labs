import ctypes
import math

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_LINES,
    GL_TRIANGLES,
    glBindBuffer,
    glBindVertexArray,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniform2f,
    glUniform3f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

import raster
from labkit import glkit

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aCell;
uniform vec2 uExtent;
uniform vec2 uOrigin;

void main()
{
    vec2 cell = aCell + uOrigin;
    vec2 ndc = vec2(cell.x / uExtent.x * 2.0 - 1.0,
                    1.0 - cell.y / uExtent.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform vec3 uColor;
out vec4 fragColor;

void main()
{
    fragColor = vec4(uColor, 1.0);
}
"""

BACKGROUND = (0.039, 0.047, 0.055)
GRID_COLOR = (0.110, 0.133, 0.149)
AXIS_COLOR = (0.333, 0.380, 0.416)
NET_COLOR = (0.165, 0.310, 0.330)
PIXEL_COLOR = (0.373, 0.827, 0.847)
IDEAL_COLOR = (0.788, 0.651, 0.420)
MARKER_COLOR = (1.0, 1.0, 1.0)

DEFAULT_GRID = 40
MARGIN = 2.0
TICK_EVERY = 5
TICK_LENGTH = 0.5
CELL_INSET = 0.06
MARKER_RADIUS = 0.38
AXIS_WIDTH = 0.07
TICK_WIDTH = 0.05
IDEAL_WIDTH = 0.09
ARROW_LENGTH = 0.9
ARROW_WIDTH = 0.34
NET_WIDTH = 0.035
NET_ANGLES = (15, 30, 45, 60, 75)
NET_ARC_STEPS = 28


class SegmentScene:

    def __init__(self, options=None):
        options = options or {}
        self.columns = int(options.get("columns", DEFAULT_GRID))
        self.rows = int(options.get("rows", DEFAULT_GRID))

        self.program = glkit.compile_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.uniforms = {name: glGetUniformLocation(self.program, name)
                         for name in ("uExtent", "uOrigin", "uColor")}

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        self.grid_lines = self._grid_lines()
        self.axes = self._axes()


    def _grid_lines(self):
        vertices = []
        for column in range(self.columns + 1):
            vertices += [column, 0, column, self.rows]
        for row in range(self.rows + 1):
            vertices += [0, row, self.columns, row]
        return vertices

    def _axes(self):
        end_x = self.columns + ARROW_LENGTH
        end_y = self.rows + ARROW_LENGTH
        vertices = []
        vertices += _thick_segment(0, 0, end_x, 0, AXIS_WIDTH)
        vertices += _thick_segment(0, 0, 0, end_y, AXIS_WIDTH)
        vertices += [end_x + ARROW_LENGTH, 0,
                     end_x, -ARROW_WIDTH,
                     end_x, ARROW_WIDTH]
        vertices += [0, end_y + ARROW_LENGTH,
                     -ARROW_WIDTH, end_y,
                     ARROW_WIDTH, end_y]
        for column in range(TICK_EVERY, self.columns + 1, TICK_EVERY):
            vertices += _thick_segment(column, -TICK_LENGTH, column, TICK_LENGTH, TICK_WIDTH)
        for row in range(TICK_EVERY, self.rows + 1, TICK_EVERY):
            vertices += _thick_segment(-TICK_LENGTH, row, TICK_LENGTH, row, TICK_WIDTH)
        return vertices

    def _spherical_net(self, distance):
        reach = math.hypot(self.columns, self.rows)
        vertices = []

        for theta in NET_ANGLES:
            radius = distance * math.tan(math.radians(theta))
            if radius <= 0 or radius > reach:
                continue
            previous = None
            for step in range(NET_ARC_STEPS + 1):
                angle = math.radians(90.0 * step / NET_ARC_STEPS)
                point = (radius * math.cos(angle), radius * math.sin(angle))
                if previous is not None:
                    vertices += _thick_segment(*previous, *point, NET_WIDTH)
                previous = point

        for phi in NET_ANGLES:
            angle = math.radians(phi)
            vertices += _thick_segment(0, 0, reach * math.cos(angle),
                                       reach * math.sin(angle), NET_WIDTH)
        return vertices

    def _cell_quads(self, pixels, inset=CELL_INSET):
        vertices = []
        for x, y in pixels:
            left, right = x + inset, x + 1 - inset
            top, bottom = y + inset, y + 1 - inset
            vertices += [left, top, right, top, right, bottom,
                         left, top, right, bottom, left, bottom]
        return vertices

    def _marker_quads(self, points):
        vertices = []
        for x, y in points:
            cx, cy = x + 0.5, y + 0.5
            left, right = cx - MARKER_RADIUS, cx + MARKER_RADIUS
            top, bottom = cy - MARKER_RADIUS, cy + MARKER_RADIUS
            vertices += [left, top, right, top, right, bottom,
                         left, top, right, bottom, left, bottom]
        return vertices


    def endpoints(self, params):
        params = params or {}
        start = (self._clamp_x(params.get("x0", 4)), self._clamp_y(params.get("y0", 3)))
        end = (self._clamp_x(params.get("x1", 31)), self._clamp_y(params.get("y1", 24)))
        return start, end

    def _clamp_x(self, value):
        return max(0, min(int(round(float(value))), self.columns - 1))

    def _clamp_y(self, value):
        return max(0, min(int(round(float(value))), self.rows - 1))


    def metadata(self):
        return {"description": f"Відрізок, 4-зв'язність, сітка {self.columns}×{self.rows}",
                "columns": self.columns, "rows": self.rows}

    def draw(self, width, height, params=None):
        start, end = self.endpoints(params)
        distance = float((params or {}).get("planeDistance", 0.0))
        pixels = raster.segment_pixels(*start, *end)

        glViewport(0, 0, width, height)
        glClearColor(*BACKGROUND, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform2f(self.uniforms["uExtent"],
                    self.columns + 2 * MARGIN, self.rows + 2 * MARGIN)
        glUniform2f(self.uniforms["uOrigin"], MARGIN, MARGIN)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        self._batch(self.grid_lines, GRID_COLOR, GL_LINES)
        self._batch(self._spherical_net(distance), NET_COLOR, GL_TRIANGLES)
        self._batch(self.axes, AXIS_COLOR, GL_TRIANGLES)
        self._batch(self._cell_quads(pixels), PIXEL_COLOR, GL_TRIANGLES)
        ideal = _thick_segment(start[0] + 0.5, start[1] + 0.5,
                               end[0] + 0.5, end[1] + 0.5, IDEAL_WIDTH)
        self._batch(ideal, IDEAL_COLOR, GL_TRIANGLES)
        self._batch(self._marker_quads([start, end]), MARKER_COLOR, GL_TRIANGLES)

    def _batch(self, vertices, color, mode):
        if not vertices:
            return
        glkit.upload_floats(vertices)
        glUniform3f(self.uniforms["uColor"], *color)
        glDrawArrays(mode, 0, len(vertices) // 2)


    def readout(self, params=None):
        params = params or {}
        start, end = self.endpoints(params)
        distance = float(params.get("planeDistance", 0.0))
        pixels = raster.segment_pixels(*start, *end)
        span = abs(end[0] - start[0]) + abs(end[1] - start[1]) + 1

        rows = [{"label": "ПЛОЩИНА z = d", "value": f"{distance:.0f} пікс"}]
        for name, (x, y) in (("ПОЧАТОК", start), ("КІНЕЦЬ", end)):
            radius, theta, phi = raster.to_spherical(x, y, distance)
            rows.append({"label": f"{name} (x, y)", "value": f"{x}, {y}"})
            rows.append({"label": f"{name} (r, θ, φ)",
                         "value": f"{radius:.1f}, {theta:.1f}°, {phi:.1f}°"})
        rows.append({"label": "ПІКСЕЛІВ", "value": f"{len(pixels)} (dx+dy+1 = {span})"})
        return rows


def _thick_segment(x0, y0, x1, y1, width):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return []

    offset_x = -dy / length * width / 2
    offset_y = dx / length * width / 2
    return [x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y,
            x1 - offset_x, y1 - offset_y,
            x0 + offset_x, y0 + offset_y, x1 - offset_x, y1 - offset_y,
            x0 - offset_x, y0 - offset_y]


SCENES = {"segment": SegmentScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
