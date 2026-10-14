import ctypes
import os

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_LINEAR,
    GL_NEAREST,
    GL_RGB,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_STRIP,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

from . import glkit

LIGHT = os.environ.get("LABKIT_THEME", "").lower() == "light"

if LIGHT:
    BACKGROUND = (1.0, 1.0, 1.0)
    BACKGROUND_HEX = "#FFFFFF"
    RULE_HEX = "#000000"
    GRID_HEX = "#E5E5E5"
    TEXT_HEX = "#000000"
    MUTED_HEX = "#555555"
    ACCENT_HEX = "#1F77B4"
    GOLD_HEX = "#FF7F0E"
    SERIES_HEX = ("#1F77B4", "#FF7F0E", "#D62728", "#2CA02C", "#9467BD", "#8C564B")
else:
    BACKGROUND = (0.039, 0.047, 0.055)
    BACKGROUND_HEX = "#0A0C0E"
    RULE_HEX = "#2A3238"
    GRID_HEX = "#1C2226"
    TEXT_HEX = "#C9D1D6"
    MUTED_HEX = "#6B7780"
    ACCENT_HEX = "#5FD3D8"
    GOLD_HEX = "#C9A66B"
    SERIES_HEX = ("#5FD3D8", "#C9A66B", "#D8735F", "#8FB86B", "#A58AD8", "#D8C35F")

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aCorner;   // 0..1, upper-left first
uniform vec2 uScale;                     // image size as a share of the viewport
out vec2 vTexCoord;

void main()
{
    vTexCoord = aCorner;
    vec2 ndc = vec2(aCorner.x * 2.0 - 1.0, 1.0 - aCorner.y * 2.0) * uScale;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;
uniform sampler2D uImage;
out vec4 fragColor;

void main()
{
    fragColor = vec4(texture(uImage, vTexCoord).rgb, 1.0);
}
"""


class ImageQuad:

    def __init__(self, smooth=True):
        self.program = glkit.compile_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._scale = glGetUniformLocation(self.program, "uScale")
        self._sampler = glGetUniformLocation(self.program, "uImage")

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glkit.upload_floats([0, 0, 1, 0, 0, 1, 1, 1])
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        filtering = GL_LINEAR if smooth else GL_NEAREST
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.size = (0, 0)

    def upload(self, rgb):
        image = np.asarray(rgb)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        image = np.ascontiguousarray(image[:, :, :3])
        height, width = image.shape[:2]

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, image)
        self.size = (width, height)

    def draw(self, width, height, background=BACKGROUND):
        glViewport(0, 0, width, height)
        glClearColor(*background, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        if not all(self.size):
            return

        image_aspect = self.size[0] / self.size[1]
        view_aspect = width / max(height, 1)
        if image_aspect > view_aspect:
            scale = (1.0, view_aspect / image_aspect)
        else:
            scale = (image_aspect / view_aspect, 1.0)

        glUseProgram(self.program)
        glUniform2f(self._scale, *scale)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(self._sampler, 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


def plot_style():
    from cycler import cycler

    if LIGHT:
        return {"font.size": 10, "axes.prop_cycle": cycler(color=list(SERIES_HEX))}
    return {
        "figure.facecolor": BACKGROUND_HEX,
        "axes.facecolor": BACKGROUND_HEX,
        "savefig.facecolor": BACKGROUND_HEX,
        "axes.edgecolor": RULE_HEX,
        "axes.labelcolor": TEXT_HEX,
        "axes.titlecolor": TEXT_HEX,
        "axes.grid": True,
        "grid.color": GRID_HEX,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED_HEX,
        "ytick.color": MUTED_HEX,
        "text.color": TEXT_HEX,
        "legend.facecolor": BACKGROUND_HEX,
        "legend.edgecolor": RULE_HEX,
        "font.family": "DejaVu Sans Mono",
        "font.size": 10,
        "axes.prop_cycle": cycler(color=list(SERIES_HEX)),
    }


def style():
    import matplotlib

    return matplotlib.rc_context(plot_style())


def new_figure(width_px=960, height_px=640, dpi=100, **subplot_kw):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.subplots(**subplot_kw)
    return figure, axes


def figure_to_rgb(figure):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = figure.canvas if isinstance(figure.canvas, FigureCanvasAgg) else FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
