import ctypes

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_LINEAR,
    GL_R32F,
    GL_RED,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRIANGLE_STRIP,
    GL_UNPACK_ALIGNMENT,
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
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

import dicom_data
from labkit import glkit

TRIANGLE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPosition;
layout (location = 1) in vec3 aColor;

uniform float uSpin;

out vec3 vColor;

void main()
{
    float s = sin(uSpin);
    float c = cos(uSpin);
    vec2 p = vec2(aPosition.x * c - aPosition.y * s, aPosition.x * s + aPosition.y * c);
    vColor = aColor;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

TRIANGLE_FRAGMENT_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
"""

TRIANGLE_VERTICES = [
     0.0,   0.7,   1.0,  0.15, 0.15,
    -0.75, -0.6,   0.15, 0.85, 0.35,
     0.75, -0.6,   0.20, 0.45, 1.00,
]

TRIANGLE_STRIDE = 5 * ctypes.sizeof(ctypes.c_float)

IMAGE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPosition;
uniform vec2 uScale;

out vec2 vTexCoord;

void main()
{
    vTexCoord = vec2(aPosition.x * 0.5 + 0.5, 0.5 - aPosition.y * 0.5);
    gl_Position = vec4(aPosition * uScale, 0.0, 1.0);
}
"""

IMAGE_FRAGMENT_SHADER = """
#version 330 core
in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uImage;
uniform float uCenter;
uniform float uWidth;
uniform bool  uInvert;

void main()
{
    float value = texture(uImage, vTexCoord).r;

    float width = max(uWidth, 1.0);
    float gray = clamp((value - (uCenter - 0.5)) / (width - 1.0) + 0.5, 0.0, 1.0);
    if (uInvert) {
        gray = 1.0 - gray;
    }
    fragColor = vec4(vec3(gray), 1.0);
}
"""

QUAD_VERTICES = [
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
]


class TriangleScene:
    background = (0.09, 0.10, 0.13, 1.0)

    def __init__(self, options=None):
        self.program = glkit.compile_program(TRIANGLE_VERTEX_SHADER, TRIANGLE_FRAGMENT_SHADER)
        self.uniforms = {"uSpin": glGetUniformLocation(self.program, "uSpin")}

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glkit.upload_floats(TRIANGLE_VERTICES)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, TRIANGLE_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        offset = ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, TRIANGLE_STRIDE, offset)
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def metadata(self):
        return {"description": "Графічний примітив: трикутник"}

    def draw(self, width, height, params=None):
        params = params or {}
        glViewport(0, 0, width, height)
        glClearColor(*self.background)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glUniform1f(self.uniforms["uSpin"], float(params.get("spin", 0.0)))
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)


class DicomScene:
    background = (0.0, 0.0, 0.0, 1.0)

    def __init__(self, options=None):
        options = options or {}
        self.values, self.dataset = dicom_data.load_dicom(options.get("file"))
        self.info = dicom_data.metadata(self.dataset, self.values)
        self.invert = self.info["inverted"]
        self.center = self.info["defaultCenter"]
        self.width = self.info["defaultWidth"]

        self.program = glkit.compile_program(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.uniforms = {name: glGetUniformLocation(self.program, name)
                         for name in ("uImage", "uCenter", "uWidth", "uInvert", "uScale")}

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glkit.upload_floats(QUAD_VERTICES)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        rows, columns = self.values.shape
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, columns, rows, 0, GL_RED, GL_FLOAT, self.values)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self.aspect = columns / rows

    def metadata(self):
        return dict(self.info)

    def draw(self, width, height, params=None):
        params = params or {}
        center = float(params.get("windowCenter", self.center))
        window = float(params.get("windowWidth", self.width))

        glViewport(0, 0, width, height)
        glClearColor(*self.background)
        glClear(GL_COLOR_BUFFER_BIT)

        view_aspect = width / max(height, 1)
        if view_aspect > self.aspect:
            scale = (self.aspect / view_aspect, 1.0)
        else:
            scale = (1.0, view_aspect / self.aspect)

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(self.uniforms["uImage"], 0)
        glUniform1f(self.uniforms["uCenter"], center)
        glUniform1f(self.uniforms["uWidth"], window)
        glUniform1i(self.uniforms["uInvert"], int(self.invert))
        glUniform2f(self.uniforms["uScale"], *scale)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)


SCENES = {"primitive": TriangleScene, "dicom": DicomScene}


def create_scene(name, options=None):
    if name not in SCENES:
        raise KeyError(f"unknown scene: {name}")
    return SCENES[name](options)
