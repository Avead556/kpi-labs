import ctypes
import struct
import sys
import zlib

import glfw
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_PACK_ALIGNMENT,
    GL_RENDERBUFFER,
    GL_RGB,
    GL_RGBA8,
    GL_STATIC_DRAW,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBufferData,
    glCheckFramebufferStatus,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteFramebuffers,
    glDeleteRenderbuffers,
    glDeleteShader,
    glFramebufferRenderbuffer,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glPixelStorei,
    glReadPixels,
    glRenderbufferStorage,
    glShaderSource,
)

CONTEXT_VERSION = (3, 3)
MAX_OFFSCREEN_SIZE = 2048


class GLError(RuntimeError):
    pass


def create_window(title, width=800, height=600, visible=True):
    if not glfw.init():
        raise GLError("GLFW failed to initialise")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, CONTEXT_VERSION[0])
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, CONTEXT_VERSION[1])
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise GLError("could not create an OpenGL 3.3 Core context")

    glfw.make_context_current(window)
    if visible:
        glfw.swap_interval(1)
    return window


def create_offscreen_context():
    return create_window("offscreen", 1, 1, visible=False)


def compile_program(vertex_source, fragment_source):
    vertex = _compile_shader(vertex_source, GL_VERTEX_SHADER, "vertex")
    fragment = _compile_shader(fragment_source, GL_FRAGMENT_SHADER, "fragment")

    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode(errors="replace")
        raise GLError(f"shader program failed to link:\n{log}")

    glDeleteShader(vertex)
    glDeleteShader(fragment)
    return program


def _compile_shader(source, shader_type, kind):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode(errors="replace")
        raise GLError(f"{kind} shader failed to compile:\n{log}")
    return shader


def upload_floats(values, target=GL_ARRAY_BUFFER, usage=GL_STATIC_DRAW):
    array = (ctypes.c_float * len(values))(*values)
    glBufferData(target, ctypes.sizeof(array), array, usage)


class Offscreen:

    def __init__(self):
        self.size = (0, 0)
        self._framebuffer = None
        self._renderbuffer = None

    def bind(self, width, height):
        width = max(1, min(int(width), MAX_OFFSCREEN_SIZE))
        height = max(1, min(int(height), MAX_OFFSCREEN_SIZE))
        if (width, height) != self.size:
            self._allocate(width, height)
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffer)
        return self.size

    def _allocate(self, width, height):
        self.release()
        self._framebuffer = glGenFramebuffers(1)
        self._renderbuffer = glGenRenderbuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._framebuffer)
        glBindRenderbuffer(GL_RENDERBUFFER, self._renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, self._renderbuffer)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise GLError("offscreen framebuffer is incomplete")
        self.size = (width, height)

    def read_png(self, compression=1):
        width, height = self.size
        return encode_png(width, height, read_pixel_rows(width, height), compression)

    def release(self):
        if self._framebuffer is not None:
            glDeleteFramebuffers(1, [self._framebuffer])
            self._framebuffer = None
        if self._renderbuffer is not None:
            glDeleteRenderbuffers(1, [self._renderbuffer])
            self._renderbuffer = None
        self.size = (0, 0)


def read_pixel_rows(width, height):
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    return [pixels[y * width * 3:(y + 1) * width * 3] for y in reversed(range(height))]


def encode_png(width, height, rows, compression=6):
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    for tag, payload in (
        (b"IHDR", header),
        (b"IDAT", zlib.compress(raw, compression)),
        (b"IEND", b""),
    ):
        out += struct.pack(">I", len(payload))
        out += tag + payload
        out += struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    return bytes(out)


def save_screenshot(window, path):
    width, height = glfw.get_framebuffer_size(window)
    with open(path, "wb") as stream:
        stream.write(encode_png(width, height, read_pixel_rows(width, height), 9))
    print(f"saved {path} ({width}x{height})", file=sys.stderr)


def run(window, draw, screenshot=None):
    try:
        if screenshot:
            draw()
            save_screenshot(window, screenshot)
            return
        while not glfw.window_should_close(window):
            draw()
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        glfw.terminate()


def close_on_escape(window):
    def on_key(target, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(target, True)

    glfw.set_key_callback(window, on_key)
