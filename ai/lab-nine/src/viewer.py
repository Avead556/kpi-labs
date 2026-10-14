import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
from labkit import glkit

WINDOW_SIZE = (1280, 800)


def run(title, create_scene, defaults, keys, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=float, default=value)
    args = parser.parse_args()

    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = create_scene()
    params = {name: getattr(args, name) for name in defaults}

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<24} {row['value']}")
        glfw.set_window_title(window, f"{title} | " + ", ".join(
            f"{name} = {value:g}" for name, value in params.items()))

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
        elif key == glfw.KEY_R:
            params.update(defaults)
            announce()
        elif key in keys:
            name, step, limit = keys[key]
            params[name] = min(max(0.0, params[name] + step), limit)
            announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
