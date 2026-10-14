import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import scenes
from labkit import glkit

WINDOW_SIZE = (1280, 800)


def run(description, scene_name, title, params, keys):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    for name, value in params.items():
        parser.add_argument(f"--{name}", type=float, default=value)
    args = parser.parse_args()
    params = {name: getattr(args, name) for name in params}
    initial = dict(params)

    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = scenes.create_scene(scene_name)

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<26} {row['value']}")

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
        elif key == glfw.KEY_R:
            params.update(initial)
            announce()
        elif key in keys:
            name, step, low, high = keys[key]
            params[name] = min(max(params[name] + step, low), high)
            announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
