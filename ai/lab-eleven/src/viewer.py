import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
from labkit import glkit

WINDOW_SIZE = (1200, 760)


def run(scene_class, title, defaults, step_key, step_range, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=int, default=value)
    args = parser.parse_args()

    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = scene_class()
    params = {name: getattr(args, name) for name in defaults}
    low, high = step_range

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<28} {row['value']}")

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
            return
        if key in (glfw.KEY_UP, glfw.KEY_DOWN):
            params["dataset"] = 1 - params["dataset"]
        elif key == glfw.KEY_RIGHT:
            params[step_key] = min(high, params[step_key] + 1)
        elif key == glfw.KEY_LEFT:
            params[step_key] = max(low, params[step_key] - 1)
        elif key == glfw.KEY_R:
            params.update(defaults)
        else:
            return
        announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
