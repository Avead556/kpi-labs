import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import scenes
from labkit import glkit

WINDOW_SIZE = (1200, 780)


def run_view(doc, title, scene_name, defaults, keys):
    parser = argparse.ArgumentParser(description=doc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    for name, value in defaults.items():
        parser.add_argument(f"--{name}", type=int, default=value)
    args = parser.parse_args()

    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = scenes.create_scene(scene_name)
    params = {name: getattr(args, name) for name in defaults}

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<26} {row['value']}")
        glfw.set_window_title(window, f"{title} | " + ", ".join(
            f"{name} = {value}" for name, value in params.items()))

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
        elif key == glfw.KEY_R:
            params.update(defaults)
            announce()
        elif key in keys:
            name, step, low, high = keys[key]
            params[name] = max(low, min(params[name] + step, high))
            announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
