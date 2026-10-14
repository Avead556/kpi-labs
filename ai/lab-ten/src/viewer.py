import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
from labkit import glkit

WINDOW_SIZE = (1200, 800)


def main(scene_class, title, sliders, doc):
    parser = argparse.ArgumentParser(description=doc)
    for name, (default, low, high) in sliders.items():
        parser.add_argument(f"--{name}", type=int, default=default,
                            help=f"{low}..{high}, default {default}")
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    params = {name: getattr(args, name) for name in sliders}
    names = list(sliders)
    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = scene_class()

    def announce():
        glfw.set_window_title(window, f"{title} | " + ", ".join(
            f"{name} = {value}" for name, value in params.items()))
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<28} {row['value']}")

    def step(name, delta):
        default, low, high = sliders[name]
        params[name] = max(low, min(params[name] + delta, high))
        announce()

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
        elif key in (glfw.KEY_RIGHT, glfw.KEY_LEFT):
            step(names[0], 1 if key == glfw.KEY_RIGHT else -1)
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN) and len(names) > 1:
            step(names[1], 1 if key == glfw.KEY_UP else -1)
        elif key in (glfw.KEY_PAGE_UP, glfw.KEY_PAGE_DOWN) and len(names) > 2:
            step(names[2], 1 if key == glfw.KEY_PAGE_UP else -1)

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
