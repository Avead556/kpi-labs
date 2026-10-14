import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import scenes
from labkit import glkit

WINDOW_SIZE = (1200, 800)


def main(scene_name, title, choice, choices, epochs, epoch_step, doc):
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    parser.add_argument(f"--{choice}", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=epochs)
    args = parser.parse_args()

    window = glkit.create_window(title, *WINDOW_SIZE)
    scene = scenes.create_scene(scene_name)
    params = {choice: getattr(args, choice), "epoch": args.epoch}

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<34} {row['value']}")

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
            return
        if key == glfw.KEY_RIGHT:
            params[choice] = (params[choice] + 1) % choices
        elif key == glfw.KEY_LEFT:
            params[choice] = (params[choice] - 1) % choices
        elif key == glfw.KEY_UP:
            params["epoch"] = min(epochs, params["epoch"] + epoch_step)
        elif key == glfw.KEY_DOWN:
            params["epoch"] = max(0, params["epoch"] - epoch_step)
        elif key == glfw.KEY_HOME:
            params["epoch"] = 0
        elif key == glfw.KEY_END:
            params["epoch"] = epochs
        else:
            return
        announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)
