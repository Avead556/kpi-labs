import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import scenes
from labkit import glkit

WINDOW = (1280, 720)
DEFAULTS = {"function": 0, "optimizer": 0, "logLr": -3.0, "x0": 8.0, "y0": 8.0, "steps": 500}
LR_FACTOR = math.log10(1.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=scenes.FUNCTION_ORDER, default="parabola")
    parser.add_argument("--optimizer", choices=scenes.OPTIMIZER_ORDER, default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--start", type=float, nargs=2, default=(8.0, 8.0))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    window = glkit.create_window("Градієнтний спуск", *WINDOW)
    scene = scenes.DescentScene()
    params = {
        "function": scenes.FUNCTION_ORDER.index(args.function),
        "optimizer": scenes.OPTIMIZER_ORDER.index(args.optimizer),
        "logLr": math.log10(args.lr), "x0": args.start[0], "y0": args.start[1],
        "steps": args.steps,
    }

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<22} {row['value']}")

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
            return
        if key == glfw.KEY_F:
            params["function"] = 1 - params["function"]
        elif key == glfw.KEY_O:
            params["optimizer"] = (params["optimizer"] + 1) % len(scenes.OPTIMIZER_ORDER)
        elif key == glfw.KEY_UP:
            params["logLr"] = min(0.0, params["logLr"] + LR_FACTOR)
        elif key == glfw.KEY_DOWN:
            params["logLr"] = max(-4.0, params["logLr"] - LR_FACTOR)
        elif key == glfw.KEY_RIGHT:
            params["steps"] = min(2000, params["steps"] + 50)
        elif key == glfw.KEY_LEFT:
            params["steps"] = max(10, params["steps"] - 50)
        elif key == glfw.KEY_R:
            params.update(DEFAULTS)
        else:
            return
        announce()

    def on_mouse(target, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(target)
        width, height = glfw.get_window_size(target)
        point = scene.point_at(x / width, y / height, width, height)
        if point is not None:
            params["x0"], params["y0"] = (round(value * 2) / 2 for value in point)
            announce()

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
