import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import raster
import scenes
from labkit import glkit

WINDOW_SIZE = 760
DISTANCE_STEP = 5.0
DEFAULTS = {"x0": 4, "y0": 3, "x1": 31, "y1": 24, "planeDistance": 20.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    window = glkit.create_window("Відрізок: 4-зв'язність, сферичні координати",
                                 WINDOW_SIZE, WINDOW_SIZE)
    scene = scenes.SegmentScene()
    params = dict(DEFAULTS)
    place_start = True

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<18} {row['value']}")
        glfw.set_window_title(
            window,
            f"Відрізок ({params['x0']},{params['y0']})-({params['x1']},{params['y1']})"
            f" | d = {params['planeDistance']:.0f}")

    def on_mouse(target, button, action, mods):
        nonlocal place_start
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return

        x, y = glfw.get_cursor_pos(target)
        width, height = glfw.get_window_size(target)
        extent_x = scene.columns + 2 * scenes.MARGIN
        extent_y = scene.rows + 2 * scenes.MARGIN
        column = scene._clamp_x(math.floor(x / max(width, 1) * extent_x - scenes.MARGIN))
        row = scene._clamp_y(math.floor(y / max(height, 1) * extent_y - scenes.MARGIN))

        if place_start:
            params["x0"], params["y0"] = column, row
        else:
            params["x1"], params["y1"] = column, row
        place_start = not place_start
        announce()

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
        elif key == glfw.KEY_SPACE:
            announce()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            params["planeDistance"] += DISTANCE_STEP
            announce()
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            params["planeDistance"] = max(0.0, params["planeDistance"] - DISTANCE_STEP)
            announce()
        elif key == glfw.KEY_R:
            params.update(DEFAULTS)
            announce()

    glfw.set_mouse_button_callback(window, on_mouse)
    glfw.set_key_callback(window, on_key)

    print(f"grid: {scene.columns}x{scene.rows}, origin in the upper-left corner",
          file=sys.stderr)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
