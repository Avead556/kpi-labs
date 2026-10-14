import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
import scenes
from labkit import glkit

WINDOW = (1280, 800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=int, default=1,
                        help="index of the learning rate: 0 1e-6, 1 1e-5, 2 1e-4, 3 1.2e-4")
    parser.add_argument("--epoch", type=int, default=None,
                        help="epoch to show (default: the optimal one)")
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    window = glkit.create_window("Лінійна регресія: градієнтний спуск", *WINDOW)
    scene = scenes.RegressionScene()
    epochs = scene.losses.shape[1] - 1

    def optimal(index):
        best = scene.optimal[index]
        return best if best is not None else scene.last_finite[index]

    params = {"lr": args.lr,
              "epoch": args.epoch if args.epoch is not None else optimal(args.lr)}

    def announce():
        print()
        for row in scene.readout(params):
            print(f"  {row['label']:<30} {row['value']}")

    def on_key(target, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        stride = 100 if mods & glfw.MOD_SHIFT else 10
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(target, True)
            return
        if key == glfw.KEY_RIGHT:
            params["epoch"] = min(epochs, params["epoch"] + stride)
        elif key == glfw.KEY_LEFT:
            params["epoch"] = max(0, params["epoch"] - stride)
        elif key == glfw.KEY_L:
            params["lr"] = (params["lr"] + 1) % len(scene.learning_rates)
        elif key == glfw.KEY_B:
            params["epoch"] = optimal(params["lr"])
        else:
            return
        announce()

    glfw.set_key_callback(window, on_key)
    announce()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, params)

    glkit.run(window, draw, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
