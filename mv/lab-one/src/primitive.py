import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import glfw
from labkit import glkit
import scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    window = glkit.create_window("OpenGL", 800, 600)
    glkit.close_on_escape(window)
    scene = scenes.TriangleScene()

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height)

    glkit.run(window, draw, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
