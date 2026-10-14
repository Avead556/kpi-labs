import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
import sys

import dicom_data
import glfw
from labkit import glkit
import scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", metavar="DCM",
                        help="DICOM file to open (defaults to the CT sample shipped with pydicom)")
    parser.add_argument("--screenshot", metavar="PNG",
                        help="render one frame to a PNG file and exit")
    args = parser.parse_args()

    path = dicom_data.resolve_path(args.file)
    window = glkit.create_window("DICOM", 800, 800)
    scene = scenes.DicomScene({"file": path})
    info = scene.metadata()

    print(f"file: {path}", file=sys.stderr)
    print(f"study: {info['description']}, "
          f"values [{info['minValue']:.0f}; {info['maxValue']:.0f}]", file=sys.stderr)

    initial = (info["defaultCenter"], info["defaultWidth"])
    state = {"windowCenter": initial[0], "windowWidth": initial[1]}
    glfw.set_window_title(window, f"DICOM: {info['description']}")

    def on_key(win, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
            return
        if key == glfw.KEY_R:
            state["windowCenter"], state["windowWidth"] = initial
        step = max(state["windowWidth"] * 0.02, 1.0)
        if key == glfw.KEY_RIGHT:
            state["windowCenter"] += step
        elif key == glfw.KEY_LEFT:
            state["windowCenter"] -= step
        elif key == glfw.KEY_UP:
            state["windowWidth"] += step
        elif key == glfw.KEY_DOWN:
            state["windowWidth"] = max(state["windowWidth"] - step, 1.0)
        glfw.set_window_title(
            win, f"DICOM: {info['description']} | "
                 f"W={state['windowWidth']:.0f} L={state['windowCenter']:.0f}")

    glfw.set_key_callback(window, on_key)

    def draw():
        width, height = glfw.get_framebuffer_size(window)
        scene.draw(width, height, state)

    glkit.run(window, draw, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
