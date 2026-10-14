import glfw
import viewer

if __name__ == "__main__":
    viewer.run(__doc__, "sine", "SineNet: sin(x) і прогноз мережі", {"epoch": 2000},
               {glfw.KEY_RIGHT: ("epoch", 10, 0, 2000), glfw.KEY_LEFT: ("epoch", -10, 0, 2000),
                glfw.KEY_UP: ("epoch", 100, 0, 2000), glfw.KEY_DOWN: ("epoch", -100, 0, 2000)})
