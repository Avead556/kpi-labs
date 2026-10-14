import glfw
import viewer

if __name__ == "__main__":
    viewer.run(__doc__, "tabular", "Багатовимірна регресія", {"dataset": 1, "epoch": 2000},
               {glfw.KEY_UP: ("dataset", -1, 1, 2), glfw.KEY_DOWN: ("dataset", 1, 1, 2),
                glfw.KEY_RIGHT: ("epoch", 20, 0, 2000), glfw.KEY_LEFT: ("epoch", -20, 0, 2000)})
