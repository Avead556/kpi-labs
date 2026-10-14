import glfw
from viewer import run_view

if __name__ == "__main__":
    run_view(__doc__, "Криві навчання", "history",
             {"dataset": 0, "epoch": 40},
             {glfw.KEY_RIGHT: ("epoch", 1, 1, 200), glfw.KEY_LEFT: ("epoch", -1, 1, 200),
              glfw.KEY_PAGE_UP: ("epoch", 10, 1, 200), glfw.KEY_PAGE_DOWN: ("epoch", -10, 1, 200),
              glfw.KEY_D: ("dataset", 1, 0, 1), glfw.KEY_S: ("dataset", -1, 0, 1)})
