import glfw
from viewer import run_view

if __name__ == "__main__":
    run_view(__doc__, "Методи оптимізації", "optimizers",
             {"epoch": 20, "focus": 0, "metric": 0},
             {glfw.KEY_RIGHT: ("epoch", 1, 1, 20), glfw.KEY_LEFT: ("epoch", -1, 1, 20),
              glfw.KEY_UP: ("focus", 1, 0, 6), glfw.KEY_DOWN: ("focus", -1, 0, 6),
              glfw.KEY_M: ("metric", 1, 0, 2), glfw.KEY_N: ("metric", -1, 0, 2)})
