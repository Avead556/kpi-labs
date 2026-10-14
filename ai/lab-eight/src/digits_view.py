import glfw
from viewer import run_view

if __name__ == "__main__":
    run_view(__doc__, "Тестові цифри MNIST", "digits",
             {"index": 0},
             {glfw.KEY_RIGHT: ("index", 1, 0, 9999), glfw.KEY_LEFT: ("index", -1, 0, 9999),
              glfw.KEY_PAGE_UP: ("index", 100, 0, 9999),
              glfw.KEY_PAGE_DOWN: ("index", -100, 0, 9999)})
