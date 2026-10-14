import glfw
import viewer

if __name__ == "__main__":
    viewer.run(__doc__, "experiments", "SineNet: експерименти a)–e)", {"group": 1},
               {glfw.KEY_RIGHT: ("group", 1, 1, 5), glfw.KEY_LEFT: ("group", -1, 1, 5)})
