import viewer

import glfw
import scenes

if __name__ == "__main__":
    viewer.run("LeNet: криві навчання", scenes.CurvesScene,
               {"model": 0, "epoch": 20},
               {glfw.KEY_RIGHT: ("epoch", 1, 25),
                glfw.KEY_LEFT: ("epoch", -1, 25),
                glfw.KEY_M: ("model", 1, len(scenes.MODELS) - 1),
                glfw.KEY_N: ("model", -1, len(scenes.MODELS) - 1)},
               __doc__)
