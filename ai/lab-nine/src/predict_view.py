import viewer

import glfw
import scenes

SAMPLES = 500

if __name__ == "__main__":
    viewer.run("LeNet: передбачення і карти ознак", scenes.PredictScene,
               {"model": 0, "sample": 0, "layer": 0},
               {glfw.KEY_RIGHT: ("sample", 1, SAMPLES - 1),
                glfw.KEY_LEFT: ("sample", -1, SAMPLES - 1),
                glfw.KEY_UP: ("layer", 1, 5),
                glfw.KEY_DOWN: ("layer", -1, 5),
                glfw.KEY_M: ("model", 1, len(scenes.MODELS) - 1),
                glfw.KEY_N: ("model", -1, len(scenes.MODELS) - 1)},
               __doc__)
