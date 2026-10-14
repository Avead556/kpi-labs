import viewer

import glfw
import scenes

if __name__ == "__main__":
    viewer.run("LeNet: розкид між запусками і час шарів", scenes.RunsScene,
               {"device": 0},
               {glfw.KEY_D: ("device", 1, 1),
                glfw.KEY_S: ("device", -1, 1)},
               __doc__)
