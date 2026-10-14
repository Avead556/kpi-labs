import viewer
import scenes

if __name__ == "__main__":
    viewer.run(scenes.ArchitectureScene, "ПР №11: архітектура і параметри",
               {"dataset": 0, "model": 0}, "model", (0, 4), __doc__)
