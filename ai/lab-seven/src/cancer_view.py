import viewer
import scenes

if __name__ == "__main__":
    viewer.main("cancer", "Breast Cancer Wisconsin", "variant", len(scenes.CancerScene.VARIANTS),
                300, 5, __doc__)
