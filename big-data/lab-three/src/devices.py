import os
from pathlib import Path

import owlready2
from owlready2 import (
    AllDisjoint,
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    Thing,
    ConstrainedDatatype,
    get_ontology,
    locstr,
    OwlReadyInconsistentOntologyError,
    World,
    sync_reasoner,
)

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "results" / "devices.owl"
IRI = "http://kpi.ua/ontologies/devices.owl"
HOMEBREW_JAVA = Path("/opt/homebrew/opt/openjdk@21/bin/java")

onto = get_ontology(IRI)


def label(entity, uk, comment=None):
    entity.label = [locstr(uk, lang="uk")]
    if comment:
        entity.comment = [locstr(comment, lang="uk")]


with onto:
    class Device(Thing): pass
    class PortableDevice(Device): pass
    class StationaryDevice(Device): pass
    class Smartphone(PortableDevice): pass
    class Tablet(PortableDevice): pass
    class Laptop(PortableDevice): pass
    class Smartwatch(PortableDevice): pass
    class DesktopComputer(StationaryDevice): pass

    class Component(Thing): pass
    class Screen(Component): pass
    class Processor(Component): pass
    class Battery(Component): pass
    class Camera(Component): pass
    class Memory(Component): pass
    class Storage(Component): pass

    class Manufacturer(Thing): pass
    class OperatingSystem(Thing): pass

    class hasComponent(ObjectProperty):
        domain = [Device]
        range = [Component]
    class isComponentOf(ObjectProperty):
        inverse_property = hasComponent
    class hasScreen(hasComponent):
        range = [Screen]
    class hasProcessor(hasComponent):
        range = [Processor]
    class hasBattery(hasComponent):
        range = [Battery]
    class hasCamera(hasComponent):
        range = [Camera]
    class hasMemory(hasComponent):
        range = [Memory]
    class hasStorage(hasComponent):
        range = [Storage]
    class manufacturedBy(ObjectProperty, FunctionalProperty):
        domain = [Device]
        range = [Manufacturer]
    class manufactures(ObjectProperty):
        inverse_property = manufacturedBy
    class runsOn(ObjectProperty):
        domain = [Device]
        range = [OperatingSystem]
    class developedBy(ObjectProperty, FunctionalProperty):
        domain = [OperatingSystem]
        range = [Manufacturer]

    class modelName(DataProperty, FunctionalProperty):
        domain = [Device]
        range = [str]
    class releaseYear(DataProperty, FunctionalProperty):
        domain = [Device]
        range = [int]
    class screenSizeInches(DataProperty, FunctionalProperty):
        domain = [Screen]
        range = [float]
    class resolution(DataProperty, FunctionalProperty):
        domain = [Screen]
        range = [str]
    class clockSpeedGHz(DataProperty, FunctionalProperty):
        domain = [Processor]
        range = [float]
    class coreCount(DataProperty, FunctionalProperty):
        domain = [Processor]
        range = [int]
    class capacityMAh(DataProperty, FunctionalProperty):
        domain = [Battery]
        range = [int]
    class megapixels(DataProperty, FunctionalProperty):
        domain = [Camera]
        range = [float]
    class sizeGB(DataProperty, FunctionalProperty):
        domain = [Memory | Storage]
        range = [int]

    AllDisjoint([Device, Component, Manufacturer, OperatingSystem])
    AllDisjoint([Smartphone, Tablet, Laptop, Smartwatch])
    AllDisjoint([PortableDevice, StationaryDevice])
    AllDisjoint([Screen, Processor, Battery, Camera, Memory, Storage])

    PortableDevice.is_a += [hasBattery.some(Battery)]
    Smartphone.is_a += [hasScreen.exactly(1, Screen), hasProcessor.some(Processor),
                        hasBattery.exactly(1, Battery), hasCamera.some(Camera),
                        runsOn.some(OperatingSystem)]
    Tablet.is_a += [hasScreen.exactly(1, Screen), hasProcessor.some(Processor)]
    Laptop.is_a += [hasScreen.exactly(1, Screen), hasProcessor.some(Processor),
                    hasMemory.some(Memory), hasStorage.some(Storage)]
    Smartwatch.is_a += [hasScreen.exactly(1, Screen), hasCamera.max(0, Camera)]
    DesktopComputer.is_a += [hasProcessor.some(Processor), hasScreen.max(0, Screen)]

    class CameraDevice(Device):
        equivalent_to = [Device & hasCamera.some(Camera)]
    class LargeScreenDevice(Device):
        equivalent_to = [Device & hasScreen.some(
            Screen & screenSizeInches.some(ConstrainedDatatype(float, min_inclusive=10.0)))]

onto.metadata.label = [locstr("Онтологія технічних пристроїв", lang="uk")]
onto.metadata.comment = [locstr(
    "Технічні пристрої (смартфон, планшет, ноутбук, смарт-годинник, настільний комп'ютер), "
    "їхні компоненти, виробники й операційні системи. Варіант 5 лабораторної роботи №3 "
    "з дисципліни «Обробка надвеликих масивів даних»", lang="uk")]
onto.metadata.versionInfo = ["1.0"]

for entity, uk, comment in [
    (Device, "Технічний пристрій", "Електронний пристрій, яким користується людина"),
    (PortableDevice, "Портативний пристрій", "Пристрій з власним акумулятором, який носять із собою"),
    (StationaryDevice, "Стаціонарний пристрій", "Пристрій, що працює від мережі на одному місці"),
    (Smartphone, "Смартфон", None), (Tablet, "Планшет", None), (Laptop, "Ноутбук", None),
    (Smartwatch, "Смарт-годинник", None), (DesktopComputer, "Настільний комп'ютер", None),
    (Component, "Компонент", "Складова частина пристрою"),
    (Screen, "Екран", None), (Processor, "Процесор", None), (Battery, "Акумулятор", None),
    (Camera, "Камера", None), (Memory, "Оперативна пам'ять", None), (Storage, "Накопичувач", None),
    (Manufacturer, "Виробник", None), (OperatingSystem, "Операційна система", None),
    (CameraDevice, "Пристрій з камерою", "Визначений клас: пристрій, що має хоча б одну камеру"),
    (LargeScreenDevice, "Пристрій з великим екраном", "Визначений клас: екран від 10 дюймів"),
    (hasComponent, "має компонент", None), (isComponentOf, "є компонентом", None),
    (hasScreen, "має екран", None), (hasProcessor, "має процесор", None),
    (hasBattery, "має акумулятор", None), (hasCamera, "має камеру", None),
    (hasMemory, "має оперативну пам'ять", None), (hasStorage, "має накопичувач", None),
    (manufacturedBy, "виготовлений", None), (manufactures, "виготовляє", None),
    (runsOn, "працює під", None), (developedBy, "розроблена", None),
    (modelName, "модель", None), (releaseYear, "рік випуску", None),
    (screenSizeInches, "розмір екрана, дюймів", None), (resolution, "роздільна здатність", None),
    (clockSpeedGHz, "частота процесора, ГГц", None), (coreCount, "кількість ядер", None),
    (capacityMAh, "ємність акумулятора, мА·год", None), (megapixels, "роздільність камери, Мп", None),
    (sizeGB, "обсяг, ГБ", None),
]:
    label(entity, uk, comment)

DEVICES = [
    ("iPhone15", Smartphone, "iPhone 15", 2023, "Apple", "iOS",
     (6.1, "2556x1179"), ("A16Bionic", 3.46, 6), 3349, 48.0, 6, 128),
    ("GalaxyS24", Smartphone, "Galaxy S24", 2024, "Samsung", "Android",
     (6.2, "2340x1080"), ("Exynos2400", 3.2, 10), 4000, 50.0, 8, 128),
    ("GalaxyTabS9", Tablet, "Galaxy Tab S9", 2023, "Samsung", "Android",
     (11.0, "2560x1600"), ("Snapdragon8Gen2", 3.36, 8), 8400, 13.0, 8, 128),
    ("MacBookAirM3", Laptop, "MacBook Air 13 M3", 2024, "Apple", "macOS",
     (13.6, "2560x1664"), ("AppleM3", 4.05, 8), 4614, 2.1, 8, 256),
]
OS_MAKERS = {"iOS": "Apple", "macOS": "Apple", "Android": "Google"}

with onto:
    makers = {name: Manufacturer(name) for name in ["Apple", "Samsung", "Google"]}
    systems = {}
    for name, maker in OS_MAKERS.items():
        systems[name] = OperatingSystem(name)
        systems[name].developedBy = makers[maker]
    for (name, cls, model, year, maker, os_name, (inches, res), (cpu, ghz, cores),
         mah, mp, ram, storage) in DEVICES:
        device = cls(name)
        device.modelName = model
        device.releaseYear = year
        device.manufacturedBy = makers[maker]
        device.runsOn = [systems[os_name]]
        screen = Screen(f"{name}_Screen", screenSizeInches=inches, resolution=res)
        processor = Processor(cpu, clockSpeedGHz=ghz, coreCount=cores)
        device.hasScreen = [screen]
        device.hasProcessor = [processor]
        device.hasBattery = [Battery(f"{name}_Battery", capacityMAh=mah)]
        device.hasCamera = [Camera(f"{name}_Camera", megapixels=mp)]
        device.hasMemory = [Memory(f"{name}_RAM", sizeGB=ram)]
        device.hasStorage = [Storage(f"{name}_SSD", sizeGB=storage)]


def main():
    OUT.parent.mkdir(exist_ok=True)
    onto.save(file=str(OUT), format="rdfxml")
    print(f"saved {OUT.relative_to(LAB)}")
    print(f"classes {len(list(onto.classes()))}, object properties "
          f"{len(list(onto.object_properties()))}, data properties "
          f"{len(list(onto.data_properties()))}, individuals {len(list(onto.individuals()))}")
    if os.environ.get("JAVA_HOME"):
        owlready2.JAVA_EXE = str(Path(os.environ["JAVA_HOME"]) / "bin" / "java")
    elif HOMEBREW_JAVA.exists():
        owlready2.JAVA_EXE = str(HOMEBREW_JAVA)
    with onto:
        sync_reasoner(infer_property_values=True, debug=0)
    print("HermiT: ontology is consistent")
    for cls in [CameraDevice, LargeScreenDevice]:
        print(f"{cls.name}: {sorted(i.name for i in cls.instances())}")
    print(f"Apple manufactures: {sorted(d.name for d in makers['Apple'].manufactures)}")
    print(f"AppleM3 isComponentOf: {[d.name for d in onto.AppleM3.isComponentOf]}")
    for cls, prop, part in [(Smartwatch, "hasCamera", Camera), (DesktopComputer, "hasScreen", Screen)]:
        world = World()
        copy = world.get_ontology(OUT.resolve().as_uri()).load()
        with copy:
            bad = copy[cls.name]("Broken" + cls.name)
            setattr(bad, prop, [copy[part.name]("Broken" + part.name)])
        try:
            sync_reasoner(world, debug=0)
            print(f"negative test {cls.name}.{prop}: not detected")
        except OwlReadyInconsistentOntologyError:
            print(f"negative test: {cls.name} with {prop} -> HermiT: inconsistent")


if __name__ == "__main__":
    main()
