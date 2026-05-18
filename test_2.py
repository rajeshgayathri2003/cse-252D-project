import prior
from ai2thor.controller import Controller
from PIL import Image
import matplotlib.pyplot as plt

# Pinned to the pre-5.0-compatible revision; works with ai2thor==5.0.0 from PyPI.
PROCTHOR_REVISION = "ab3cacd0fc17754d4c080a3fd50b18395fae8647"

dataset = prior.load_dataset("procthor-10k", revision=PROCTHOR_REVISION)
dataset

house = dataset["train"][0]
controller = Controller(scene=house)

curr = Image.fromarray(controller.last_event.frame)
curr.save("test.png")
# print(type(house), house.keys(), house)

event = controller.step(action="RotateRight")
Image.fromarray(event.frame).save("test2.png")