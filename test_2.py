import prior
from ai2thor.controller import Controller
from PIL import Image
import matplotlib.pyplot as plt

dataset = prior.load_dataset("procthor-10k")
dataset

house = dataset["train"][0]
controller = Controller(scene=house)

curr = Image.fromarray(controller.last_event.frame)
curr.save("test.png")
# print(type(house), house.keys(), house)

event = controller.step(action="RotateRight")
Image.fromarray(event.frame).save("test2.png")