import random
from animals import Animals


class Apple(Animals):
    def __init__(self,game):
        super().__init__("photos\pngwing.com.png")
        self.width = 40
        self.height = 40
        self.center_x = random.randint(10,game.width-10) // 8 *8
        self.center_y = random.randint(10,game.height-10) // 8 * 8
        self.change_x = 0
        self.change_y = 0
        self.score = 1