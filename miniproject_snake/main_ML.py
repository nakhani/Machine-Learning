import arcade
from apple import Apple
from snake import Snake
from frog import Frog
from shit import Shit
import numpy as np
from keras.models import load_model

class Game(arcade.Window):
   
   def __init__(self):
       super().__init__(width=400,height=400,title="Super Snake 🐍")
       self.background = arcade.load_texture("photos\—Pngtree—cartoon gray land plant free_4620256.png")
       self.text = arcade.load_texture("photos/a.png")
       self.food = []
       self.snake = Snake(self)
       self.run_update = True
       self.foodd()
       self.flag = 0
       self.direction= self.snake.choose_direction()
       self.model = load_model('model/model.keras')


   def foodd(self):
       self.food.append(Apple(self))
       self.food.append(Frog(self))
       self.food.append(Shit(self))
        
      
   def on_key_release(self, symbol: int, modifiers: int):
            if symbol == arcade.key.Q:
                self.dataset.to_csv("dataset/Dataset.csv", index=False)
                arcade.close_window()
                exit(0)


   def on_draw(self):
       arcade.start_render()
       arcade.set_background_color(arcade.color.KHAKI)
       arcade.draw_texture_rectangle(250,300,500,1000,self.background)
       if self.flag == 1:
           arcade.draw_texture_rectangle(self.width//2,self.height//2,400,250,self.text)
           self.dataset.to_csv("dataset/Dataset.csvDataset.csv", index=False)
           self.run_update = False


       elif self.flag == 0:
           self.snake.draw()
           for food in self.food:
             food.draw()

           score_text = f" Score : { self.snake.score }" 
           arcade.draw_text ( score_text , self.width - 130 , 12 , arcade.color.CG_RED , 12, font_name="Kenney Future" )


       arcade.finish_render()


   def on_update(self, delta_time: float):
        
    if self.run_update == True:
       
        #(self.snake.change_x,self.snake.change_y) = self.direction
        self.snake.move()
        for food in self.food:
          if arcade.check_for_collision(self.snake, food):
             self.snake.eat(food)
             self.food = []
             self.foodd()

        for  i,part in enumerate(self.snake.body):
                for j in range(i + 1, len(self.snake.body)):
                    if part['X'] == self.snake.body[j]['X'] and part['Y'] == self.snake.body[j]['Y']:
                        self.flag = 1

        if (self.snake.center_x < 0 or self.snake.center_x > 400 or 
            self.snake.center_y < 0 or self.snake.center_y > 400 or self.snake.score < 0):
                self.flag = 1

        
        data = [
                self.snake.center_x / 400,
                self.snake.center_y / 400,
                self.food[0].center_x / 400,
                self.food[0].center_y / 400
            ]

        data = np.array([data])
        prediction = np.argmax(self.model.predict(data))
        if prediction == 0:
            self.snake.change_x = 0
            self.snake.change_y = -1
        elif prediction == 1:
            self.snake.change_x = 0
            self.snake.change_y = 1
        elif prediction == 2:
            self.snake.change_x = -1
            self.snake.change_y = 0
        elif prediction == 3:
            self.snake.change_x = 1
            self.snake.change_y = 0


if __name__ == "__main__":
    game = Game()
    arcade.run()