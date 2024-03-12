from tkinter import *
import json


class GUI:
    def __init__(self, goal, objects, people):
        self.root = Tk()
        self.text = Text(self.root,height=30,width=100, font=("Helvetica", 16))
        self.text.pack()
        self.yaw = 0.0
        self.indx = 0
        self.text.bind("<Return>", self.add)
        self.text.tag_config('red', foreground='red')
        self.text.tag_config('blue', foreground='blue')
        self.text.tag_config('green', foreground='green')
        self.text.tag_config('brown', foreground='brown')
        self.text.tag_config('orange', foreground='orange')
        self.text.tag_config('purple', foreground='purple')
        self.data = None
        self.button = Button(self.root, text="Done", command=self.done_callback)
        self.button.pack()
        self.done = False
        self.init_perception(goal, objects, people)

    def add_result(self, pick_name, place_name, pick_xyz, place_id):
        self.text.insert(END, "===result== \n", 'red')
        self.text.insert(END, "pick: {}\n".format(pick_xyz), 'red')
        self.text.insert(END, "place: {}\n".format(place_id), 'red')
        self.text.insert(END, "pick_name: {}\n".format(pick_name), 'red')
        self.text.insert(END, "place_name: {}\n".format(place_name), 'red')
        self.indx += 5
        
    def done_callback(self):
        self.done = True
        
    def init_perception(self, goal, objects, people):
        self.text.insert(END, "objects: {}\n".format(objects), 'green')
        self.text.insert(END, "people: {}\n".format(people), 'green')
        if goal != None:
            self.text.insert(END, "goal: {}\n".format(goal), 'red')
        else: 
            self.text.insert(END, "type goal: \n")
        self.indx += 3
        self.goal = goal

    def add_goal(self, goal):
        self.text.insert(END, "goal: {}\n".format(goal), 'red')
        self.indx += 1

    def add_action(self, obj, subj, unct):
        self.text.insert(END, "robot action: robot.pick_and_give({}, {})\n".format(obj, subj), 'brown')
        self.text.insert(END, "uncertainty: {}\n".format(unct), 'blue')
        self.indx += 2

    def add_question(self, feasiblity, reason, question):
        self.text.insert(END, "feasiblity: {}\n".format(feasiblity), 'orange')
        self.text.insert(END, "robot thought: {}\n".format(reason), 'purple')
        self.text.insert(END, "question: {}\n".format(question), 'purple')
        self.indx += 2

    def add(self, event):
        self.indx += 1
        data = self.text.get("end-1c linestart", "end-1c lineend")
        # print(line)
        # data = self.text.get("{}.0".format(self.indx), END)
        data = data.split('\n')[0]
        print(data)
        self.data = data


if __name__ == '__main__':
    gui = GUI('a','b','c')
    gui.root.mainloop()