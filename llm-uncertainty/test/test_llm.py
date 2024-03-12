from llm.chat import lm_planner_unct_chat
from llm.gui import GUI
from llm.run import run, translate_to_action
import argparse
import json
import tkinter as tk
def main(args):
    name = "demo_{}".format(args.indx)
    planner = lm_planner_unct_chat()
    with open("./data/det/{}.json".format(name), "r") as f:
        det_res = json.load(f)
    found_object_name = det_res["object_name"]
    objects_xyz = det_res["objects_xyz"]
    person_data = det_res["person_data"]
    found_person_name = det_res["person_name"]
    # check if there is overlapping in found_person_names
    for i, name in enumerate(found_person_name):
        count = found_person_name.count(name)
        if count > 1:
            over_lapped_indx = [i for i, x in enumerate(found_person_name) if x == name]
            humans_data = [person_data[i] for i in over_lapped_indx]
            max_indx = humans_data.index(max(humans_data))
            min_indx = humans_data.index(min(humans_data))
            left_name = "left " + name
            right_name = "right " + name
            found_person_name[over_lapped_indx[max_indx]] = left_name
            found_person_name[over_lapped_indx[min_indx]] = right_name
    # check if there is overlapping in found_object_names
    for i, name in enumerate(found_object_name):
        count = found_object_name.count(name)
        if count > 1:
            over_lapped_indx = [i for i, x in enumerate(found_object_name) if x == name]
            objects_data = [objects_xyz[i][1] for i in over_lapped_indx]
            max_indx = objects_data.index(max(objects_data))
            min_indx = objects_data.index(min(objects_data))
            left_name = "left " + name
            right_name = "right " + name
            found_object_name[over_lapped_indx[max_indx]] = left_name
            found_object_name[over_lapped_indx[min_indx]] = right_name
    print("Found object: ", found_object_name)
    print("Objects xyz: ", objects_xyz)
    print("Person data: ", person_data)
    print("Found person: ", found_person_name)

    gui = GUI(None, found_object_name, found_person_name)
    while gui.data == None:
        gui.root.update()
    goal = gui.data
    gui.data = None
    gui.add_goal(goal)
    gui.root.update()
    planner.objects = found_object_name
    planner.people = found_person_name
    planner.set_goal(goal)
    
    pick, place, _ = run(planner, gui)
    pick_xyz, place_id, pick_name, place_name = translate_to_action(planner, pick, place, found_object_name, found_person_name, objects_xyz, person_data)
    print("pick: ", pick_xyz)
    print("place: ", place_id)
    print("pick_name: ", pick_name)
    print("place_name: ", place_name)
    gui.add_result(pick_name, place_name, pick_xyz, place_id)
    while gui.done == False:
        gui.root.update()
    print("Done")
    data = gui.text.get(0.0, tk.END)
    with open("./data/{}.txt".format(args.indx), "w") as f:
        f.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--indx", type=int, default=1)
    args = parser.parse_args()
    main(args)