from llm.chat import lm_planner_unct_chat
from llm.gui import GUI
from llm.run import run, translate_to_action
import argparse
import json, copy
import tkinter as tk

def main(args):
    planner = lm_planner_unct_chat()
    if args.move_robot:
        name = "demo_{}".format(args.indx)
        with open("./data/det/{}.json".format(name), "r") as f:
            det_res = json.load(f)
        found_object_name = det_res["object_name"]
        objects_xyz = det_res["objects_xyz"]
        person_data = det_res["person_data"]
        found_person_name = det_res["person_name"]
    else:
        from vision.from_image import gpt4_v_helper
        det = gpt4_v_helper()
        print("Detecting from previous image")
        obj_cand = ['fanta can','monster can', 'coffee can', 'lemon', 'redbull can',
                      'apple', 'starbucks can', 'orange', 'pepsi can', 'coca cola can']
        file_path = "./data/det/ori_{}.jpg".format(args.indx)
        found_object_name,_,_ = det.detection(file_path, obj_cand)
        print(found_object_name)
        det.reset(); det.set_system_prompt()
        human_cand = ['person wearing green shirt', 'person wearing yellow shirt',
                    'person wearing black shirt', 'person wearing blue shirt', 'person wearing red shirt', 'person wearing brown shirt']
        file_path = "./data/det/human_ori_{}.jpg".format(args.indx)
        found_person_name = det.detection(file_path, human_cand, True)
        print(found_person_name)
        objects_xyz = ['none']*len(found_object_name); person_data = [i for i in range(len(found_person_name))]
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
    ori_obj_name = copy.deepcopy(found_object_name)
    ori_obj_xyz = copy.deepcopy(objects_xyz)
    ori_person_name = copy.deepcopy(found_person_name)
    ori_person_data = copy.deepcopy(person_data)

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
    
    pick, place, fes = run(planner, gui)
    if fes:
        pick_xyz, place_id, pick_name, place_name = translate_to_action(planner, pick, place, ori_obj_name, ori_person_name, ori_obj_xyz, ori_person_data)
        print("pick: ", pick_xyz)
        print("place: ", place_id)
        print("pick_name: ", pick_name)
        print("place_name: ", place_name)
        gui.add_result(pick_name, place_name, pick_xyz, place_id)
    else: 
        gui.add_result("infeasible", "infeasible", "infeasible", "infeasible")
    while gui.done == False:
        gui.root.update()
    print("Done")
    """
    Move robot
    """
    if args.move_robot:
        from model.robot_client import RobotClient
        import rclpy
        rclpy.init()
        client = RobotClient()
        for _ in range(10):
            rclpy.spin_once(client, timeout_sec=1.0)
        if not fes:
            client.infeasible_callback()
        else:
            client.pick(pick_xyz)
            client.give(place_id)
            if place_id ==3 or place_id ==0:
                client.reset_capture_pose()
            else:
                client.reset_from_back()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--indx", type=int, default=1)
    parser.add_argument("--move_robot", type=int, default=1)
    args = parser.parse_args()
    main(args)