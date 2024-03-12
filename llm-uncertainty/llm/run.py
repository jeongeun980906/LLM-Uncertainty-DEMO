from llm.chat import lm_planner_unct_chat
from llm.gui import GUI
import numpy as np

def run(planner:lm_planner_unct_chat, gui: GUI):
    planner.reset()
    unct = 100
    fes = True
    while unct > 0.3:
        pick, place, unct= infer_unct(planner, gui)
        if unct > 0.3:
            planner.append(pick, place)
            reason, question, feasiblity = planner.question_generation()
            # planner.append_reason_and_question(reason, question)
            gui.add_question(feasiblity, reason, question)
            if question != None:
                gui.root.update()
                while gui.data == None:
                    gui.root.update()
                answer = gui.data
                gui.data = None
                planner.append(None, None, answer)
            else: fes = False; break
    return pick, place, fes

def translate_to_action(planner: lm_planner_unct_chat,\
                        pick, place, object_names, people_names,objects_xyz, people_data):
    print(pick, object_names, place, people_names)
    pick = planner.translate(pick, object_names)
    # print(place)
    # if 'left' in place or 'right' in place:
    #     if 'left' in place: left = True
    #     else: left = False
    #     new_place = place.replace("left", "").replace("right", "").replace("on", "")
    #     place = planner.translate(new_place, people_names)
    #     candidate_indxs = [i for i, x in enumerate(people_names) if x == new_place]
    #     candidate_data = [people_data[i] for i in candidate_indxs]
    #     if left: # choose max in candidate_data
    #         place_id = max(candidate_data)
    #     else: place_id = min(candidate_data)
    # else:
    place = planner.translate(place, people_names)
    place_id = people_data[people_names.index(place)]
    pick_id = object_names.index(pick)
    pick_xyz = objects_xyz[pick_id]
    return pick_xyz, place_id, pick, place

def infer_unct(planner, gui: GUI):
    tasks, scores , unct = planner.plan_with_unct()
    if tasks != None:
        scores = np.asarray(scores)
        idxs= np.argsort(scores).tolist()
        idxs.reverse()
        print(tasks, scores)
        for idx in idxs:
            string = tasks[idx]
            if "robot." in string:
                pick, place = tasks[idx].replace("robot action: robot.pick_and_give(", "").replace(")", "").split(", ")
                break
        gui.add_action(pick, place, unct['total'])
        gui.root.update()
        return pick, place, unct['total']
    else:
        return None, None