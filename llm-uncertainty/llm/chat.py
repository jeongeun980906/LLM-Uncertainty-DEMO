import openai
import spacy
import scipy
import random
import numpy as np
import copy
import time
from llm.prompts import get_prompts
from openai import OpenAI
class lm_planner_unct_chat():
    def __init__(self, type = 2, example = False,key_file_path='./key/key.txt'):
        api_key = self.set_openai_api_key_from_txt(key_file_path,VERBOSE=False)
        self.client = OpenAI(api_key=api_key)
       
        self.few_shots = get_prompts(example)
        self.type = type
        self.new_lines = ""
        self.nlp = spacy.load('en_core_web_lg')
        self.type = type
        self.verbose = True
        self.objects = ["blue block", "red block", "yellow bowl", "green block", "green bowl",'blue bowl']
        self.people = ["person with yellow shirt", "person with black shirt", "person with white shirt"]
    
    def set_openai_api_key_from_txt(self, key_path='./key/key.txt',VERBOSE=True):
        """
            Set OpenAI API Key from a txt file
        """
        with open(key_path, 'r') as f: 
            OPENAI_API_KEY = f.read()
        if VERBOSE:
            print("OPENAI KEY SET")
        return OPENAI_API_KEY
    

    def plan_with_unct(self, verbose= False):
        obj_cand = []
        subj_cand = []
        self.verbose = verbose
        goal_num = 5
        while len(obj_cand) <2 or len(subj_cand)<2:
            for _ in range(goal_num):
                if self.type == 2:
                    self.sample_prompt()
                object, subject = self.inference()
                if len(object) != 0:
                    obj_cand += object
                if len(subject) != 0:
                    subj_cand += subject
        tasks = []
        scores = []
        for x,y in zip(obj_cand, subj_cand):
            prompt = 'robot action: robot.pick_and_give({}, {})'.format(x,y)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1
        scores = [s/sum(scores) for s in scores]
        # print(obj_cand,subj_cand)
        obj2 = self.get_word_diversity(obj_cand)
        sub2 = self.get_word_diversity(subj_cand)
        if str(obj2) == 'nan': obj2 = 0
        if str(sub2) == 'nan': sub2 = 0
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /10,
            'sub': sub2/10,
            'total': (obj2+sub2)/10
        }

        return tasks, scores, unct

    def inference(self):
        # print(self.prompt)
        while True:
            try:
                response = self.client.chat.completions.create(
                model="gpt-4-0613", 
                messages=[{"role": "user", "content": self.prompt}], 
                temperature = 1.0, top_p = 1, n = 3, stop=')'
                )
                break
            except:
                time.sleep(1)
                continue
        # print(response)
        objects = []
        subjects = []
        results = response.choices
        for res in results:
            res = res.message.content
            res = res.split("\n")
            if res[0] == "":
                try:
                    res = res[1]
                except:
                    continue
            else:
                res = res[0]
            print(res)
            if "robot action: done()" in res:
                objects.append("done")
                subjects.append("done")
            if "robot.pick_and_give" not in res:
                continue
            try: res = res.split(":")[-1]
            except: pass
            try:
                pick, place = res.replace("robot.pick_and_give(", "").replace(")", "").split(", ")
            except: continue
            pick = pick.split("\n")[-1]
            place = place.split("\n")[0]
            if pick[-1] == " ":
                pick = pick[:-1]
            if place[-1]==" ":
                place = place[:-1]
            for i, l in enumerate(pick):
                if l != " ":
                    pick = pick[i:]
                    break
            for i, l in enumerate(place):
                if l != " ":
                    place = place[i:]
                    break
            objects.append(pick)
            subjects.append(place)
        return objects, subjects

    def set_goal(self, goal):
        self.goal = goal

    def set_prompt(self,choices=None):
        des = ""
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        temp = ""
        for e, obj in enumerate(self.objects):
            temp += obj
            if e != len(self.objects)-1:
                temp += ", "
        temp2 = ""
        for e, obj in enumerate(self.people):
            temp2 += obj
            if e != len(self.people)-1:
                temp2 += ", "
        des += "task: considering the ambiguity of the goal, "
        des += self.goal + "\n"
        des += "# Do not generate objects. Please pick or give the object from the following list \n"
        des += '# Do not generate outside of the robot.pick_and_give(a,b) format'
        # des += "\n where the place object is not dependent from the selected pick object \n"
        des += "scene: objects = [" + temp + "] \n"
        des += "scene: people = [" + temp2+ "] \n"
        # des += "\n The order can be changed"
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def sample_prompt(self):
        lengs = len(self.few_shots)
        # print(lengs)
        k = random.randrange(4,lengs+1)
        A = np.arange(lengs)
        A = np.random.permutation(A)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        random.shuffle(self.objects)
        random.shuffle(self.people)
        self.set_prompt(choices)
        # print(self.prompt)

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\nrobot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += 'robot thought: what can I ask to the user?\n question: please ' + question + '\n'

    def append(self, object, subject, task=None):
        if task == None:
            next_line = "\n" + "    robot action: robot.pick_and_give({}, {})".format(object, subject)
        else:
            next_line = "\n" + "answer: "+ task
        self.new_lines += next_line


    def translate(self, object, candidates):
        object = object.replace("_", " ")
        vec = self.nlp(object).vector
        vec1 = np.reshape(vec, (1,-1))
        vec2 = []
        for word in candidates:
            rw = word.replace("_"," ")
            vec = self.nlp(rw).vector
            vec2.append(vec)
        vec2 = np.vstack(vec2)
        # print(vec2.shape, vec1.shape)
        dis = scipy.spatial.distance_matrix(vec1,vec2)[0] # n
        argmin = np.argmin(dis)
        print(dis, argmin)
        return candidates[argmin]

    def get_word_diversity(self, words):
        print(words)
        vecs = []
        size = len(words)
        for word in words:
            word = word.replace("_", " ")
            vec = self.nlp(word).vector
            vecs.append(vec)
        vecs = np.vstack(vecs)
        dis = scipy.spatial.distance_matrix(vecs,vecs)
        div = np.sum(dis)/((size)*(size-1))
        print(div, dis)
        return div

    def question_generation(self):
        form = '\nrobot thought: I am a robot that can pick an object and give it to someone. Considering the action set, pick and give, can I {} \
            if the user gives more information? Answer in Yes or No'.format(self.goal)
        form += "\nrobot answer: "
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        # inp += self.new_lines
        inp += form
        while True:
            try:
                response = self.client.chat.completions.create(
                model="gpt-4-0613", 
                messages=[{"role": "user", "content": inp}], 
                temperature = 0.5, top_p = 1, n = 1, stop=':'
                )
                break
            except:
                time.sleep(1)
                continue
        affor = response.choices[0].message.content.split('\n')[0]
        print(affor)
        temp = affor.lower().replace(".","").replace(",","").split(' ')
        if 'no' in temp or 'cannot' in temp or 'can not' in temp or "can't" in temp:
            return None, None, False
        form = '\nrobot thought: this code is uncertain because '
        self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        response = self.client.chat.completions.create(
            model="gpt-4-0613", 
            messages=[{"role": "user", "content": inp}], 
            temperature = 1.0, top_p = 1, n = 1, stop=':'
            )
        reason = response.choices[0].message.content.split('\n')[0]
        print('reason: ',reason)
        inp += reason
        self.new_lines += reason + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        response = self.client.chat.completions.create(
            model="gpt-4-0613", 
            messages=[{"role": "user", "content": inp}], 
            temperature = 1.0, top_p = 1, n = 1, stop='\n'
            )
        ques = response.choices[0].message.content
        ques = ques.split('\n')[0]
        print('question: please',ques)
        self.new_lines += ques
        return reason, ques, True
    
    def reset(self):
        self.new_lines = ""