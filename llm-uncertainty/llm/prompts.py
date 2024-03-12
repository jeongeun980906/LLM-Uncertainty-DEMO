PROMPTS = [
        """
        task: give a person wearing a blue shirt a pepsi can.
        scene: objects = [pepsi can, coca cola can, apple, banana]
        scene: people = [person wearing blue shirt, person wearing black shirt]
        robot action: robot.pick_and_give(pepsi can, person wearing blue shirt)
        robot action: done()
        """
        ,
        """
        task: give a person wearing ligt colored shirt a monster can.
        scene: objects = [pepsi can, coca cola can, monster can, lime]
        scene: people = [person wearing yellow shirt, person wearing black shirt]
        robot action: robot.pick_and_give(monster can, person wearing yellow shirt)
        robot action: done()
        """
        ,
        """
        task: give a person wearing a white shirt a coca cola can.
        scene: objects = [starbucks can, coca cola can, monster can, lime]
        scene: people = [person wearing red shirt, person wearing white shirt]
        robot action: robot.pick_and_give(coca cola can, person wearing white shirt)
        robot action: done()
        """
        ,
        """
        task: give a person wearing black shirt a monster can.
        scene: objects = [pepsi can, coca cola can, starbucks can, lime, apple]
        scene: people = [person wearing green shirt, person wearing black shirt]
        robot action: robot.pick_and_give(monster can, person wearing black shirt)
        robot action: done()
        """
        ]


def get_prompts(example=False):
    prompts = PROMPTS

    if example:
        prompts[2] = """
    task: give a person wearing ligt colored shirt a monster can.
    scene: objects = [pepsi can, coca cola can, monser can, lime]
    scene: people = [person wearing yellow shirt, person wearing black shirt]
    robot thought: This is code is uncertain because I don't know whom to give.
    robot thought: What can I ask to the user?
    question: Which person do you want to give monster can?
    answer: person wearing yellow shirt
    robot action: robot.pick_and_give(monster can, person wearing yellow shirt)
    robot action: done()
    """
    return prompts


OBJECT_SETS = [
    'apple','strawberry','banana','orange',"plate", 
    'lemon','lime','peach','coca cola can', 'pepsi can', 
    'starbucks can','monster can']

PERSON_SETS = [
    "person with blue shirt", "person with gray shirt",
    "person with yellow shirt", "person with black shirt", "person with white shirt",

]