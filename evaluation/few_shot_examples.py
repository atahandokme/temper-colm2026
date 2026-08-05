"""Few-shot CoT exemplars for GSM8K and MultiArith."""

# Standard 8-shot CoT exemplars for GSM8K (from Wei et al., 2022)
GSM8K_FEW_SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. #### 33

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8"""

# Few-shot for MultiArith (simple arithmetic, 4 examples sufficient)
MULTIARITH_FEW_SHOT = """Q: There are 64 students trying out for the school's trivia teams. If 36 of them didn't get picked for the team and the rest were put into 4 groups, how many students would be in each group?
A: 64 students tried out and 36 didn't get picked, so 64 - 36 = 28 students were picked. They were put into 4 groups, so each group has 28 / 4 = 7 students. #### 7

Q: A pet store had 78 puppies. In one day they sold 30 of them and put the rest into cages with 8 in each cage. How many cages did they use?
A: They started with 78 puppies and sold 30, so 78 - 30 = 48 puppies were left. With 8 in each cage, they used 48 / 8 = 6 cages. #### 6

Q: Robin's hair was 14 inches long. She got it cut by 11 inches. How long is her hair now?
A: Robin's hair was 14 inches long. After cutting 11 inches, it is 14 - 11 = 3 inches long. #### 3

Q: Rachel had to complete 2 pages of math homework and 3 pages of reading homework. How many pages did she have to complete total?
A: Rachel had 2 pages of math and 3 pages of reading. 2 + 3 = 5 pages total. #### 5"""

# Few-shot CoT for ARC-Challenge (4 exemplars covering different science topics)
ARC_FEW_SHOT = """Q: Which of the following is the best conductor of electricity?
A. wood  B. rubber  C. copper  D. glass
A: Let me think step by step. Conductors of electricity are materials that allow electrons to flow freely. Wood, rubber, and glass are all insulators that resist electrical flow. Copper is a metal with free electrons that conduct electricity very well. The answer is C.

Q: What causes day and night on Earth?
A. the tilt of Earth's axis  B. the rotation of Earth on its axis  C. the revolution of Earth around the Sun  D. the distance between Earth and the Sun
A: Let me think step by step. Day and night occur because one side of Earth faces the Sun while the other faces away. This cycle happens every 24 hours due to Earth spinning on its axis, which is called rotation. Revolution around the Sun causes seasons, not day and night. The tilt of the axis also relates to seasons. The answer is B.

Q: A student pushes a box across the floor. Which form of energy is produced by the friction between the box and the floor?
A. light  B. heat  C. sound  D. magnetic
A: Let me think step by step. When two surfaces rub against each other, friction converts kinetic energy into thermal energy. The box sliding on the floor creates friction, which produces heat. While friction can also produce some sound, heat is the primary form of energy produced. The answer is B.

Q: Which property of an object can be determined just by looking at it?
A. mass  B. color  C. weight  D. temperature
A: Let me think step by step. Mass requires a scale to measure. Weight also requires measurement. Temperature requires a thermometer. Color, however, can be observed directly with our eyes without any instruments. The answer is B."""


def get_few_shot_prompt(question, dataset):
    """Build a few-shot CoT prompt for the given dataset."""
    if dataset in ["gsm8k", "gsm8k_test"]:
        exemplars = GSM8K_FEW_SHOT
    elif dataset in ["multiarith"]:
        exemplars = MULTIARITH_FEW_SHOT
    elif dataset in ["arc_challenge"]:
        exemplars = ARC_FEW_SHOT
    else:
        # For other datasets, fall back to zero-shot CoT
        return None

    return f"{exemplars}\n\nQ: {question}\nA:"
