import csv     # For TableOfAllExercises.from_csv().
import random  # For RehabPlan.random_plan().
from collections import OrderedDict


def main():
    # Load the Search Space.
    table = TableOfAllExercises.from_csv()

    # Ask the user for the target/goal Optimal Plan.
    optimal_plan = ask_for_optimal_plan()

    # Run the Genetic Algorithm.
    smart_rehab = SmartRehab(table, optimal_plan)

    smart_rehab.crossover_rate = 0.95  # 95%
    smart_rehab.mutation_rate = 0.20   # 20%

    smart_rehab.create_initial_population(70)  # 70 individuals/chromosomes.

    # Terminate either when the fitness is 1.0 (perfect)
    # or when we're at the 1,000th Generation.
    for generation in range(1_000):
        if smart_rehab.fittest_fitness >= 1.0:
            break  # Perfect!
        print(smart_rehab.fittest_fitness)
        smart_rehab.evolve()

    # Output the results.
    rehab_plan = smart_rehab.fittest
    fitness = smart_rehab.fittest_fitness

    rehab_plan.print_plan(fitness)

    print('Hope you fast recovery!\n')


def ask_for_optimal_plan():
    name = input_loop('\nWelcome to SmartRehabilitation! What is your name?')

    age_category = input_loop(
        'Hi {}! Please enter your Age Category'
        ' (A for Adult and C for Child).'.format(name),
        {'A': Exercise.ADULT, 'C': Exercise.CHILD})

    condition_type = input_loop(
        'Please enter your Condition Type'
        ' (S for Stroke, SC for Spinal cord, and B for Brain injuries).',
        {'S': Exercise.STROKE, 'SC': Exercise.SPINAL_CORD_INJURY,
         'B': Exercise.BRAIN_INJURY})

    num_of_elbow = int(input_loop(
        'Please enter the number of exercises you prefer to perform for the'
        ' elbow (1 for one type, and 2 for two types of exercises).',
        ['1', '2']))

    num_of_upper_arm = int(input_loop(
        'Please enter the number of exercises you prefer to perform for the'
        ' upper arm (1 for one type, and 2 for two types of exercises).',
        ['1', '2']))

    num_of_knee_lower_leg = int(input_loop(
        'Please enter the number of exercises you prefer to perform for the'
        ' knee/lower leg (1 for one type, and 2 for two types of exercises).',
        ['1', '2']))

    num_of_wrist = int(input_loop(
        'Please enter the number of exercises you prefer to perform for the'
        ' wrist (0 for no exercise, and 1 for one type of exercise).',
        ['0', '1']))

    print('We are working on preparing your optimal rehabilitation plan...\n')

    optimal_plan = OptimalPlan(
        age_category=age_category,
        condition_type=condition_type,

        num_of_elbow=num_of_elbow,
        num_of_upper_arm=num_of_upper_arm,
        num_of_knee_lower_leg=num_of_knee_lower_leg,
        num_of_wrist=num_of_wrist,
    )

    # For testing:
    # print(optimal_plan)

    return optimal_plan


def input_loop(text, options=None):
   
    if(options is not None and
       not(hasattr(options, 'items') and callable(options.items))):
        
        options = {option: option for option in options}

    while True:
        choice = input(text + '\n\n').strip()
        print()

        if options is None:
            if choice:
                return choice

            print('ERROR: Invalid input!')
        else:
            for option, value in options.items():
               
                if choice.lower() == option.lower():
                    return value

            print('ERROR: "{}" is an invalid choice!'.format(choice))


# The Genetic Algorithm and "Population"
#   of "Chromosomes"/"Individuals" (RehabPlans).
class SmartRehab:
    def __init__(self, table, optimal_plan):
        self._table = table
        self._optimal_plan = optimal_plan

        # Allow these to be edited, so public.
        self.crossover_rate = 0.95
        self.mutation_rate = 0.20

    def create_initial_population(self, population_size=70):
        self._population_size = population_size
        #encoding
        self._population = [
            RehabPlan.random_plan(self._table, self._optimal_plan)
            for i in range(population_size)
        ]
        #to initialize array of zeros and it size = population size
        self._fitnesses = [0.0] * population_size

        self.compute_whole_individuals_fitness()

    def compute_whole_individuals_fitness(self):
        self._total_fitness = 0.0

        self._fittest = None
        self._fittest_fitness = 0.0
 #to loop through all the population and caculate the fitness to each one of them
        for i, rehab_plan in enumerate(self._population):
            fitness = rehab_plan.compute_fitness(self._optimal_plan)

            self._fitnesses[i] = fitness
            self._total_fitness += fitness
 #to find the best fitness
            if self._fittest is None or fitness > self._fittest_fitness:
                self._fittest = rehab_plan
                self._fittest_fitness = fitness

        self._average_fitness = self._total_fitness / self._population_size

    def evolve(self):#to create new population for the next generation by crossover and mutation
        wheel_slices = self.build_roulette_wheel_slices()

        # New generation.
        new_population = []

        while len(new_population) < self._population_size:
            rehab_plan = None

            # Reproduce
            if random.random() < self.crossover_rate:
                rehab_plan = self.crossover_by_roulette_wheel(wheel_slices)
            else:
                rehab_plan = self.select_by_roulette_wheel(wheel_slices)#select the indivisual as it is without crossover 

            # Mutate
            if random.random() < self.mutation_rate:
                rehab_plan = rehab_plan.mutate()

            new_population.append(rehab_plan)

        # replace with new generation.
        self._population = new_population

        # Compute fitness of new generation.
        self.compute_whole_individuals_fitness()

    def crossover_by_roulette_wheel(self, wheel_slices):
        mom_index = self.select_index_by_roulette_wheel(wheel_slices)
        dad_index = self.select_index_by_roulette_wheel(
            wheel_slices, mom_index)

        mom = self._population[mom_index]
        dad = self._population[dad_index]

        # Make a child.
        return mom.cross_with(dad)

    def build_roulette_wheel_slices(self):
        wheel_slices = []
        current_sum = 0.0

        # _fitnesses was filled in compute_fitness().
        for fitness in self._fitnesses:
            # Beginning of slice (range).
            wheel_slice = current_sum

            # Avoid divide by 0 and negative (invalid) slices.
            if fitness > 0.0 and self._total_fitness > 0.0:
                # End of slice (range).
                wheel_slice += (fitness / self._total_fitness)

            wheel_slices.append(wheel_slice)

            # Update sum.
            current_sum = wheel_slice

        #Last slice must be 1,
        #because the slices are from range 0.0 to 1.0.
        #because the last one take the last slice so we dont have to compute the slices
        #-1 means the last slice and always the last slice end in 1
        wheel_slices[-1] = 1.0

        return wheel_slices

    def select_by_roulette_wheel(self, wheel_slices, partner_index=-1,
                                 selection_slice=None):
        index = self.select_index_by_roulette_wheel(
            wheel_slices, partner_index, selection_slice)

        return self._population[index]


    @staticmethod
    def select_index_by_roulette_wheel(wheel_slices, partner_index=-1,
                                       selection_slice=None):
        length = len(wheel_slices)
        
        if length <= 1:
            return 0
        #to find the index of mom and dad
        # Pick a random slice .
        if selection_slice is None:
            selection_slice = random.random()

  
        #to get the index of the selected slice 
        #(we will take the next index of random)
        if selection_slice <= wheel_slices[0]:
            selection_index=0
        else:   
          for i in range(0, length, 1):
            if i==length-1:
                selection_index=length-1
                break
            else:
                wheel_slice1 = wheel_slices[i]
                wheel_slice2 = wheel_slices[i+1]
                if  selection_slice >=  wheel_slice1 and selection_slice <= wheel_slice2:
                    selection_index=i+1
                    break

        # Try to avoid same parents
        # where partner_index is the current partner (other parent).
        if selection_index == partner_index:#the dad's turn will be the index of the mom which is the partner (the mom becuse she the first there is no partner so it will be -1)
            if selection_index > 0:#always take the previous slice except if the selection_index =0
                selection_index -= 1
            else :#the selection_index =0 take the next slice
                selection_index += 1  # 0 => 1

        return selection_index

    @property
    def population_size(self):
        return self._population_size

    @property
    def total_fitness(self):
        return self._total_fitness

    @property
    def average_fitness(self):
        return self._average_fitness

    @property
    def fittest(self):
        return self._fittest

    @property
    def fittest_fitness(self):
        return self._fittest_fitness

    def error_value(self):
        return 1.0 - self._fittest_fitness

    def __repr__(self):
        buff = []

        for i, rehab_plan in enumerate(self._population):
            buff.append(
                '{:2}: {} => {}'.format(i, rehab_plan, self._fitnesses[i])
            )

        return '\n'.join(buff)


# One "Chromosome"/"Individual" in the "Population" (SmartRehab).
class RehabPlan:
    # 1 elbow, 1 upper arm, 1 knee/lower leg, 0 wrist.
    MIN_NUM_OF_EXERCISES = 3

    # 2 elbow, 2 upper arm, 2 knee/lower leg, 1 wrist.
    MAX_NUM_OF_EXERCISES = 7

    def __init__(self, table, exercises):
        self._table = table
        self._exercises = exercises

    @classmethod
    def random_plan(klass, table, optimal_plan):
        exercises = []

        for i in range(random.randint(1, 2)):
            exercises.append(table.random_exercise(Exercise.ELBOW))

        for i in range(random.randint(1, 2)):
            exercises.append(table.random_exercise(Exercise.UPPER_ARM))

        for i in range(random.randint(1, 2)):
            exercises.append(table.random_exercise(Exercise.KNEE_LOWER_LEG))

        for i in range(random.randint(0, 1)):
            exercises.append(table.random_exercise(Exercise.WRIST))

        return klass(table, exercises)

  
    def compute_fitness(self, optimal_plan):
        # First, calculate the weighted sums.
        #to calculate the difference between the optimal and what the genetic generate 
        age_category_sum = 0
        condition_type_sum = 0
        num_of_elbow = 0
        num_of_upper_arm = 0
        num_of_knee_lower_leg = 0
        num_of_wrist = 0

 #to get the exercises from the table after we collect the indexs randomly 
        for exercise in self._exercises:
 
# to check if the generated age and condition_type are the same as optimal (what the user entered)       
            if exercise.age_category == optimal_plan.age_category:
                age_category_sum += 1
            if exercise.condition_type == optimal_plan.condition_type:
                condition_type_sum += 1
 #count the number of generated exercises 
            if exercise.body_part == Exercise.ELBOW:
                num_of_elbow += 1
            elif exercise.body_part == Exercise.UPPER_ARM:
                num_of_upper_arm += 1
            elif exercise.body_part == Exercise.KNEE_LOWER_LEG:
                num_of_knee_lower_leg += 1
            elif exercise.body_part == Exercise.WRIST:
                num_of_wrist += 1

 #n => to calculate the difference between the optimal and generated 
        #we use abs to avoid the impact of negative numbers
        #so that the increasing or decreasing of the number of exercises has the same affect
        #for example : if the optimal = 2 and the generated =1 has the same impact  if the optimal = 1 and the generated = 2
        num_of_exercises_sum = (
        abs(optimal_plan.num_of_elbow - num_of_elbow)
         + abs(optimal_plan.num_of_upper_arm - num_of_upper_arm)
         + abs(optimal_plan.num_of_knee_lower_leg - num_of_knee_lower_leg)
         + abs(optimal_plan.num_of_wrist - num_of_wrist)
        )

#noe = number of exercises which mean the sum of the optimal exercises (what the user entered)
        #max_noe_sum = calculate all possible probabilities so that 4 = the number of the body parts in the table
        #num_of_exercises_sum = possible probabilities - difference between the optimal and generated 
        noe = len(self._exercises)

        max_noe_sum = 4 * noe

        num_of_exercises_sum = max_noe_sum - num_of_exercises_sum


  #as mentioed in the questions Age Category and number of Exercises should be equally important, 
        # but half as important as the Condition Type.
#num_of_exercises = len(self)

        fitness = 0.0

        fitness += 0.25 * (age_category_sum / noe)

        fitness += 0.25 * (num_of_exercises_sum / max_noe_sum)

        fitness += 0.50 * (condition_type_sum / noe)



        return fitness

    def cross_with(self, partner):
        child_exercises = []

        # to avoid out of range exception when choosing crossover point 
        # e.g. if mom=3 and dad=6 the crossover point should not exceed 2
        num_of_exercises = min(len(self), len(partner))

       
        #randrange() will be from 1 to num_of_exercises-1
        # prevent identical child to one parent so the random number should start from 1
        crossover_point = random.randrange(1, num_of_exercises)

        
        for i in range(crossover_point):  # ====== cross from first parent
            child_exercises.append(self._exercises[i])

        
        for i in range(crossover_point, len(partner)):  # ===== cross from the second parent
            child_exercises.append(partner._exercises[i])

        # Create the child.
        child = self.__class__(self._table, child_exercises)  # call rehap plan

        return child   # === retarne the child that is generated

    def mutate(self):
        # Make a copy for the new mutant.
        mutated_exercises = self._exercises.copy()
        length = len(mutated_exercises)

        # 1 = add one index
        # 2 = remove one index 
        # 3 = mutate
        randomAction = random.randint(1, 3)

        if randomAction == 1 and length < 7:
            # add one index.
            mutated_exercises.append(self._table.random_exercise())
        else:
            # get a random index.
            mutation_point = random.randrange(0, length)

            if randomAction == 2 and length > 3:
                # remove one index.
                mutated_exercises.pop(mutation_point)
            else:
                # Mutate a random exercise.
                body_part = mutated_exercises[mutation_point].body_part

                mutated_exercises[mutation_point] = (
                    self._table.random_exercise(body_part)
                )

        # Create a new mutant.
        mutant = self.__class__(self._table, mutated_exercises)

        return mutant

    def print_plan(self, fitness=None):
        print(
            'Your rehabilitation plan is ready! Your plan is presented below'
            ' with {} exercises per day.\n'.format(self.len_as_word())
        )

        if fitness:
            print('Fitness: {}, {}, {}\n'.format(
                fitness, str(self), repr(self)))

        # OrderedDict to output in correct order of Body Part.
        plan = OrderedDict([
            [Exercise.ELBOW, []],
            [Exercise.UPPER_ARM, []],
            [Exercise.KNEE_LOWER_LEG, []],
            [Exercise.WRIST, []],
        ])

        for exercise in self._exercises:
            plan[exercise.body_part].append(exercise)

        for body_part, exercises in plan.items():
            buff = '{}:'.format(body_part)

            for i, exercise in enumerate(exercises):
                buff += ' {}. {}.'.format(i + 1, exercise.exercise)

            print(buff + '\n')

    def len_as_word(self):
        words = [
            'zero', 'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten',
        ]

        length = len(self)

        if length < len(words):
            return words[length]
        else:
            return str(length)

    def __len__(self):
        return len(self._exercises)

    def __repr__(self):
        buff = '[ '

        for exercise in self._exercises:
            body_part_exercises = self._table.get_exercises(exercise.body_part)

            buff += '{}{!s:<2s} '.format(
                exercise.body_part[0],
                body_part_exercises.index(exercise),
            )

        return buff + ']'

    def __str__(self):
        buff = '[ '

        for exercise in self._exercises:
            buff += '{}{}{} '.format(
                exercise.body_part[0],
                # Because of 'Stroke' and 'Spinal', can't just use [0],
                #   which would be 'S' for both.
                exercise.condition_type[0:2],
                exercise.age_category[0],
            )

        return buff + ']'


# Inputted values from the User for the Optimal Rehab Plan
#   that we are trying to create using SmartPlan (the Genetic Algorithm).
#
# These are the target values used for the fitness function compute_fitness().
class OptimalPlan:
    def __init__(self, age_category, condition_type, num_of_elbow,
                 num_of_upper_arm, num_of_knee_lower_leg, num_of_wrist):
        if age_category not in Exercise.AGE_CATEGORIES:
            raise ValueError('Invalid age category: ' + age_category)
        if condition_type not in Exercise.CONDITION_TYPES:
            raise ValueError('Invalid condition type: ' + condition_type)

        if num_of_elbow not in [1, 2]:
            raise ValueError('Number of elbow exercises must be 1 or 2: '
                             + num_of_elbow)
        if num_of_upper_arm not in [1, 2]:
            raise ValueError('Number of upper arm exercises must be 1 or 2: '
                             + num_of_upper_arm)
        if num_of_knee_lower_leg not in [1, 2]:
            raise ValueError(
                'Number of knee/lower leg exercises must be 1 or 2: '
                + num_of_knee_lower_leg)
        if num_of_wrist not in [0, 1]:
            raise ValueError('Number of wrist exercises must be 0 or 1: '
                             + num_of_wrist)

        self._age_category = age_category
        self._condition_type = condition_type

        self._num_of_elbow = num_of_elbow
        self._num_of_upper_arm = num_of_upper_arm
        self._num_of_knee_lower_leg = num_of_knee_lower_leg
        self._num_of_wrist = num_of_wrist

        self._num_of_exercises = (
            num_of_elbow
            + num_of_upper_arm
            + num_of_knee_lower_leg
            + num_of_wrist
        )

    @property
    def age_category(self):
        return self._age_category

    @property
    def condition_type(self):
        return self._condition_type

    @property
    def num_of_elbow(self):
        return self._num_of_elbow

    @property
    def num_of_upper_arm(self):
        return self._num_of_upper_arm

    @property
    def num_of_knee_lower_leg(self):
        return self._num_of_knee_lower_leg

    @property
    def num_of_wrist(self):
        return self._num_of_wrist

    @property
    def num_of_exercises(self):
        return self._num_of_exercises

    def __repr__(self):
        return(
            '[ {}, {}'
            ', Elbow: {}, Arm: {}, Knee/Leg: {}, Wrist: {}, Total: {} ]'
            .format(
                self._age_category,
                self._condition_type,
                self._num_of_elbow,
                self._num_of_upper_arm,
                self._num_of_knee_lower_leg,
                self._num_of_wrist,
                self._num_of_exercises,
            )
        )


# The "Search Space" (table of exercise data from CSV file).
class TableOfAllExercises:
    DEFAULT_FILE = 'smart_rehab.csv'

    def __init__(self):
        # OrderedDict so output will match the CSV file for easy checking.
        self._data = OrderedDict()

    @classmethod
    def from_csv(klass, filename=DEFAULT_FILE):
        table = klass()
        table.add_from_csv(filename)

        return table

    def add_from_csv(self, filename=DEFAULT_FILE):
        with open(filename, 'rt') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    self.add_exercise(Exercise(
                        row['Body Part'],
                        row['Exercise'],
                        row['Condition Type'],
                        row['Age Category'],
                    ))
                except ValueError as e:
                    raise ValueError(
                        'Invalid value from CSV row: ' + str(row)) from e

    def add_exercise(self, exercise):
        body_part_exercises = self._data.get(exercise.body_part)

        if body_part_exercises is None:
            body_part_exercises = []
            self._data[exercise.body_part] = body_part_exercises

        body_part_exercises.append(exercise)

    def get_exercises(self, body_part):
        return self._data[body_part]

    def random_exercise(self, body_part=None):
        if body_part is None:
            body_part = random.choice(Exercise.BODY_PARTS_LIST)

        exercises = self._data[body_part]

        return random.choice(exercises)

    def __repr__(self):
        buff = []

        for body_part, exercises in self._data.items():
            for exercise in exercises:
                buff.append('{}'.format(exercise))

        return '\n'.join(buff)


# The data from a CSV row.
class Exercise:
    # Body Parts.
    ELBOW = 'Elbow'
    UPPER_ARM = 'Upper Arm'
    KNEE_LOWER_LEG = 'Knee/Lower leg'
    WRIST = 'Wrist'
    BODY_PARTS = {ELBOW, UPPER_ARM, KNEE_LOWER_LEG, WRIST}
    BODY_PARTS_LIST = list(BODY_PARTS) 

    # Condition Types.
    STROKE = 'Stroke'
    SPINAL_CORD_INJURY = 'Spinal cord injuries'
    BRAIN_INJURY = 'Brain injury'
    CONDITION_TYPES = {STROKE, SPINAL_CORD_INJURY, BRAIN_INJURY}

    # Age Categories.
    ADULT = 'Adult'
    CHILD = 'Child'
    AGE_CATEGORIES = {ADULT, CHILD}

    def __init__(self, body_part, exercise, condition_type, age_category):
        if body_part not in self.BODY_PARTS:
            raise ValueError('Invalid body part: ' + body_part)
        if condition_type not in self.CONDITION_TYPES:
            raise ValueError('Invalid condition type: ' + condition_type)
        if age_category not in self.AGE_CATEGORIES:
            raise ValueError('Invalid age category: ' + age_category)

        self._body_part = body_part
        self._exercise = exercise
        self._condition_type = condition_type
        self._age_category = age_category

    @property
    def body_part(self):
        return self._body_part

    @property
    def exercise(self):
        return self._exercise

    @property
    def condition_type(self):
        return self._condition_type

    @property
    def age_category(self):
        return self._age_category

    def __repr__(self):
        return '[ {}, {}, {}, {} ]'.format(
            self._body_part,
            self._condition_type,
            self._age_category,
            self._exercise,
        )


if __name__ == '__main__':
    main()

