#!/usr/bin/env python3


from smart_rehab import *


table = TableOfAllExercises.from_csv()

optimal_plan = OptimalPlan(
    age_category=Exercise.ADULT,
    condition_type=Exercise.BRAIN_INJURY,
    num_of_elbow=1,
    num_of_upper_arm=2,
    num_of_knee_lower_leg=1,
    num_of_wrist=1,
)

smart_rehab = SmartRehab(table, optimal_plan)

smart_rehab.create_initial_population(10)


def print_exercise(name, exercise):
    print('{!s:10s}:  {} = {!r:20s} => {}'.format(
        name, exercise, exercise,
        exercise.compute_fitness(optimal_plan)))


# -----------------------------------------------------------------------------

# mom = smart_rehab._population[0]
# dad = smart_rehab._population[1]

# child = mom.cross_with(dad)

# print_exercise('Mom', mom)
# print_exercise('Dad', dad)
# print_exercise('=> Child', child)
# print()

# mutant = child.mutate()

# print_exercise('Child', child)
# print_exercise('=> Mutant', mutant)
# print()

# -----------------------------------------------------------------------------

print(smart_rehab)
print()

wheel_slices = smart_rehab.build_roulette_wheel_slices()
begin_slice = 0.0

for index, wheel_slice in enumerate(wheel_slices):
    chance = (wheel_slice - begin_slice) * 100.0
    print('{:2} => {:<22} # {:>5.2f}%'.format(index, wheel_slice, chance))
    begin_slice = wheel_slice

print()

for i in range(20):
    random_slice = random.random()

    index = smart_rehab.select_index_by_roulette_wheel(
        wheel_slices, selection_slice=random_slice)

    print(' {:22s} => {}'.format(str(random_slice), index))

# -----------------------------------------------------------------------------

# smart_rehab.create_initial_population(100)

# for generation in range(100):
#     print('Generation #{}...'.format(generation + 1))
#     print('  Total fitness:  {}'.format(smart_rehab.total_fitness))
#     print('  Avg fitness:    {}'.format(smart_rehab.average_fitness))
#     print('  Fittest:        {}'.format(smart_rehab.fittest_fitness))
#     print('                  {}'.format(smart_rehab.fittest))
#     print('                  {}'.format(repr(smart_rehab.fittest)))

#     if smart_rehab.fittest_fitness >= 1.0:
#         print()
#         break
#     if input() in ['q', 'e', 'x']:
#         break

#     smart_rehab.evolve()
