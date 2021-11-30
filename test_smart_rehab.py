#!/usr/bin/env python3


from smart_rehab import *


# exc = Exercise(Exercise.WRIST, 'Crawling',  Exercise.STROKE, Exercise.CHILD)
# print(exc)

# -----------------------------------------------------------------------------

# table = TableOfAllExercises.from_csv('smart_rehab.csv')

# table = TableOfAllExercises.from_csv()

# table = TableOfAllExercises()
# table.add_from_csv('smart_rehab.csv')

# print(table)

# -----------------------------------------------------------------------------

# table = TableOfAllExercises()

# print(table)

# table.add_exercise(exc)
# table.add_exercise(
#     Exercise(Exercise.WRIST, 'Crawling',  Exercise.STROKE, Exercise.ADULT))

# print(table)

# print(table.get_exercise(0))
# print(table.get_exercise(1))

# print(len(table))

# -----------------------------------------------------------------------------

# optimal_plan = OptimalPlan(
#     age_category=Exercise.ADULT,
#     condition_type=Exercise.BRAIN_INJURY,
#     num_of_elbow=1,
#     num_of_upper_arm=2,
#     num_of_knee_lower_leg=1,
#     num_of_wrist=1,
# )

# -----------------------------------------------------------------------------

# rehab_plan = RehabPlan.random_plan(table, optimal_plan)

# print(rehab_plan._exercises)
# print(rehab_plan)
# print()

# for exercise in rehab_plan._exercises:
#     print('{}'.format(exercise))
# print()

# rehab_plan.print_plan()
# print()
# print(rehab_plan.compute_fitness(optimal_plan))
# print()
