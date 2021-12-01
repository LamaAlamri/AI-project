import csv

from smart_rehab import *


def main():
    optimal_plan = OptimalPlan(
        age_category=Exercise.ADULT,
        condition_type=Exercise.BRAIN_INJURY,
        num_of_elbow=2,
        num_of_upper_arm=2,
        num_of_knee_lower_leg=2,
        num_of_wrist=1,
    )

    experiment = Experiment(optimal_plan)


    experiment.pop_sizes = [10, 40, 70]
    experiment.crossover_rates = [0.55, 0.75, 0.95]
    experiment.mutation_rates = [0.05, 0.12, 0.20]

    experiment.run('./graph_data.csv')


class Experiment:

    def __init__(self, optimal_plan):
        self.table = TableOfAllExercises.from_csv()
        self.optimal_plan = optimal_plan

        self.smart_rehab = SmartRehab(self.table, self.optimal_plan)

    def run(self, csv_filename):
        with open(csv_filename, 'wt', newline='', encoding='utf-8') \
                as csv_file:
            csv_writer = csv.writer(csv_file)

            all_results = [[] for i in range(len(self.pop_sizes))]

            pop_i = 0
            cross_i = 0
            mutate_i = 0

            while pop_i < len(self.pop_sizes):
                results = self.run_ga(pop_i, cross_i, mutate_i)

                all_results[pop_i].extend(results)
                mutate_i += 1

                if mutate_i >= len(self.mutation_rates):
                    mutate_i = 0
                    cross_i += 1

                    if cross_i >= len(self.crossover_rates):
                        cross_i = 0
                        pop_i += 1

            for i in range(len(all_results[0])):
                csv_row = []

                for j in range(len(all_results)):
                    csv_row.append(all_results[j][i])

                csv_writer.writerow(csv_row)

    def run_ga(self, pop_i, cross_i, mutate_i):
        pop_size = self.pop_sizes[pop_i]
        crossover_rate = self.crossover_rates[cross_i]
        mutation_rate = self.mutation_rates[mutate_i]

        title = '[p: {}, c: {}, m: {}]'.format(
            pop_size, crossover_rate, mutation_rate

        )

        # "Runs: Run your GA 20 times and report the average fitness."
        times = 20
        gens = 1_000  # generations.
        per_gen = 100  # per generation to record result.

        # +1 to get 1,000th generation.
        avg_fits = [0.0 for i in range((gens // per_gen) + 1)]

        for t in range(times):
            self.smart_rehab.crossover_rate = crossover_rate
            self.smart_rehab.mutation_rate = mutation_rate

            self.smart_rehab.create_initial_population(pop_size)

            print('\r{}... time #{}   '.format(title, t + 1), end='')

            # +1 to get 1,000th generation.
            for g in range(gens + 1):
                if (g % per_gen) == 0:#to get the muliple of 100
                    pg = g // per_gen
                    avg_fits[pg] += self.smart_rehab.average_fitness

                self.smart_rehab.evolve()

        print()

        results = [title]

        # Turn the sums into averages and add to results.
        for avg_fit in avg_fits:
            avg_fit = avg_fit / times

            results.append(avg_fit)
            print('best: {}'.format(self.smart_rehab.fittest_fitness))
        return results


if __name__ == '__main__':
    main()
