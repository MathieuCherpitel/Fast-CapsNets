import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import time
from CapsNet import CapsNet
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json


class Nsga_II:
    def __init__(self, name, n_gen, pop_size, rate_crossover, rate_mutation, rate_local_search, step_size, train_data, test_data) -> None:
        """NSGA-II pareto dominance class for optimising inference and accuracy of CapsNet

        Args:
            n_gen (int): Number of generations
            pop_size (int): Number of individuals in the population
            rate_crossover (int): Rate of crossovers (higher = more frequent)
            rate_mutation (int): Rate of mutations (higher = more frequent)
            rate_local_search (int): Rate of local searches (higher = more frequent)
            step_size (float): Step used for local search
            train_data (tupple): train_data[0] = X_train, train_data[1] = y_train
            test_data (tupple): test_data[0] = X_test, test_data[1] = y_test
        """
        self.name = name
        if os.path.exists(f"../saved_ga/{self.name}"):
            print(f'A ga has been saved under the same name ({self.name}) replace ?')
            res = input("y / n : ")
            if res != "y":
                print("Exiting ...")
                self.valid = False
                return
            else:
                self.valid = True
        else:
            os.makedirs(f"../saved_ga/{self.name}")
            self.valid = True
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.rate_crossover = rate_crossover
        self.rate_mutation = rate_mutation
        self.rate_local_search = rate_local_search
        self.step_size = step_size
        self.train_data = train_data
        self.test_data = test_data

        self.n_var = 7
        self.vars = ['epochs', 'epsilon', 'm_plus', 'm_minus', 'lambda_', 'alpha', 'r']
        self.mins = [1, 0.01, 0.9, 0.05, 0.1, 0.0001, 2]
        self.maxs = [10, 0.1, 0.99, 0.2, 1.0, 0.01, 5]

        self.time_start = None


    def random_pop(self):
        """Initialise pop_size random individuals

        Returns:
            array of shape=(self.pop_size, self.n_var): initialised population
        """
        pop = np.zeros((self.pop_size, self.n_var))
        for i in range(self.pop_size):
            # making sure the epochs and r are integers.
            pop[i][0] = np.random.randint(self.mins[0], self.maxs[0])
            pop[i][-1] = np.random.randint(self.mins[-1], self.maxs[-1])
            for j in range(1, self.n_var - 1):
                pop[i][j] = np.random.uniform(self.mins[j], self.maxs[j])
        return pop

    # Get two parents from the population
    def select_random_parents(self, pop):
        """Get two random parents from the population

        Args:
            pop (np.array): Population pool

        Returns:
            np.array, np.array: parent1 and parent2
        """
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2: # make sure we have two different parents
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        return pop[r1], pop[r2]

    def crossover(self, pop):
        """Randomly generates 2 crossover offsprings from 2 parents by mixing their genes
        The crossover occurs depending on the crossover rate. (higher = more likely)

        Args:
            pop (np.array): population

        Returns:
            np.array: offsprings ready to be added to the population
        """
        offsprings = []
        for _ in range(int(self.pop_size / 2)):
            if self.rate_crossover > np.random.rand():
                p1, p2 = self.select_random_parents(pop)
                crossover_point = np.random.randint(1, pop.shape[1])
                o1 = np.append(p1[:crossover_point], p2[crossover_point:])
                o2 = np.append(p2[:crossover_point], p1[crossover_point:])
                offsprings.append(o1)
                offsprings.append(o2)
        print(f'crossover created {len(offsprings)} offspring(s)')
        return np.array(offsprings)

    def mutation(self, pop):
        """on each itteration, if the mutation occurs,
        out of 2 randomly selected parents, create 2 offsprings by exchanging one gene between parents

        Args:
            pop (np.array): population

        Returns:
            np.array: offsprings ready to be added to the population
        """
        offsprings = []
        for i in range(int(self.pop_size / 2)):
            if self.rate_mutation > np.random.rand():
                p1, p2 = self.select_random_parents(pop)
                cutting_point = np.random.randint(0, pop.shape[1])
                o1 = p1
                o2 = p2
                save = p1[cutting_point]
                o1[cutting_point] = p2[cutting_point]
                o2[cutting_point] = save
                offsprings.append(o1)
                offsprings.append(o2)
        print(f'mutation created {len(offsprings)} offspring(s)')
        return np.array(offsprings)

    def local_search(self, pop):
        """On each itteration if the local search occurs,
        create one offspring by adding fixed coordinates displacement to a randomly selected parent's genes

        Args:
            pop (np.array): population

        Returns:
            np.array: offsprings ready to be added to the population
        """
        offsprings = []
        for _ in range(int(self.pop_size / 2)):
            if self.rate_local_search > np.random.rand():
                r1 = np.random.randint(0, pop.shape[0])
                offspring = pop[r1, :]
                r2 = np.random.randint(0, pop.shape[1])
                offspring[r2] += np.random.uniform(-self.step_size, self.step_size)
                # make sure we stay in bounds
                if offspring[r2] < self.mins[r2]:
                    offspring[r2] = self.mins[r2]
                if offspring[r2] > self.maxs[r2]:
                    offspring[r2] = self.maxs[r2]
                offsprings.append(offspring)
        print(f'local search created {len(offsprings)} offspring(s)')
        return np.array(offsprings)

    def evaluation(self, pop):
        """Create a CapsNet model for each individual in the population,
        evaluate it inference time and accuracy.

        Args:
            pop (np.array): population

        Returns:
            np.array: fitness values with index 0 = inference time and index 1 = accuracy
        """
        fitness_values = np.zeros((len(pop), 2))
        for i, ind in enumerate(pop):
            genotype = {key: value for key, value in zip(self.vars, ind)}
            # making sure we have integer values for epochs and rounds
            genotype['epochs'] = round(genotype['epochs'])
            genotype['r'] = round(genotype['r'])
            # TODO
            genotype['no_of_conv_kernels'] = 128
            genotype['no_of_primary_capsules'] = 32
            genotype['no_of_secondary_capsules'] = 10
            genotype['primary_capsule_vector'] = 8
            genotype['secondary_capsule_vector'] = 16
            
            # build model with found genotype
            model = CapsNet(**genotype)
            print(f'Fitting individual {i + 1}/{len(pop)} with genotype {genotype}')
            model.fit(self.train_data[0], self.train_data[1], tf.keras.optimizers.legacy.Adam())

            # evaluate model
            start = time.time()
            y_preds = model.predict(self.test_data[0])
            end = time.time()
            inference = end - start
            accuracy = accuracy_score(self.test_data[1], y_preds)

            fitness_values[i,0] = inference
            fitness_values[i,1] = accuracy
        return fitness_values

    def pareto_front_finding(self, fitness_values, pop_index):
        """Get the indeces of the best individuals
        The best individuals maximise the accuracy and minimise the inference time

        Args:
            fitness_values (np.array): population fitnesses
            pop_index (np.array): indeces of the population

        Returns:
            np.array: The most fit individuals
        """
        pop_size = fitness_values.shape[0]
        pareto_front = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size): # a solution is dominant if the inference is lower and the accuracy higher
                if (fitness_values[j][0] <= fitness_values[i][0] and fitness_values[j][1] >= fitness_values[i][1]) and \
                        (fitness_values[j][0] < fitness_values[i][0] or fitness_values[j][1] > fitness_values[i][1]):
                    pareto_front[i] = 0
                    break
        return pop_index[pareto_front]

    # Estimate how tightly clumped fitness values are on Pareto front. 
    def crowding_calculation(self, fitness_values):
        """Estimates how close some fitness values are

        Args:
            fitness_values (np.array): population fitnesses

        Returns:
            float: crowding distance
        """
        pop_size = len(fitness_values[:, 0])
        fitness_value_number = len(fitness_values[0, :])
        matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
        normalized_fitness_values = (fitness_values - fitness_values.min(0))/fitness_values.ptp(0)
        
        for i in range(fitness_value_number):
            crowding_results = np.zeros(pop_size)
            crowding_results[0] = 1
            crowding_results[pop_size - 1] = 1
            sorted_normalized_fitness_values = np.sort(normalized_fitness_values[:,i])
            sorted_normalized_values_index = np.argsort(normalized_fitness_values[:,i])
            crowding_results[1:pop_size - 1] = (sorted_normalized_fitness_values[2:pop_size] - sorted_normalized_fitness_values[0:pop_size - 2])
            re_sorting = np.argsort(sorted_normalized_values_index)
            matrix_for_crowding[:, i] = crowding_results[re_sorting]
        
        crowding_distance = np.sum(matrix_for_crowding, axis=1)
        return crowding_distance

    def remove_using_crowding(self, fitness_values, number_solutions_needed):
        """Removes the solutions that are too close to ensure diversity on the Pareto front

        Args:
            fitness_values (np.array): population fitnesses
            number_solutions_needed (int): number of solutions needed

        Returns:
            np.array: remaining individuals
        """
        pop_index = np.arange(fitness_values.shape[0])
        crowding_distance = self.crowding_calculation(fitness_values)
        selected_pop_index = np.zeros(number_solutions_needed)
        selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))    # arr(num_sol_needed x 2)
        for i in range(number_solutions_needed):
            pop_size = pop_index.shape[0]
            solution_1 = rn.randint(0, pop_size - 1)
            solution_2 = rn.randint(0, pop_size - 1)
            if crowding_distance[solution_1] >= crowding_distance[solution_2]:
                # solution 1 is better than solution 2
                selected_pop_index[i] = pop_index[solution_1]
                selected_fitness_values[i, :] = fitness_values[solution_1, :]
                pop_index = np.delete(pop_index, (solution_1), axis=0)
                fitness_values = np.delete(fitness_values, (solution_1), axis=0)
                crowding_distance = np.delete(crowding_distance, (solution_1), axis=0)
            else:
                # solution 2 is better than solution 1
                selected_pop_index[i] = pop_index[solution_2]
                selected_fitness_values[i, :] = fitness_values[solution_2, :]
                pop_index = np.delete(pop_index, (solution_2), axis=0)
                fitness_values = np.delete(fitness_values, (solution_2), axis=0)
                crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)
        
        selected_pop_index = np.asarray(selected_pop_index, dtype=int)
        return selected_pop_index

    def selection(self, pop, fitness_values):
        """Perform pareto front selection to have a population equal to pop_size

        Args:
            pop (np.array): populationb
            fitness_values (np.array): current population fitnesses

        Returns:
            np.array: selected population
        """
        pareto_front_index = []

        pop_index_0 = np.arange(self.pop_size)
        pop_index = np.arange(self.pop_size)
        
        while len(pareto_front_index) < self.pop_size:
            new_pareto_front = self.pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
            total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

            # check the size of pareto_front, if larger than pop_size then remove some
            if total_pareto_size > self.pop_size:
                number_solutions_needed = self.pop_size - len(pareto_front_index)
                selected_solutions = self.remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
                new_pareto_front = new_pareto_front[selected_solutions]
            
            pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
            remaining_index = set(pop_index) - set(pareto_front_index)
            pop_index_0 = np.array(list(remaining_index))

        selected_pop = pop[pareto_front_index.astype(int)]
        return selected_pop

    def plot_nsga(self, metrics):
        """Saves a scatter plot of the fitnesses in ./saved_ga

        Args:
            metrics (list(dict)): nas metrics
        """
        # Remove possible outliers in the inference
        df = pd.DataFrame([(gen, metrics[gen][i]['inference'], metrics[gen][i]['accuracy'])
                        for gen in metrics
                        for i in range(len(metrics[gen]))],
                        columns=['Generation', 'Inference', 'Accuracy'])

        inf_threshold = np.mean(df['Inference']) + np.std(df['Inference'])
        df_filtered = df[df['Inference'] < inf_threshold]

        filtered_metrics = {}
        for _, row in df_filtered.iterrows():
            generation = row['Generation']
            inference = row['Inference']
            accuracy = row['Accuracy']
            
            if generation not in filtered_metrics:
                filtered_metrics[generation] = []
            
            filtered_metrics[generation].append({'inference': inference, 'accuracy': accuracy})

        metrics = filtered_metrics

        fig, ax = plt.subplots()
        self.plot_individuals(ax, metrics)

        legend = ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=(len(metrics) + 9) // 10)
        ax.set_title('Ga optimisation metrics')

        print(f'Saving GA metrics under {self.name}/generations_metrics.png')
        fig.savefig(f'../saved_ga/{self.name}/generations_metrics.png', bbox_extra_artists=(legend,), bbox_inches='tight')

    def plot_individuals(self, ax, metrics):
        for key, value in metrics.items():
            x = [v['accuracy'] for v in value]
            y = [v['inference'] for v in value]
            plt.scatter(x, y, label=key)
        ax.grid('on')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Inference')
        ax.invert_yaxis()

    def plot_solutions(self, fitness_values):
        metrics = {}
        for index, fitness in enumerate(fitness_values):
            metrics[f'Solution {index+1}'] = [{
                'accuracy': fitness[1],
                'inference': fitness[0]
            }]

        fig, ax = plt.subplots()
        self.plot_individuals(ax, metrics)

        legend = ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=(len(metrics) + 9) // 10)
        ax.set_title('Ga solutions metrics')

        print(f'Saving GA solutions under {self.name}/solutions_metrics.png')
        fig.savefig(f'../saved_ga/{self.name}/solutions_metrics.png', bbox_extra_artists=(legend,), bbox_inches='tight')

    def save_solutions(self, solutions, fitnesses):
        res = {}
        for index in range(len(solutions)):
            res[f'Solution {index + 1}'] = {
                'accuracy': fitnesses[index][1],
                'inference': fitnesses[index][0],
                'params': {key: value for key, value in zip(self.vars, solutions[index])}
            }

        with open(f'../saved_ga/{self.name}/solutions.json', 'w') as f:
            f.write(json.dumps(res))

    def save_params(self):
        params = {
            'n_gen': self.n_gen,
            'pop_size': self.pop_size,
            'rate_crossover': self.rate_crossover,
            'rate_mutation': self.rate_mutation,
            'rate_local_search': self.rate_local_search,
            'step_size': self.step_size
        }

        with open(f'../saved_ga/{self.name}/nas_params.json', 'w') as f:
            f.write(json.dumps(params))

    def optimise(self):
        """Performs self.n_gen itterations of evolutionary algorithm to optimise population

        Returns:
            list: Best individuals found
        """
        self.time_start = datetime.now()
        pop = self.random_pop()
        metrics = {}
        for i in range(self.n_gen):
            print(f"=== Generation {i+1}/{self.n_gen}")

            print(f"Len of pop before manipulations : {len(pop)}")
            crossover_offsprings = self.crossover(pop)
            mutation_offsprings = self.mutation(pop)
            local_search_offsprings = self.local_search(pop)
            if crossover_offsprings.size > 0:
                pop = np.append(pop, crossover_offsprings, axis=0)
            if mutation_offsprings.size > 0:
                pop = np.append(pop, mutation_offsprings, axis=0)
            if local_search_offsprings.size > 0:
                pop = np.append(pop, local_search_offsprings, axis=0)

            print(f"Len of pop after manipulations : {len(pop)}")
            fitness_values = self.evaluation(pop)
            metrics[f'Generation {i+1}'] = [{"inference": value[0], "accuracy": value[1]} for value in fitness_values]


            pop = self.selection(pop, fitness_values)
            print(f"Len of pop after selection : {len(pop)}")
        fitness_values = self.evaluation(pop)
        index = np.arange(pop.shape[0]).astype(int)
        pareto_front_index = self.pareto_front_finding(fitness_values, index)
        solutions = pop[pareto_front_index, :]
        self.plot_nsga(metrics)
        print("Optimal solution(s):")
        print(solutions)
        fitness_values = fitness_values[pareto_front_index]
        print("______________")
        print("Fitness values:")
        print("  Inference    Accuracy")
        print(fitness_values)
        print(f"=== GA done : {datetime.now()} ===")

        self.plot_solutions(fitness_values)
        self.save_solutions(solutions, fitness_values)
        self.save_params()


# Optimal solution(s):
# [[9.00000000e+00 4.88360436e-02 9.39566399e-01 1.36611055e-01
#   4.91056369e-01 1.00000000e-04 2.00000000e+00]
#  [4.00000000e+00 5.49037474e-02 9.48021275e-01 7.78102753e-02
#   2.94636268e-01 9.37210247e-03 2.00000000e+00]
#  [1.00000000e+00 5.99024912e-02 9.06820830e-01 9.35568090e-02
#   2.16236020e-01 8.19128145e-03 2.00000000e+00]]
# ______________
# Fitness values:
#   Inference    Accuracy
# [[5.57220721 0.95507812]
#  [5.56180882 0.94335938]
#  [5.39654398 0.76367188]]