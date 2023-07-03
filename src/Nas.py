"""
    Multi objective NAS for CapsNet:

    We want to optimise the following parameters :
        epochs
        epsilon
        m_minus
        m_plus
        lambda_
        alpha
        no_of_conv_kernels
        no_of_primary_capsules
        primary_capsule_vector
        no_of_secondary_capsules
        secondary_capsule_vector
        r

    Our goal is to increase the accuracy and reduce the inference time of the model. (2 goals)

    Note : We don't have to deal with network architecture since we always have 2 layers of capsules. Just the number of capsules per layer and the hyperparameters.
    
    
    Pseudo Code :
        1. Initialise random population of models

        2. Until the number of genreations is reached :
            2.2. Evaluate each individual in the population, assign fitness values based on their performance on accuracy and inference time

            2.3. Multi objective parent selection (Pareto dominance)

            2.4. Apply genetic operators (Crossover and mutation) to generate offsprings

            2.5. Evaluate the offsprings

            2.6. Replace the less fit individuals with the offsprings

        3. Select the best individual(s) using Pareto dominance

        4. Perform additional evaluation

        5. Return the best individual
        
"""
import numpy as np
from tqdm import tqdm
from CapsNet import CapsNet
import time
import tensorflow as tf
from typing import List
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, index, genotype) -> None:
        self.index = index
        self.fitnesses = [None, None] # accuracy, inference, epochs
        self.model = CapsNet(**genotype)

class Nas:
    def __init__(self, train_data, test_data, pop_size=10):

        self.X_train = train_data[0]
        self.X_test = test_data[0]

        self.y_train = train_data[1]
        self.y_test = test_data[1]

        self.pop_size = pop_size
        self.population = []

        self.current_gen = 0

        self.metrics = {}
    
    def initialise_population(self) -> None:
        """
        Initialise a population of self.pop_size Individuals respoecting the following rules :

        epochs : 5 < epochs < 1000
        epsilon : 0.01 < e < 0.1
        m_plus : 0.9 to 0.99
        m_minus : 0.05 to 0.2 times m_plus
        lambda_ : 0.1 to 1.0 (usually 0.5)
        alpha : 0.0001 to 0.01 usually (0.0005)
        r: 2 to 5 (usually 3)
        # TODO
        no_of_conv_kernels
        no_of_primary_capsules
        primary_capsule_vector
        no_of_secondary_capsules
        secondary_capsule_vector
        """

        for i in range(self.pop_size):
            genotype = {}
            genotype['epochs'] = np.random.randint(low=1, high=2)
            genotype['epsilon'] = np.random.uniform(0.01,0.1)
            genotype['m_plus'] = np.random.uniform(0.9, 0.99)
            genotype['m_minus'] = np.random.uniform(0.05, 0.2) * genotype['m_plus']
            genotype['lambda_'] = np.random.uniform(0.1, 1.0)
            genotype['alpha'] = np.random.uniform(0.0001, 0.01)
            genotype['r'] = np.random.randint(low=2, high=5)
            # TODO
            genotype['no_of_conv_kernels'] = 128
            genotype['no_of_primary_capsules'] = 32
            genotype['no_of_secondary_capsules'] = 10
            genotype['primary_capsule_vector'] = 8
            genotype['secondary_capsule_vector'] = 16

            self.population.append(Individual(i, genotype))

    def train_population(self):
        """Train the population
        """
        for index, individual in enumerate(self.population):
            print(f"Training individual {index + 1}/{len(self.population)}")
            individual.model.fit(self.X_train, self.y_train, tf.keras.optimizers.legacy.Adam())

    def evaluate(self, individuals):
        """Assignes fitness to individuals. Accuracy and inference are evaluated.

        Args:
            individuals (list(Individuals)): list of individuals to be evaluated
            pbar (pbar): Used to update progress bar

        Returns:
            list(Individuals): individuals with their fitnesses assigned
        """
        for index, individual in enumerate(individuals):
            print(f"Evaluating individual {index + 1}/{len(individuals)}")
            start = time.time()
            y_preds = individual.model.predict(self.X_test)
            end = time.time()
            individual.fitnesses[0] = accuracy_score(self.y_test, y_preds)
            individual.fitnesses[1] = end - start
        metrics = {
            'accuracy': [individual.fitnesses[0] for individual in individuals],
            'inference': [individual.fitnesses[1] for individual in individuals],
        }
        self.metrics[f'generation_{self.current_gen}'] = metrics
        return individuals

    def select_parents(self):
        """Select the best individuals from the population using Pareto Dominance

        Returns:
            list(Individuals): best individuals from the current population
        """
        return []

    def create_offsprings(self, parents):
        """Apply genetic operators (crossover and mutation) on parents to generate offsprings

        Args:
            parents (list(Individuals)): individuals used as base for offsprings

        Returns:
            list(Individuals): generated offsprings
        """
        return []

    def replace_with_offsprings(self, offsprings):
        """Replaces less fit individuals from the population with offsprings

        Args:
            offsprings (list(Individuals)): list of offsprings
        """
        pass

    def get_best_individuals(self):
        """Return best individuals from pareto front

        Returns:
            list(Individuals): best individuals
        """
        return []

    def get_best_fitness(self):
        """Gets the fitness of the best individual in self.population

        Returns:
            list: the two fitness values
        """
        return [0.0233, 0.0213]

    def optimise(self, n_gen):
        """Entry point of the Genetic Algorithm. Generates a random population, assess it and modify it to
        find the best solution.

        Args:
            n_gen (int): number of itterations
        """
        self.initialise_population()
        print("Initialised", len(self.population), "individuals")

        for i in range(1, n_gen + 1, 1):
            print(f"\n==== Generation  {i}/{n_gen} ====")
            self.current_gen = i
            print("Training population...")
            self.train_population()
            print("Evaluating population...")
            self.population = self.evaluate(self.population)
            # parents = self.select_parents()
            # offsprings = self.create_offsprings(parents)
            # offsprings = self.evaluate(offsprings)
            # self.replace_with_offsprings(offsprings)
            # print(f"Best fitnesses : {self.get_best_fitness()}")

    def plot(self):
        cmap = plt.get_cmap('viridis')
        colors = np.linspace(0, 1, len(self.metrics))
        for i, (generation, values) in enumerate(self.metrics.items()):
            accuracy = values['accuracy']
            inference = values['inference']
            color = cmap(colors[i])
            plt.scatter(accuracy, inference, colorcl=color, label=generation)
        plt.xlabel('Accuracy')
        plt.ylabel('Inference')
        plt.title('GA Scatter Plot')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=len(self.metrics))
        plt.savefig('saved_ga/ga_scatter_plot.png')

def main():
    (X_train, y_train), (X_test , y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_train = tf.cast(X_train, dtype=tf.float32)
    X_train = tf.expand_dims(X_train, axis=-1)

    X_test = X_test / 255.0
    X_test = tf.cast(X_test, dtype=tf.float32)
    X_test = tf.expand_dims(X_test, axis=-1)

    y_train = y_train.reshape((len(y_train),))
    y_test = y_test.reshape((len(y_test),))

    # Reducing the size of the samples for faster training
    X_train = X_train[:512]
    y_train = y_train[:512]

    X_test = X_test[:512]
    y_test = y_test[:512]
    
    nas = Nas((X_train, y_train), (X_test , y_test), pop_size=10)
    nas.optimise(4)
    nas.plot()

if __name__ == "__main__":
    main()
