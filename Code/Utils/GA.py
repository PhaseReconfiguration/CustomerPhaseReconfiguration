import pygad
import numpy as np

class GA:

    def __init__(self, feederbalancing, reconstruct=False) -> None:
        self.feederbalancing = feederbalancing
        self.okay = 0
        self.total = 0
        self.mutation_rate = 0.1
        self.keep_parents = None
        self.num_generations = 60 # Number of generations.
        self.population_size = 50 # Number of solutions in the population.
        self.best_solution = None
        self.mistakes = []
        self.fitnesses = []

        self.reconstruct = reconstruct
        self.feeder = None
        self.initial_solution = None
        self.customer_mappings = None
        self.customer_mapping = None

    def generate_individual(self, constraints):
        return [np.random.choice(c) for c in constraints]

    def reconstruct_solution(self, solution):
        temp_solution = self.initial_solution.copy()
        for i,c in enumerate(self.customer_mapping):
            temp_solution[c] = solution[i]
        return temp_solution

    def fitness_func(self, ga_instance, solution, solution_idx):
        if(self.reconstruct):
            solution = self.reconstruct_solution(solution)
        B = self.feederbalancing.get_B_from_genetic(solution)
        self.total+=1
        if(self.feederbalancing.check_constraint_feasible_configuration(B) == -1):
            self.mistakes.append(solution.copy())
            return 0
        self.okay+=1
        loss = self.feederbalancing.objective_function(B)
        fitness = 1.0 / (loss + 1e-9)
        return fitness

    def on_generation(self, ga_instance):
        tmp_best_solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        self.fitnesses.append(ga_instance.last_generation_fitness)
        print(f"Generation = {ga_instance.generations_completed}/{self.num_generations}")
        diff = tmp_best_solution[1] if not self.best_solution else tmp_best_solution[1] - self.best_solution[1]
        if(diff>0):
            print(f"New solution    = {tmp_best_solution[0]}")
            print(f"Fitness         = {tmp_best_solution[1]} (diff: {diff})")
            self.best_solution = tmp_best_solution
            if(self.reconstruct):            
                B_sol = self.feederbalancing.get_B_from_genetic(
                    self.reconstruct_solution(self.best_solution[0])
                    )
            else:
                B_sol = self.feederbalancing.get_B_from_genetic(self.best_solution[0])
            print(f'Solution loss: {self.feederbalancing.objective_function(B_sol, False)} ({self.feederbalancing.objective_function(B_sol)}). N. changes: {np.sum(B_sol * self.feederbalancing.B_init_opposite)}.')
            print()
            mutation_rate = self.mutation_rate
        else:
            mutation_rate = np.random.rand() * 0.8 + 0.05 


        #Sort population
        population_with_fitness = list(zip(ga_instance.last_generation_fitness, ga_instance.population))
        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x[0], reverse=True)
        sorted_fitness, sorted_population = zip(*sorted_population_with_fitness)
        ga_instance.population = np.array(sorted_population)
        ga_instance.last_generation_fitness = np.array(sorted_fitness)

        for i,gene in enumerate(ga_instance.population[self.keep_parents:]):
            for cust, conf in enumerate(gene):
                if(self.reconstruct):
                    feasible_configs = self.feederbalancing.B_feas_nobinary_per_customer[self.customer_mapping[cust]]
                else:
                    feasible_configs = self.feederbalancing.B_feas_nobinary_per_customer[cust]
                if(conf not in feasible_configs or np.random.rand() < mutation_rate): #Ensure that all the genes are feasible + add mutation
                    ga_instance.population[self.keep_parents+i,cust] = np.random.choice(feasible_configs)
        if(self.best_solution):
            # ga_instance.population[self.keep_parents+1] = self.best_solution[0] #Add solution since to assure feasability you may remove the best solution
            ga_instance.population[0] = self.best_solution[0] #Add solution since to assure feasability you may remove the best solution

    def initialize_run(self):
        num_parents_mating = max(4, int(self.population_size*0.4)) # Number of solutions to be selected as parents in the mating pool.
        self.keep_parents = np.max([4,int(self.population_size*0.1)])
        
        self.customer_mappings = self.feederbalancing.feeder_index_eans
        if(self.reconstruct):
            self.customer_mapping = self.customer_mappings[self.feeder]
            num_genes = len(self.customer_mapping)
            constraints = [self.feederbalancing.B_feas_nobinary_per_customer[i] for i in self.customer_mapping]
            print(f"Approximated solution! Solving for feeder: {self.feeder}, customers: {self.customer_mapping}")
        else:
            constraints = self.feederbalancing.B_feas_nobinary_per_customer
            num_genes = len(self.feederbalancing.net.asymmetric_load)

        possible_combinations = 1
        for i in constraints:
            possible_combinations = possible_combinations * len(i)
        tested_combinations = self.num_generations*self.population_size
        print(f"There are {possible_combinations:e} possible combinations to test (good luck with that!!). Solutions that will be tested: {tested_combinations} ({(tested_combinations/possible_combinations*100):.2f}%)")
        print(f"Initial solution: {self.feederbalancing.B_init_nobinary}. Number customers: {len(self.feederbalancing.B_init_nobinary)}\n")

        initial_population = [self.generate_individual(constraints) for _ in range(self.population_size)]
        initial_population[0] = [self.feederbalancing.B_init_nobinary[j] for j in self.customer_mapping] if self.reconstruct else self.feederbalancing.B_init_nobinary

        ga_instance = pygad.GA(num_generations=self.num_generations,
                            num_genes=num_genes,
                            gene_type=int,
                            sol_per_pop=self.population_size,
                            num_parents_mating=num_parents_mating,
                            keep_parents=self.keep_parents,
                            keep_elitism=self.keep_parents,
                            save_solutions=True,
                            save_best_solutions=True,
                            initial_population=initial_population,
                            fitness_func=self.fitness_func,
                            on_generation=self.on_generation,
                            random_seed=14,
                            suppress_warnings=True
                            )
        return ga_instance

    def runGA(self):
        ga_instance = self.initialize_run()

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        ga_instance.plot_fitness()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = self.best_solution
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        if(self.reconstruct):            
                B_sol = self.feederbalancing.get_B_from_genetic(
                    self.reconstruct_solution(solution)
                    )
        else:
            B_sol = self.feederbalancing.get_B_from_genetic(solution)
            
        print(f'Solution loss: {self.feederbalancing.objective_function(B_sol, False)} ({1/solution_fitness}). N. changes: {np.sum(B_sol * self.feederbalancing.B_init_opposite)}.')

        if ga_instance.best_solution_generation != -1:
            print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

        # Saving the GA instance.
        filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
        # ga_instance.save(filename=filename)

        # # Loading the saved GA instance.
        # loaded_ga_instance = pygad.load(filename=filename)
        # loaded_ga_instance.plot_fitness()

        print(self.okay, self.total, self.okay/self.total)
        self.ga_instance = ga_instance
