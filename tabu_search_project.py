import sys
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np

#hyper parameters
TEMPERATURE_INITIAL = 11500
STOP_TEMPERATURE = 1
DECREASE_TEMPERATURE = 0.999


# --------------------------- Simulated Annealing ---------------------------
def updateTemp(bestScore, newScore, worstScore):
    bestScore = min(bestScore, newScore)
    worstScore = max(worstScore, newScore)
    return bestScore, worstScore


def FirstPermotationOfCities(numberOfCities):
    permutation = np.arange(numberOfCities)
    np.random.shuffle(permutation)
    permutation.append(0)
    return permutation

# for case we want simple first same solution for 2 algo
def permutationConstantFirst(numberOfCities):
    permutation = np.arange(numberOfCities)
    permutation = np.append(permutation, 0)
    return permutation

def EvaluateSolution(citiesArr, solution):
    distPoints = 0
    for i in range(len(solution)):
        a = solution[i]
        b = solution[i - 1]
        xDelta = citiesArr[a][0] - citiesArr[b][0]
        yDelta = citiesArr[a][1] - citiesArr[b][1]
        distPoints += (xDelta * xDelta + yDelta * yDelta) ** 0.5
    return distPoints


def changePosition(currentSolution):
    lenSolution = len(currentSolution)
    a = np.random.randint(lenSolution)
    b = np.random.randint(lenSolution)
    while b == a:
        b = np.random.randint(lenSolution)
    newArr = currentSolution.copy()
    newArr[a], newArr[b] = newArr[b], newArr[a]
    return newArr

#####################GRAPHS#################

def graphCurrentCost(epochs_list, currentCostList):
    plt.bar(epochs_list, currentCostList, width=0.5, color='purple')
    plt.ylabel('cost of current score')
    plt.xlabel('iteration')
    plt.title('Simulated Annealing solving TSP :cost of current solution')
    plt.show()

def graphBestCost(epochs_list, bestScore):
    plt.scatter(epochs_list, bestScore, c='coral')
    plt.ylabel('cost of best score')
    plt.xlabel('iteration')
    plt.title('Simulated Annealing solving TSP :cost of best solution')
    plt.show()

def graphTemperature(epochs_list, temperature):
    plt.bar(epochs_list, temperature, width = 0.5, color='purple')
    plt.ylabel('Value of temperature')
    plt.xlabel('iteration')
    plt.title('Temperature vs iteration')
    plt.show()

def compare2Algorithms(tabu, simulated, epochs):
    plt.plot(epochs, tabu, color='b', label='tabu search')
    plt.plot(epochs, simulated, color='r', label='simulated annealing')
    plt.ylabel('cost of best score')
    plt.xlabel('iteration')
    plt.title('Simulated Annealing VS tabu search solving TSP :cost of best solution')
    plt.legend()
    plt.show()

def compare2Algorithms(tabu, simulated, tabu_time, time_simu):
    plt.plot(tabu_time, tabu, color='b', label='tabu search')
    plt.plot(time_simu, simulated, color='r', label='simulated annealing')
    plt.ylabel('cost of best score')
    plt.xlabel('time')
    plt.title('Simulated Annealing VS tabu search solving TSP :cost of best solution')
    plt.legend()
    plt.show()

def simulatedAnnealingAlgo(graph, numberOfCities, same_first_sol, graphShow):
    times=[]
    start = time.time()
    temperature = TEMPERATURE_INITIAL
    listCurrentScore = []
    listBestScore = []
    temperatureList = []
    citiesArr = list(graph.nodes.values())
    currentSolution, scoreCurrent = generate_first_solution(graph, same_first_sol)
    #if(same_first_sol):
     #   currentSolution = permutationConstantFirst(numberOfCities)
    #else:
   #     currentSolution = FirstPermotationOfCities(numberOfCities)
    print("initial solution for simulated annealing ", currentSolution)
    #scoreCurrent = EvaluateSolution(citiesArr, currentSolution)
    print("cost of initial solution is ", scoreCurrent)
    worstScore = scoreCurrent
    bestScore = scoreCurrent
    iterations = 0
    temp = 0
    epochList = []
    # Check if the temperature is still relevant
    while temperature > STOP_TEMPERATURE: # in case we do without epochs value
    #for i in range(epochs):
        newSolution = changePosition(currentSolution)
        newScore = EvaluateSolution(citiesArr, newSolution)
        bestScore, worstScore = updateTemp(bestScore, newScore, worstScore)
        listCurrentScore.append(scoreCurrent)

        # if the new score is smaller that the current score - update it
        if newScore < scoreCurrent:
            currentSolution = newSolution
            scoreCurrent = newScore
        else:
            # if the new score is greater that the current score - make a probability to do this step
            differentScore = newScore - scoreCurrent
            probability = np.exp(-differentScore / temperature)
            if probability > np.random.uniform():
                currentSolution = newSolution
                scoreCurrent = newScore

        # after update temp -> decrease the temp
        temperature = DECREASE_TEMPERATURE * temperature
        listBestScore.append(bestScore)
        times.append(time.time() - start)
        temperatureList.append(temperature)
        iterations = iterations + 1
        #print("iteration ", iterations)
        #print("temperature ", temperature)
        #print("Current score", scoreCurrent)
        #print("best score", bestScore)
        temp = temp + 1
        epochList.append(temp)

    # printing graphs
    if (graphShow):
        graphBestCost(epochList, listBestScore)
        graphCurrentCost(epochList, listCurrentScore)
        graphTemperature(epochList, temperatureList)
    #return listBestScore, times
    return currentSolution, listBestScore, times


#--------------------- Tabu search ------------------------------------
cost_list=[]
def distance(node1,node2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(node1, node2)]))

def cost(sol, graph):
    cost = 0
    size = len(sol) - 1
    for i in range(size):
        node1 = sol[i]
        node2 = sol[i + 1]
        cost += graph.distances[node1][node2]
    return cost

class Graph:

    def __init__(self, path):
        self.nodes = {}
        nb_nodes = 0
        with open(path) as f:
            for line in f:
                line_split = line.split()
                nb_nodes += 1
                node_name = int(line_split[0])
                node_x = int(line_split[1])
                node_y = int(line_split[2])
                self.nodes[node_name] = (node_x, node_y)
                if nb_nodes == 1:
                    self.start_node = node_name
        self.distances = []
        for node1 in self.nodes:
            distances_node1 = []
            for node2 in self.nodes:
                if node1 != node2:
                    d = distance(self.nodes[node1], self.nodes[node2])
                else:
                    d = 0
                distances_node1.append(d)
            self.distances.append(distances_node1)
        self.distances = np.array(self.distances)


def get_neighborhood(best_sol, graph):
    neighboors = []
    for i in range(1, graph.nb_nodes - 1):
        for j in range(2, graph.nb_nodes):
            if j > i:
                candidate = best_sol.copy()
                temp = best_sol[i]
                candidate[i] = best_sol[j]
                candidate[j] = temp
                sol = candidate
                neighboors.append(sol)
    return neighboors

def tabu_search(best_sol, graph, epochs):
    times = []
    times.append(0)
    start_time = time.time()
    s_best = best_sol
    best_candidate = s_best
    tabu_list = list()
    for i in range(epochs - 1):
        candidates_sol = get_neighborhood(s_best, graph)
        for candidate in candidates_sol:
            cost_candidate = cost(candidate, graph)
            cost_best_candidate = cost(best_candidate,graph)
            if candidate not in tabu_list and cost_candidate <= cost_best_candidate:
                best_candidate = candidate
        if cost_best_candidate <= cost(s_best, graph):
            s_best = best_candidate
        tabu_list.append(best_candidate)
        if len(tabu_list) > tabu_list_max_size:
            tabu_list.pop(0)
        #print("iteration ", i)
        #print("cost for cost list", cost(s_best,graph))
        cost_list.append(cost(s_best,graph))
        times.append(time.time() - start_time)
    return (s_best, cost(s_best,graph), times)

def generate_first_solution(graph, basic):
    lst = list(range(1, graph.nb_nodes))
    if not basic:
        random.shuffle(lst)
    lst.insert(0, 0)
    lst.insert(graph.nb_nodes, 0)
    c = cost(lst, graph)
    return (lst,c)

def show_graph_cost(epochs_list, cost_list):
    plt.bar(epochs_list, cost_list, width = 0.5, color = 'purple')
    plt.ylabel('cost of current best path')
    plt.xlabel('iteration')
    plt.title('Tabu search solving  TSP :cost of best solution')
    plt.show()

def show_graph_time_VS_tabu_size(size_list, times):
    plt.plot(size_list, times, width = 0.5, color ='green')
    plt.ylabel('time for ?')
    plt.xlabel('tabu size')
    plt.title('Tabu search solving  TSP :execution time to find best solution depends on tabu list size')
    plt.show()

def main():
    global tabu_list_max_size
    graph_path = sys.argv[1]
    tabu_list_max_size = int(sys.argv[2])
    graph = Graph(graph_path)
    graph.nb_nodes = int(sys.argv[3])
    numberOfEpochs = int(sys.argv[4])

    same_first_sol = False
    graphShow = False
    a = len(sys.argv)
    if a > 5:
        if int(sys.argv[5] == "-s"):
            same_first_sol = True
        if int(sys.argv[6] == "-g"):
            graphShow = True

    # simulated Annealing Algorithm

    simulatedBest, listBestscores, times_simu = simulatedAnnealingAlgo(graph, graph.nb_nodes, same_first_sol, graphShow)
    #print(f"Best solution of Simulated annealing: {simulatedBest}, \n with total cost: {cost(simulatedBest, graph)}.")
    print(f"Best solution of Simulated annealing: {simulatedBest}, \n with total cost: ?.")

    # Tabu Search Algorithm
    first_solution, distance_of_first_solution = generate_first_solution(graph, same_first_sol)
    cost_list.append(distance_of_first_solution)

    print("initial solution for Tabu search ", first_solution)
    print("with cost ", distance_of_first_solution)
    best_sol, best_cost, time_line_tabu = tabu_search(first_solution, graph, numberOfEpochs)
    print(f"Best solution of Tabu search: {best_sol}, \n with total cost: {best_cost}.")
    epochs_list = list(range(1, numberOfEpochs + 1))
    if graphShow:
        show_graph_cost(epochs_list, cost_list)
        compare2Algorithms(cost_list, listBestscores, time_line_tabu, times_simu)

if __name__ == "__main__":
    main()