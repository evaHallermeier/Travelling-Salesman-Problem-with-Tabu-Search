# Optimization project - Tabu search

This project is a python code which implement the Tabu search algorithm and solve the travelling salesman problem TSP.
Travelling Salesman Problem (TSP): Given a set of cities and their position, the problem is to find the shortest possible route that visits every city exactly once and returns to the starting point.
We also implemented the algorithm of simulated annealing in order to compare the performance with the Tabu search algorithm.

### Prerequisites

The code uses few modules that you need to download: numpy and matplotlib for graphs.

## Running

1. Download the project and open it in an IDE like Pycharm.
2. Create a graph by running the code generateGraph.py with one argument: the number of cities for the problem: it will create a file with cities and their position called graph{n}.txt : each line in the file will represent a city with their attributes: 
  name x_position y_position
  exemple 0 5 9 
  0 will be hte name of the city, 5 will be x_position and 9 is y_position of this city
  
3. Run the code tabu_search_project.py with arguments separated by space
  
  - first argument is name of the file created as exemple graph30.txt (need to be in the same directory as the code)
  
  - second argument is the size of the tabu list (it s a hyper parameter)
  
  - third argument is the number of cities in the given graph
  
  - fourth argument is the number of epochs for tabu search algorithm
  
  - fifth argument is the number of epochs for simulated annealing algorithm
  
  - sixth argument is -s  (if we want to run the 2 algorithms with the same initial solution) if not write 0
  
  - seventh argument is -g (if we want to show graph and show the performance of the 2 algorithms)
  
  
  Example:   graph30.txt 20 30 50 1000 -s -g
  for running tabu_search_project.py

## Authors
- Eva Hallermeier
- Hadar Reuven
