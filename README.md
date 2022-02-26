# Optimization project - Tabu search

This project is a python code which implement the Tabu search algorithm and solve travelling salesman problem TSP.
This problem consist of given n cities with position : the goal is to find the shortest path that over all cities and return to the original cities (the first given)
We also implement the algorithm of simulated annealing in order to compare the performance with the tabu search one.

### Prerequisites

The code use few module that you need to dowload: numpy and matplotlib for graphs.

## Running

1. Download the project and open it in a IDE like Pycharm.
2. Create a graph by running the code genearteGraph.py with arg a number that represent number of cities for the problem: it will create a file with cities and their position called graph<n>.txt : each line will represent a city with their attributes: 
  name x_position y_position
  
3. Run the code tabu_search_project.py with arguments separated by space
  first argument is name of the file created as exemple graph30.txt (need to be in the sme directory as the code)
  
  second argument is the size of the tabu list
  
  third argument is the number of cities in the given graph
  
  fourth argument is is number of epochs for tabu search
  
  fifth argument is number of epochs for simulated annealing
  
  sixth argument is -s  (is=f we want to run the 2 algoruthm with the same initial solution) if not write 0
  
  seventh argument is -g (if we want to show graph and show the performance of the 2 algorithms)
  

## Authors
- Eva Hallermeier
- Hadar Reuven
