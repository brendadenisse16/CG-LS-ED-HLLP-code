from common.utils.data_utils import get_hubs, get_path_cost_2
import numpy as np

class Label:
    def __init__(self, id, node, time, cost, path=None, resource=0, label_type=1, visited=None):
        self.id = id
        self.node = node      # Current node
        self.time = time      # Total time
        self.cost = cost      # Total cost
        self.path = path or [node]  # List of nodes visited
        self.origin = self.path[0]
        self.type = label_type  # Type of label (e.g., 'pickup', 'delivery', 'return')
        self.resource = resource # for now, this will hold the h used (hub arcs)
        self.visited = visited # this will be an array of bits
    
    
    def add_node(self, node, edge_time, edge_cost):
        """ 
        Add a node to the label's path.
        """
        assert self.visited[node] == 0, f"can't add node {node} to path {self.path}"
        self.node = node
        self.time += edge_time
        self.cost += edge_cost
        self.path.append(node)
        if self.visited != None:
            self.visited[node] = 1
    
    def dominates(self, otherLabel) -> bool:
        """returns true if this label dominates other

        Args:
            otherLabel (Label): the other label

        Returns:
            bool: true if this label dominates the other, false otherwise
        """
        

        if self.node == otherLabel.node:
            if self.time <= otherLabel.time:
                if self.cost >= otherLabel.cost: # aqui consideramos a cost como ganancia, por eso domina el que tenga mas
                    #if self.resource < otherLabel.resource - 1:  
                    if self.resource <= otherLabel.resource:
                        leq = self.visited <= otherLabel.visited
                        if np.all(leq):                    
                            return True
        return False
    

    def __lt__(self, other):
        """
        Comparison method for labels based on cost.
        """
        return self.cost < other.cost

    def __str__(self):
        """
        String representation of the label.
        """
        return f"Node: {self.node}, Type: {self.type}, Time: {self.time}, Cost: {self.cost}, Path: {self.path}, Visited: {self.visited}"
    

    
    def to_result(self, general_info):    
        return [
            self.path,
            get_hubs(self.path, self.type),
            self.type,
            self.path[0],
            self.path[-1],
            self.time,
            get_path_cost_2(general_info, self.path[0], self.node, self.time), # profit real
            self.cost # profit reducido, acumulado en el labeling
        ]
    
    def __repr__(self):
        return f"<<id:{self.id}, node:{self.node}, time:{self.time}, cost: {self.cost}, path: {self.path}, type: {self.type}, resource: {self.resource}>>"