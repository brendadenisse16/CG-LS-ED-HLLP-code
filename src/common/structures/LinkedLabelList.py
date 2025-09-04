from common.structures.label import Label

class Node:
  def __init__(self, id:int, label:Label=None):
    self.id = id
    self.label = label
    self.next = None

  def __repr__(self):
    return self.label


class LinkedList:
  def __init__(self):
    self.head = None
    self.tail = None
  
  def add_first(self, node):
    node.next = self.head
    self.head = node
    if node.next is None:
      self.tail = node
  
  def add_last(self, node):
    if self.head == None:
      self.head = node
    if self.tail != None:
      self.tail.next = node
    self.tail = node
    

  # def add_after(self, target_node_id, new_node):
  #   if self.head is None:
  #     raise Exception("List is empty")

  #   for node in self:
  #     if node.id == target_node_id:
  #       new_node.next = node.next
  #       node.next = new_node
  #       return
      
  #   raise Exception("Node with data '%s' not found" % target_node_id)
  
  
  # def add_before_domination(self, label):
  #   node = Node(label.id, label)
  #   found = False
  #   pointer = self.head
  #   for node in self:
  #     l = node.label
  #     if l.dominates(label):
  #       node.next = l
  #       pointer.next = node
  #       found=True
  #       break
  #     pointer = l
  #   if not found:
  #     self.add_last(node)
  
  
  def add_by_cost_order_desc(self, new_label:Label):
    """this method will maintain the order in the labels.
    The labels will be stored in descendant order by cost (profit) 

    Args:
        label (Label): _description_
    """
    node_to_be_added = Node(new_label.id, new_label)
    
    if self.head is None or new_label.cost > self.head.label.cost :
      return self.add_first(node_to_be_added)
    
    prev = self.head
        
    while prev.next is not None:
      node = prev.next
      l = node.label
      if new_label.cost > l.cost:
        prev.next = node_to_be_added
        node_to_be_added.next = node
        return
      
      prev = node
    
    # if the new label's cost is lower than the costs of the labels already in the 
    # list OR
    # the list is empty, then
    # this node get's added last
    return self.add_last(node_to_be_added)
    
  def add_by_cost_order_asc(self, new_label:Label):
    """this method will maintain the order in the labels.
    The labels will be stored in descendant order by cost (profit) 

    Args:
        label (Label): _description_
    """
    node_to_be_added = Node(new_label.id, new_label)
    
    if self.head is None or new_label.cost < self.head.label.cost :
      return self.add_first(node_to_be_added)
    
    prev = self.head
        
    while prev.next is not None:
      node = prev.next
      l = node.label
      if new_label.cost < l.cost:
        prev.next = node_to_be_added
        node_to_be_added.next = node
        return
      
      prev = node
    
    return self.add_last(node_to_be_added)
    
  
  def evaluate_domination(self, label):
    for node in self:
      l = node.label
      if l.dominates(label):
        return True
    
    return False
  
  
  def __repr__(self):
    node = self.head
    nodes = []
    while node is not None:
      nodes.append(str(node.id))
      node = node.next
    nodes.append("None")
    return " -> ".join(nodes)
  
  
  def __iter__(self):
    node = self.head
    while node is not None:
      yield node
      node = node.next