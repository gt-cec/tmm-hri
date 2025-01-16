# contains methods used to calculate graph similarity metrics between belief states

import math
import itertools

def smcc(state_1:list, state_2:list) -> float:
    """
    Calculate the SMCC between two states.
    :param state_1: The first state, of the form [[(x,y), (x,y), ...], [(x,y), (x,y), ...], ...]
    :param state_2: The second state, of the form [[(x,y), (x,y), ...], [(x,y), (x,y), ...], ...]
    :return: The SMCC between the two states.
    """
    if len(state_1) != len(state_2):
        raise ValueError("The two states must have the same number of classes and nodes.")
    if len(state_1) == 0:
        return 0
    
    summed_cost = 0
    for i in range(len(state_1)):
        summed_cost += min_cost_between_sets(state_1[i], state_2[i])
    return summed_cost


def m_smcc(state_1:list, state_2:list) -> float:
    """
    Calculate the mean SMCC between two states, normalized by the number of objects in each set.
    This weights the importance of each class by the number of objects in the class.
    :param state_1: The first state, of the form [[(x,y), (x,y), ...], [(x,y), (x,y), ...], ...]
    :param state_2: The second state, of the form [[(x,y), (x,y), ...], [(x,y), (x,y), ...], ...]
    :return: The SMCC between the two states.
    """
    if len(state_1) != len(state_2):
        raise ValueError("The two states must have the same number of nodes.")
    if len(state_1) == 0:
        return 0
    
    summed_cost = 0
    for i in range(len(state_1)):
        summed_cost += min_cost_between_sets(state_1[i], state_2[i]) / len(state_1[i])
    return summed_cost


def min_cost_between_sets(set_1:list, set_2:list) -> float:
    """
    Calculates the minimum distance cost between two sets. Uses a very inefficient method
    that could be optimized.
    :param set_1: The first set, of the form [(x,y), (x,y), ...]
    :param set_2: The second set, of the form [(x,y), (x,y), ...]
    :return: The minimum distance cost between the two sets
    """

    if len(set_1) != len(set_2):
        raise ValueError("The two sets must have the same number of nodes.")

    combinations_1 = list(itertools.combinations(set_1, len(set_1)))  # all possible combinations of the first set
    combinations_2 = list(itertools.combinations(set_2, len(set_2)))  # all possible combinations of the second set
    min_cost = float('inf')
    for combination_1 in combinations_1:
        for combination_2 in combinations_2:
            cost = 0
            for i in range(len(combination_1)):
                cost += math.sqrt((combination_1[i][0] - combination_2[i][0]) ** 2 + (combination_1[i][1] - combination_2[i][1]) ** 2)
            min_cost = min(min_cost, cost)
        
    return min_cost