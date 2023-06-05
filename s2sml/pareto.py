import numpy as np

def pareto_front(costs, directions):
    """
    Find the Pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param directions: A list of objective directions, "max" for maximization and "min" for minimization
    :return: A (n_points, ) boolean array indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            for j, d in enumerate(directions):
                if "max" in d:
                    is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient