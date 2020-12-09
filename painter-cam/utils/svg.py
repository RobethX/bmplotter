from penkit_optimize.greedy import greedy_walk
from penkit_optimize.path_graph import PathGraph
from penkit_optimize.route_util import (
    get_route_from_solution,
    join_close_paths,
    cost_of_route,
)

DEFAULT_MERGE_THRESHOLD = 2.0

def optimize(paths, greedy=True, merge_paths=False, max_runtime=300, origin_x=0.0, origin_y=0.0):
    initial_cost = cost_of_route(paths)
    print("Initial cost: {}".format(initial_cost)) # DEBUG: for testing

    path_graph = PathGraph(paths, origin=origin_x + (origin_y * 1j))
    greedy_solution = list(greedy_walk(path_graph))
    greedy_route = get_route_from_solution(greedy_solution, path_graph)

    greedy_cost = cost_of_route(greedy_route)
    print("Cost after greedy optimization: {}".format(greedy_cost))

    assert greedy_cost < initial_cost # make sure optimized path costs less

    if greedy:
        route = greedy_route
    else:
        route = paths # DEBUG: add proper vehicle routing/travelling salesman problem optimization

    if merge_paths is not False:
        if merge_paths is None:
            threshold = DEFAULT_MERGE_THRESHOLD
        else:
            threshold = merge_paths
        print("Routes before merging: {}".format(len(route)))
        route = join_close_paths(route, threshold)
        print("Routes after merging: {}".format(len(route)))

    return route