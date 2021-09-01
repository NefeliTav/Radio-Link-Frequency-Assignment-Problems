import random
import time
from operator import neg
from sortedcontainers import SortedSet
from functools import wraps
from time import time

debug_time = True  # time


def timer(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        if not debug_time:
            return func(*args, **kwargs)
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            elapsed = end_ if end_ > 0 else 0
            print('Function "{name}" took {time} s.'.format(
                name=func.__name__, time=elapsed/1000))
    return _time_it


variables = []
dom_id = []
domains = {}
temp = {}
neighbors = {}
constr = {}

# read files

with open("../data/var2-f24.txt") as f:
    next(f)
    for line in f:
        var, d_id = line.split()
        variables.append(int(var))
        dom_id.append(int(d_id))


with open("../data/dom2-f24.txt") as f:
    next(f)
    for line in f:
        data = line.split()
        str_id = int(data.pop(0))
        data.pop(0)
        values = [int(i) for i in data]
        temp.update({str_id: values})

count = 0
# for every var
for id in dom_id:
    # key = 0 or 1
    # value = values for each category
    domains.update({variables[count]: temp[id]})
    count += 1

with open("../data/ctr2-f24.txt") as f:
    next(f)
    for line in f:
        data = line.split()
        data[0] = int(data[0])
        data[1] = int(data[1])
        data[3] = int(data[3])
        if data[0] in neighbors.keys():  # already exists
            neighbors[data[0]].append(data[1])
        else:
            neighbors[data[0]] = [data[1]]
        if data[1] in neighbors.keys():
            neighbors[data[1]].append(data[0])
        else:
            neighbors[data[1]] = [data[0]]

        data.append(0)  # constraint weight = 0
        if data[0] in constr.keys():
            # constr is a dict that contains all the constraints and their weights
            constr[data[0]].append(data)
        else:
            constr[data[0]] = [data]
        if data[1] in constr.keys():
            constr[data[1]].append(data)
        else:
            constr[data[1]] = [data]


def constraints(A, a, B, b):
    csp.nconstraints += 1
    for v in constr[A]:  # i use the dict so i dont have to open the file again
        if (v[0] == B or v[1] == B):
            val = abs(a-b)
            if v[2] == ">" and val > v[3]:
                return True
            if v[2] == "=" and val == v[3]:
                return True
            return False
    return True  # not even neighbors


def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))


class CSP():
    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0
        self.conf_set = {}
        self.nconstraints = 0
        self.visited = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently

        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    # These methods are for the tree and graph-search interface:
    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {
                v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC3

def revise2(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise2(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering
def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def identity(x): return x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


def seekSupport(C, x, a):
    for val2 in csp.curr_domains[C]:  # every value of neighbor
        if csp.constraints(x, a, C, val2) == True:
            return True
    return False


def revise(csp, C, x):
    dom = csp.curr_domains[x]  # local domain
    all_are_false = True
    for a in dom[:]:
        if seekSupport(C, x, a) == True:  # there wont be a domain wipeout
            all_are_false = False  # so i can stop now
            break

    if all_are_false:
        # print("DOMAIN WIPEOUT",x,C)
        for v in constr[x]:
            if (v[0] == C or v[1] == C):  # update dictionary with constraint weights
                v[4] += 1
                return v[4]  # return in order to update variable weight
    return 0  # no wipeout

# Variable ordering


def dom_wdeg(assignment, csp):
    csp.support_pruning()
    result = -1
    minimum = float('inf')
    dom = 0
    for var in csp.variables:
        if var not in assignment:
            t = 0
            futvars = 0
            wdeg = 1
            neighbors = []
            for C in csp.neighbors[var]:
                neighbors.append(C)
                if C not in assignment:  # at least one not assigned
                    futvars = 1
            if futvars == 1:
                for C in neighbors:
                    wdeg += revise(csp, C, var)  # sum weights of constraints

            dom = len(csp.curr_domains[var])
            t = dom/wdeg
            if minimum > t:
                minimum = t
                result = var

    #print("result = ",result)
    return result

# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


def fc_cbj(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return B  # B had a domain wipeout because of the instantiation of var
    return -1  # if no domain wipeout occured


@timer
def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
        csp.visited += 1
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    print("Constraint Violations:", len(conflicted))
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


@timer
def backjumping_search(csp, select_unassigned_variable=dom_wdeg,
                       order_domain_values=unordered_domain_values, inference=fc_cbj):
    def backjump(assignment):
        if len(assignment) == len(csp.variables):
            return assignment, -1  # solution
        var = select_unassigned_variable(assignment, csp)
        csp.visited += 1
        csp.conf_set[var] = []  # empty conflict set
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                fail = inference(csp, var, value, assignment, removals)
                if fail == -1:  # go to next var
                    assignment, jump = backjump(assignment)
                    if jump != var:
                        if jump != -1:
                            csp.restore(removals)  # cancel instantiation
                            csp.unassign(var, assignment)
                        return assignment, jump

                csp.restore(removals)  # cancel instantiation

                if fail != -1:
                    for j in csp.neighbors[var]:
                        if j in assignment and j not in csp.conf_set[var]:
                            for val2 in csp.domains[fail]:
                                if not csp.constraints(j, assignment[j], fail, val2):
                                    csp.conf_set[var].append(j)
                                    break
        for j in csp.neighbors[var]:
            if j in assignment and j not in csp.conf_set[var]:
                for val2 in csp.domains[var]:
                    if not csp.constraints(j, assignment[j], var, val2):
                        csp.conf_set[var].append(j)
                        break

        temp = []  # sort conf_set according to which is the most recent in assignment
        for i in assignment:
            if i in csp.conf_set[var]:
                temp.append(i)
        csp.conf_set[var] = []
        csp.conf_set[var] = temp

        if csp.conf_set[var] != []:
            h = csp.conf_set[var][-1]  # most recent var
            for i in csp.conf_set[var]:
                if i not in csp.conf_set[h]:
                    csp.conf_set[h].append(i)  # without duplicates
            if h in csp.conf_set[h]:
                csp.conf_set[h].remove(h)  # remove h from conflict set of h
            csp.unassign(var, assignment)
            return assignment, h

        csp.unassign(var, assignment)
        return None, -1

    result, nothing = backjump({})
    assert result is None or csp.goal_test(result)
    return result


@timer
def backtracking_search(csp, select_unassigned_variable=dom_wdeg,
                        order_domain_values=unordered_domain_values, inference=forward_checking):

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        csp.visited += 1
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


csp = CSP(variables, domains, neighbors, constraints)
print(backtracking_search(csp))
print(backjumping_search(csp))  # uncomment
# print(min_conflicts(csp))
print("Constraint checks:", csp.nconstraints)
print("Visited nodes:", csp.visited)
