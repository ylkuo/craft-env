import copy
import networkx as nx
import nltk
import numpy as np
import random
import signal
import spot

from collections import defaultdict
from graphviz.dot import Digraph
from itertools import combinations
from nltk import CFG
from nltk.parse import BottomUpLeftCornerChartParser

spot.setup()


def parse_simplebool(boolstr, alphabets):
    alphabet_str = ' | '.join(["'"+a+"'" for a in alphabets])
    bool_grammar = """
        S -> '(' S ')' | S OP_2 S | OP_1 S | TERM
        OP_1 -> '!'
        OP_2 -> '&' | '|'
        TERM -> %s
    """ % alphabet_str
    grammar = CFG.fromstring(bool_grammar)
    parser = BottomUpLeftCornerChartParser(grammar)
    trees = [tree for tree in parser.parse(boolstr)]
    tree = trees[0]
    return tree


def get_node_val(tree):
    if len(tree) == 1:
        if type(tree[0]) != nltk.Tree:
            return tree[0]
        else:
            return tree[0][0]
    return None


def gen_symbols(tree):
    symbols = set()
    for i in range(len(tree)):
        node_val = get_node_val(tree[i])
        if node_val == '|' or node_val == '&':
            left_sym = gen_symbols(tree[i-1])
            right_sym = gen_symbols(tree[i+1])
            if node_val == '&':
                symbols = left_sym.union(right_sym)
            else:
                symbols = random.choice([left_sym, right_sym, left_sym.union(right_sym)])
            break
        elif node_val == '!':
            # skip if it is not
            # TODO: but ideally should flip true to false and randomly turn false to true
            break
        elif node_val is not None and node_val not in ['(', ')']:
            symbols.add(node_val)
        elif node_val is None:
            symbols = gen_symbols(tree[i])
    return symbols


def eval_formula(true_symbols, formula, alphabets):
    v = defaultdict(bool)
    for symbol in true_symbols:
        v[symbol] = True
    formula = formula.replace('&', 'and').replace('|', 'or')
    tokens = formula.replace('(', '( ').replace(')', ' )').replace('!', '! ').split()
    # eval the formula
    for i in range(len(tokens)):
        if tokens[i] not in ['(', ')', 'or', 'and', '!', 'True']:
            tokens[i] = "v['%s']" % tokens[i]
        if tokens[i] is '!':
            tokens[i] = 'not'
    formula = ' '.join(tokens)
    formula = formula.replace('( ', '(').replace(' )', ')')
    out = eval(formula)
    return out


def gen_symbols_sample(alphabets, formula):
    symbols = set(); count = 0
    while count == 0 or not eval_formula(symbols, formula, alphabets):
        symbols = set()
        for a in alphabets:
            add = np.random.choice([True, False])
            if add:
                symbols.add(a)
        count += 1
    return symbols


def num_true_assignments(formula, alphabets):
    if formula == '1':
        return 2**len(alphabets)
    elif formula == '0':
        return 1
    n_trues = 0
    for check_len in range(len(alphabets)+1):
        for symbols in combinations(alphabets, check_len):
            if eval_formula(symbols, formula, alphabets):
                n_trues += 1
    return n_trues


class Automaton(object):
    def __init__(self, formula, alphabets=None, add_flexible_state=False, data=None):
        self._graph = nx.DiGraph()
        self._formula = formula
        self._spot_formula = None
        self._spot_automaton = None
        self._alphabets = alphabets
        self._add_flexible_state = add_flexible_state
        if data is None:
            self._to_automaton()
        else:
            self.load(data)

    def get_data(self):
        return self._graph, self._formula, self._alphabets

    def load(self, data):
        self._graph, self._formula, self._alphabets = data

    def _to_automaton(self):
        self._spot_formula = spot.formula(self._formula)
        aut = self._spot_formula.translate('low', 'sbacc')
        self._spot_automaton = aut
        init_states = [d for d in aut.univ_dests(aut.get_init_state_number())]
        bdd_dict = aut.get_dict()
        for s in range(aut.num_states()):
            is_init = s in init_states
            is_accepting = aut.state_is_accepting(s)
            self.add_state(str(s), init=is_init, accept=is_accepting)
        state_id = aut.num_states()
        for ed in aut.edges():
            label = spot.bdd_to_formula(ed.cond, bdd_dict)
            if self._add_flexible_state and str(ed.src) != str(ed.dst):
                state_name = 'e_' + str(state_id)
                self.add_state(state_name)
                # add transition to the new state
                self.add_transition(str(ed.src), state_name, label='1')
                # add transition of the self loop
                self.add_transition(state_name, state_name, label='1')
                # add transition from the new state to destination
                self.add_transition(state_name, str(ed.dst), label=str(label))
                state_id += 1
            else:
                self.add_transition(str(ed.src), str(ed.dst), label=str(label))
        if self._alphabets is None:
            self._alphabets = set()
            for ap in spot.atomic_prop_collect(self._spot_formula):
                self._alphabets.add(str(ap))
        # replace all '1' labels to be all possible alphabets
        for src, dst, label in self._graph.edges(data='label'):
            if self._graph[src][dst]['label'] == '1':
                self._graph[src][dst]['label'] = self._get_alphabet_str()
                self._graph[src][dst]['print_label'] = '1'
            elif self._graph[src][dst]['label'] == '0':
                self._graph[src][dst]['label'] = self._get_neg_alphabet_str()
                self._graph[src][dst]['print_label'] = '0'
            else:
                self._graph[src][dst]['print_label'] = self._graph[src][dst]['label']

    def _get_alphabet_str(self):
        alphabet_str = ' | '.join(self._alphabets)
        alphabet_str = alphabet_str + ' | ' + \
                ' | '.join(['! ' + x for x in self._alphabets])
        return alphabet_str

    def _get_neg_alphabet_str(self):
        alphabet_str = ' & '.join(['! ' + x for x in self._alphabets])
        return alphabet_str

    @property
    def n_states(self):
        nodes = [node for node in self._graph.nodes if 'e_' not in node]
        return len(nodes)

    @property
    def len_min_accepting_run(self):
        init_state = self.get_initial_state()
        accept_states = self.get_accept_states()
        if len(accept_states) == 0:
            return np.inf
        lengths = [nx.shortest_path_length(self._graph, init_state, s) for s in accept_states]
        return np.min(lengths)

    @property
    def len_avg_accepting_run(self):
        init_state = self.get_initial_state()
        accept_states = self.get_accept_states()
        if len(accept_states) == 0:
            return np.inf
        lengths = [nx.shortest_path_length(self._graph, init_state, s) for s in accept_states]
        return np.mean(lengths)

    @property
    def has_accept(self):
        accept = False
        for n in self._graph:
            if self.is_accept(n):
                accept = True
        return accept

    def add_state(self, name, init=False, accept=False):
        if not self._graph.has_node(name):
            self._graph.add_node(name, init=init, accept=accept)
        else:
            self._graph.node[name]['init'] = init
            self._graph.node[name]['accept'] = accept

    def add_transition(self, src, dst, label):
        self._graph.add_edge(src, dst, label=label)

    def get_initial_state(self):
        return list(filter(lambda n: n[1], self._graph.nodes(data='init')))[0][0]

    def get_accept_states(self):
        accept_states = list(filter(lambda n: n[1], self._graph.nodes(data='accept')))
        return [s[0] for s in accept_states]

    def is_accept(self, state):
        return self._graph.node[state]['accept']

    def get_transitions(self, src):
        dsts = list(self._graph.successors(src))
        labels = []
        for dst in dsts:
            labels.append(self._graph[src][dst]['label'])
        return dsts, labels

    def num_accept_str(self, k):
        trans_matrix = np.array(nx.adjacency_matrix(self._graph).todense())
        len_table = np.zeros((k, self.n_states + 1))
        # add a new node connecting to the accept states
        trans_matrix = np.append(trans_matrix, np.zeros((self.n_states, 1)), 1)
        acc_states = [int(s) for s in self.get_accept_states()]
        trans_matrix[acc_states, self.n_states] = 1
        # initialize the weight of transition matrix
        edges = self._graph.edges.data()
        for src, dst, info in edges:
            src = int(src)
            dst = int(dst)
            trans_matrix[src][dst] = num_true_assignments(info['label'], self._alphabets)
        # initialize the first row of the length table
        len_table[0, :] = np.append(trans_matrix[:, self.n_states], 0)
        # count number of paths of length l to target node from each state s
        for l in range(1, k):
            for s in range(self.n_states):
                len_table[l, s] = trans_matrix[s, :].dot(len_table[l-1, :])
        total_acc = 0
        init_states = [int(s) for s in self.get_initial_state()]
        for s in init_states:
            total_acc += len_table[-1, s]
        return total_acc

    def recognize(self, seq):
        init_state = self.get_initial_state()
        branches = [init_state]
        for symbols in seq:
            new_branches = []
            for branch in branches:
                next_states, trans = self.get_transitions(branch)
                for i in range(len(trans)):
                    if eval_formula(symbols, trans[i], self._alphabets):
                        new_branches.append(next_states[i])
            branches = new_branches
        for state in branches:
            if self.is_accept(state):
                return True
        return False

    def is_prefix(self, seq, last_states=None):
        # following the sequence to see if it is in the automaton
        if len(last_states) == 0:
            init_state = self.get_initial_state()
            branches = [init_state]
            last_states = set([init_state])
        else:
            branches = list(last_states)
        new_last_states = set()
        for symbols in seq:
            new_branches = []; prefix = False
            for branch in branches:
                next_states, trans = self.get_transitions(branch)
                for i in range(len(trans)):
                    if next_states[i] not in new_branches and \
                            eval_formula(symbols, trans[i], self._alphabets):
                        new_branches.append(next_states[i])
                        new_last_states.add(next_states[i])
                        prefix = True
            if not prefix:
                return False, np.inf, set()
            branches = new_branches
        # expand the states by bfs to find the shortest distance to an accepting state
        dist_to_accept = 0; n_while = 0
        while True:
            if n_while > 10:   # also, reject the transition if not leading to accept state after many steps
                return False, np.inf, set()
            new_branches = []
            for branch in branches:
                if self.is_accept(branch):
                    return True, dist_to_accept, new_last_states
                next_states, _ = self.get_transitions(branch)
                for state in next_states:
                    if state not in new_branches:
                        new_branches.append(state)
            dist_to_accept += 1
            branches = new_branches
            n_while += 1

    def random_transition(self, src):
        dst = random.choice(list(self._graph.successors(src)))
        label = self._graph[src][dst]['label']
        symbols = gen_symbols_sample(self._alphabets, label)
        return src, dst, symbols

    def gen_sequence(self, state=None, states=None, trans=None):
        if states is None:
            states = []
        if trans is None:
            trans = []
        if state is None:
            state = self.get_initial_state()
            states.append(state)
        if not self.is_accept(state) or (self.is_accept(state) and random.random() < 0.5):
            _, dst, symbol = self.random_transition(state)
            states.append(dst)
            trans.append(symbol)
            states, trans = self.gen_sequence(dst, states, trans)
        return states, trans

    def draw(self, path=None, show=True):
        dot = Digraph()
        dot.graph_attr.update(label=self._formula)
        for node in self._graph.nodes():
            num_peripheries = '2' if self._graph.node[node]['accept'] else '1'
            dot.node(node, node, shape='circle', peripheries=num_peripheries)
        for src, dst, label in self._graph.edges(data='print_label'):
            dot.edge(src, dst, label)
        if path is None:
            dot.render(view=show)
        else:
            dot.render(path, view=show)


if __name__ == '__main__':
    # simple formula label on the edge
    ba = Automaton('FGa & Fb')
    print(ba.len_min_accepting_run)
    print(ba.get_initial_state())
    print(ba.recognize([['a'], ['a'], ['a', 'b']]))  # expect to be True
    print(ba.recognize([['a'], ['a'], ['a']]))  # expect to be False
    print(ba.gen_sequence())
    prefix, dist_to_accept, new_last_states = ba.is_prefix([['a'], ['a']], [])
    print(prefix)  # expect to be True
    ba.draw('tmp_images/ba.svg', show=False)
    # complex formula label on the edge
    ba = Automaton('(a&b) | Gc')
    print(ba.len_min_accepting_run)
    print(ba.get_initial_state())
    print(ba.recognize([['a', 'b'], ['c']]))
    print(ba.recognize([['a', 'c'], ['c']]))
    print(ba.gen_sequence())
    prefix, dist_to_accept, new_last_states = ba.is_prefix([['a']], [])
    print(prefix)  # expect to be False
    ba.draw('tmp_images/ba.svg', show=False)
