#!/usr/bin/env python3
import collections
import csv
import sys


INF = 10 ** 18

class Edge():
    def __init__(self, fr, to, capacity, cost):
        self.fr = fr
        self.to = to
        self.capacity = capacity
        self.cost = cost

    def __str__(self):
        return "fr: {}, to: {}, cap: {}, cost: {}".format(self.fr, self.to,
                self.capacity, self.cost)


class MCF():
    def __init__(self, n, edges):
        self.edges = edges
        self.n = n

    def shortest_paths(self, v0):
        d = [INF] * self.n
        p = [-1] * self.n
        d[v0] = 0
        m = [2] * self.n
        q = collections.deque()
        q.append(v0)

        while len(q) > 0:
            u = q.popleft()
            m[u] = 0;
            for v in self.adj[u]:
                if self.capacity[u][v] > 0 and \
                        d[v] > d[u] + self.cost[u][v]:
                    d[v] = d[u] + self.cost[u][v]
                    p[v] = u
                    if m[v] == 2:
                        m[v] = 1
                        q.append(v)
                    elif m[v] == 0:
                        m[v] = 1
                        q.appendleft(v)

        return d, p

    def min_cost_flow(self, desired_flow, s, t):
        self.adj = [[]] * self.n
        self.cost = [[0] * self.n for i in range(self.n)]
        self.capacity = [[0] * self.n for i in range(self.n)]

        for e in self.edges:
            self.adj[e.fr].append(e.to)
            self.adj[e.to].append(e.fr)
            self.cost[e.fr][e.to] = e.cost
            self.cost[e.to][e.fr] = -e.cost
            self.capacity[e.fr][e.to] = e.capacity

        flow, cost = 0, 0
        d, p = [], []
        while flow < desired_flow:
            d, p = self.shortest_paths(s)
            if d[t] == INF:
                break

            f = desired_flow - flow
            cur = t
            while cur != s:
                f = min(f, self.capacity[p[cur]][cur])
                cur = p[cur]

            flow += f
            cost += f * d[t]
            cur = t
            while cur != s:
                self.capacity[p[cur]][cur] -= f
                self.capacity[cur][p[cur]] += f
                cur = p[cur]

            print(" :: ", flow)

        return flow, cost;


def edges_from_csv(file_name):
    """reads a csv file containing columns Q, V, Probability corresponding to
    the Q-U edges that can be picked. Ignores column names and assumes the Q
    end comes first, then U, and then the probability or weight of the edge."""
    
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)

        # skip the header
        reader.__next__()

        lines = [line for line in reader]
        edge_triples = [(int(line[0]), int(line[1]), float(line[2])) 
                for line in lines]
        
        return edge_triples


class BipartiteMapper():
    def __init__(self, start_from):
        self.qs = {}
        self.us = {}
        self.invUs = {}
        self.invQs = {}
        self.next_node = start_from

    def _map(self, mp, inv_mp, key):
        val = mp.get(key, None)
        if val is None:
            val = self.next_node
            mp[key] = val
            inv_mp[val] = key
            self.next_node += 1

        return val

    def mapQ(self, q):
        return self._map(self.qs, self.invQs, q)

    def mapU(self, u):
        return self._map(self.us, self.invUs, u)

    def _inv(self, inv_map, mapped):
        return inv_map[mapped]

    def invQ(self, mappedQ):
        return self._inv(self.invQs, mappedQ)

    def invU(self, mappedU):
        return self._inv(self.invUs, mappedU)


def main():
    if len(sys.argv) != 6:
        print("enter h l input-file.csv output-file.csv info-output.txt\
as cmdline args", file=sys.stderr)
        print("example: ./mcmf.py 2 3 data.csv out.csv info.txt",
                file=sys.stderr)
        sys.exit(1)

    h = int(sys.argv[1])
    l = int(sys.argv[2])

    input_file_name = sys.argv[3]
    out_csv_file_name = sys.argv[4]
    out_info_file_name = sys.argv[5]
    
    edges = edges_from_csv(input_file_name)

    print("Solving for: h={}, l={}, numEdges={}".format(h, l, len(edges)))

    # map node labels to integers from 1 to n
    # MCMF implementation requires us to use 0 to n for node lables, so it can
    # conveniently use matrices for weights, flows, etc.
    # We must get the inverse map of the labels when we're outputting the
    # answer.
    # When debugging, keep in mind that from this point on, the node labels are
    # the mapped ones, not the actual ones in the data.csv. One way around this
    # would be to map the node labels yourself in prior and comment the next
    # few lines of code, and also get rid of the inverse mapping at the end
    mapper = BipartiteMapper(1)
    edges_with_mapped_labels = [(mapper.mapQ(e[0]), mapper.mapU(e[1]), e[2])
            for e in edges]

    # Overwrite the previous list. We won't need it
    edges = edges_with_mapped_labels
    
    edges_with_capacities = [(e[0], e[1], 1, e[2]) for e in edges]
    edges = edges_with_capacities


    # How many digits after the decimal to consider
    weight_precision = 8
    edges_with_negative_int_weights = [
        (e[0], e[1], e[2], -int(e[3] * 10**weight_precision)) for e in edges
        ]
    edges = edges_with_negative_int_weights

    unique_qs = set([e[0] for e in edges])
    all_qs = list(unique_qs)
    unique_us = set([e[1] for e in edges])
    all_us = list(unique_us)

    n = len(all_qs) + len(all_us) + 1 + 1 ## Source and sink
    
    ## Assign the first and last labels to 
    src_label = 0
    snk_label = n - 1

    desired_flow = h * len(all_qs)

    num_edges_in_the_middle = len(edges)

    for q in all_qs:
        edges.append((src_label, q, h, 0)) # src -> q (cap: h, cost: 0)

    for u in all_us:
        edges.append((u, snk_label, l, 0)) # src -> q (cap: h, cost: 0)

    edge_list = [Edge(e[0], e[1], e[2], e[3]) for e in edges]

    # Anything prior to this point is just input sanitization
    # The following few lines are the only ones that you need if you're 
    # using this as a library
    mcf = MCF(n, edge_list)
    flow, min_cost = mcf.min_cost_flow(desired_flow, src_label, snk_label)

    assigned_u_grouped_by_q = {}

    for i in range(num_edges_in_the_middle):
        e = mcf.edges[i]
        q = e.fr
        u = e.to

        if mcf.capacity[q][u] == 0:
            if assigned_u_grouped_by_q.get(q, None) is None:
                assigned_u_grouped_by_q[q] = []
            assigned_u_grouped_by_q[q].append(u)
    
    with open(out_csv_file_name, "w") as f:
        f.write("Q,P\n")
        for q in assigned_u_grouped_by_q.keys():
            assigned_u_grouped_by_q[q] = sorted(assigned_u_grouped_by_q[q])
            for u in assigned_u_grouped_by_q[q]:
                f.write("{},{}\n".format(mapper.invQ(q), mapper.invU(u)))
    
    with open(out_info_file_name, "w") as f:
        total_benefit = -min_cost/(0.0 + 10 ** weight_precision);
        f.write("Value of left cut: |Q|.h = {}\n".format(len(all_qs) * h))
        f.write("Value of right cut: |U|.l = {}\n".format(len(all_us) * l))
        f.write("Total flow sent: {}\n".format(flow))
        f.write("Sum of the weights of the chosen edges: {}\n".
                format(total_benefit))
        if flow > 0:
            f.write("Average benefit per flow: {}\n".
                    format(total_benefit/flow))



if __name__ == "__main__":
    main()
