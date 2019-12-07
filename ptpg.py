import numpy as np
import networkx as nx


class PTPG:
    # Attribute Initiallization
    def __init__(self):
        self.node_count = int(input("Enter the number of nodes in the graph: "))
        self.edge_count = int(input("Enter the number of edges in the graph: "))
        self.north = self.node_count
        self.east = self.node_count + 1
        self.south = self.node_count + 2
        self.west = self.node_count + 3
        self.matrix = np.zeros((self.node_count, self.node_count), int)

        self.node_position = None
        self.degrees = None
        self.good_vertices = None
        self.contractions = []

        self.t1_matrix = None
        self.t2_matrix = None
        self.t1_longest_distance = [-1] * (self.node_count + 4)
        self.t2_longest_distance = [-1] * (self.node_count + 4)
        self.t1_longest_distance_value = -1
        self.t2_longest_distance_value = -1
        self.n_s_paths = []
        self.w_e_paths = []

        self.room_x = np.zeros(self.node_count)
        self.room_y = np.zeros(self.node_count)
        self.room_height = np.zeros(self.node_count)
        self.room_width = np.zeros(self.node_count)

        print("Enter each edge in new line")
        for i in range(self.edge_count):
            line = input()
            node1 = int(line.split()[0])
            node2 = int(line.split()[1])
            self.matrix[node1][node2] = 1
            self.matrix[node2][node1] = 1
    """
    Adding the NESW vertices to the original graph 
    """

    def add_nesw_vertices(self):
        G = nx.from_numpy_matrix(self.matrix)
        H = G.to_directed()

        # Get all triangles
        all_cycles = list(nx.simple_cycles(H))
        all_triangles = []
        for cycle in all_cycles:
            if len(cycle) == 3:
                all_triangles.append(cycle)

        # Get edges on outer boundary
        outer_boundary = []
        for edge in H.edges:
            count = 0
            for triangle in all_triangles:
                if edge[0] in triangle and edge[1] in triangle:
                    count += 1
            if count == 2:
                outer_boundary.append(edge)

        # Get Vertex-Set of outerboundary
        outer_vertices = []
        for edge in outer_boundary:
            if edge[0] not in outer_vertices:
                outer_vertices.append(edge[0])
            if edge[1] not in outer_vertices:
                outer_vertices.append(edge[1])

        # Get top,left,right and bottom boundaries of graph
        cip = []
        loop_count = 0
        # Finds all corner implying paths in the graph
        while len(outer_vertices) > 1:
            temp = [outer_vertices[0]]
            outer_vertices.pop(0)
            for vertices in temp:
                for vertex in outer_vertices:
                    temp1 = temp.copy()
                    temp1.pop(len(temp) - 1)
                    if (temp[len(temp) - 1], vertex) in outer_boundary:
                        temp.append(vertex)
                        outer_vertices.remove(vertex)
                        if temp1 is not None:
                            for vertex1 in temp1:
                                if (vertex1, vertex) in H.edges:
                                    temp.remove(vertex)
                                    outer_vertices.append(vertex)
            cip.append(temp)
            outer_vertices.insert(0, temp[len(temp) - 1])
            if len(outer_vertices) == 1 and loop_count == 0:
                outer_vertices.append(cip[0][0])
                loop_count += 1

        check = 0
        for vertex in cip[0]:
            if (cip[len(cip) - 1][0], vertex) in H.edges and vertex != cip[len(cip) - 1][1]:
                check = 1
                break

        if check != 1:
            cip[0].insert(0, cip[len(cip) - 1][0])
            cip.pop()
        else:
            for vertex in cip[len(cip) - 2]:
                if (cip[len(cip) - 1][0], vertex) in H.edges and vertex != cip[len(cip) - 1][1]:
                    check = 2
                    break
            if check != 2:
                cip[len(cip) - 2] = cip[len(cip) - 2] + cip[len(cip) - 1][0]
                cip.pop()
        
        print("Number of corner implying paths: ", len(cip))
        print("Corner implying paths: ", cip)

        if len(cip) > 4:
            print("Error! More than 4 corner implying paths")
            exit()

        def create_cip(index):
            cip.insert(index + 1, cip[index])
            cip[index] = cip[index][0:2]
            del cip[index + 1][0:1]

        if len(cip) == 3:
            index = cip.index(max(cip, key=len))
            create_cip(index)

        if len(cip) == 2:
            index = cip.index(max(cip, key=len))
            create_cip(index)
            create_cip(index + 1)

        if len(cip) == 1:
            index = cip.index(max(cip, key=len))
            create_cip(index)
            create_cip(index + 1)
            create_cip(index + 2)


        # Adding north, south, east and west vertices and connects them to boundary vertices
        self.node_count += 4
        new_adjacency_matrix = np.zeros([self.node_count, self.node_count], int)
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                new_adjacency_matrix[i][j] = self.matrix[i][j]

        def news_edges(cip, source_vertex):
            for vertex in cip:
                self.edge_count += 1
                new_adjacency_matrix[source_vertex][vertex] = 1
                new_adjacency_matrix[vertex][source_vertex] = 1

        news_edges(cip[0], self.north)
        news_edges(cip[1], self.east)
        news_edges(cip[2], self.south)
        news_edges(cip[3], self.west)

        new_adjacency_matrix[self.north][self.west] = 1
        new_adjacency_matrix[self.west][self.north] = 1
        new_adjacency_matrix[self.west][self.south] = 1
        new_adjacency_matrix[self.south][self.west] = 1
        new_adjacency_matrix[self.south][self.east] = 1
        new_adjacency_matrix[self.east][self.south] = 1
        new_adjacency_matrix[self.north][self.east] = 1
        new_adjacency_matrix[self.east][self.north] = 1
        self.edge_count += 4

        self.matrix = new_adjacency_matrix

    def initialize_degrees(self):
        """initializes the degrees array from the adjacency matrix of the PTPG"""
        self.degrees = [np.count_nonzero(self.matrix[node]) for node in range(self.node_count)]

    def initialize_good_vertices(self):
        """initializes the good vertices array by checking each node if its good or not"""
        self.good_vertices = []
        for node in range(self.matrix.shape[0]):
            if self.is_good_vertex(node):
                self.good_vertices.append(node)

    def is_good_vertex(self, node):
        """
        checks if the node is good vertex or not
        Definitions:
        1) Light vertex: vertex whose degree <= 19
        2) Heavy vertex: vertex whose degree >= 20
        3) Degree 5 good vertex: (vertex who has degree 5) and (has 0 or 1  heavy neighbours)
        4) Degree 4 good vertex: (vertex who has degree 4) and
                                 ((has 0 or 1 heavy neighbour) or (has 2 heavy neighbours which are not adjacent))
        5) Good vertex: Degree 4 good vertex or Degree 5 good vertex

        Note: We do not want any of the 4 boundary NESW vertices to be a good vertex since we never want to contract
              any edge connected to these vertices. LookUp: Assusmption 1 for detailed reason

        """
        if node not in [self.north, self.east, self.south, self.west]:
            if self.degrees[node] == 5:
                heavy_neighbour_count = 0
                neighbours, = np.where(self.matrix[node] == 1)
                for neighbour in neighbours:  # iterating over neighbours and checking if any of them is heavy vertex
                    if self.degrees[neighbour] >= 20:
                        heavy_neighbour_count += 1
                if heavy_neighbour_count <= 1:
                    return True  # satisfies all conditions for degree 5 good vertex

            elif self.degrees[node] == 4:
                heavy_neighbours = []
                neighbours, = np.where(self.matrix[node] == 1)
                for neighbour in neighbours:  # iterating over neighbours and checking if any of them is heavy vertex
                    if self.degrees[neighbour] >= 20:
                        heavy_neighbours.append(neighbour)
                if (len(heavy_neighbours) <= 1) or (
                        len(heavy_neighbours) == 2 and self.matrix[heavy_neighbours[0]][heavy_neighbours[1]] != 1):
                    return True  # satisfies all conditions for degree 4 good ertex
        return False

    def get_contractible_neighbour(self, v):
        v_nbr, = np.where(self.matrix[v] == 1)
        # checking if any of neighbors of the good vertex v is contractible
        # by lemma we will find one but it can be one of nesw so we need to ignore this v
        for u in v_nbr:
            if u in [self.north, self.east, self.south, self.west]:
                continue
            contractible = True
            u_nbr, = np.where(self.matrix[u] == 1)
            y_and_z = np.intersect1d(v_nbr, u_nbr, assume_unique=True)
            if len(y_and_z) != 2:
                print("Input graph might contain a complex triangle")
            for x in v_nbr:
                if x in y_and_z or x == u:
                    continue
                x_nbr, = np.where(self.matrix[x] == 1)
                intersection = np.intersect1d(x_nbr, u_nbr, assume_unique=True)
                for node in intersection:
                    if node not in y_and_z and node != v:
                        contractible = False
                        break
                if not contractible:
                    break
            if contractible:
                return u, y_and_z
        return -1, []

    def update_adjacency_matrix(self, v, u):
        self.node_position[u][0] = (self.node_position[u][0] + self.node_position[v][0]) / 2
        self.node_position[u][1] = (self.node_position[u][1] + self.node_position[v][1]) / 2
        v_nbr, = np.where(self.matrix[v] == 1)
        for node in v_nbr:
            self.matrix[v][node] = 0
            self.matrix[node][v] = 0
            if node != u:
                self.matrix[node][u] = 1
                self.matrix[u][node] = 1

    def update_good_vertices(self, v, u, y_and_z):
        self.degrees[u] += self.degrees[v] - 4
        self.degrees[y_and_z[0]] -= 1
        self.degrees[y_and_z[1]] -= 1
        self.degrees[v] = 0

        def check(node):
            if self.is_good_vertex(node) and (node not in self.good_vertices):
                self.good_vertices.append(node)
            elif (not self.is_good_vertex(node)) and (node in self.good_vertices):
                self.good_vertices.remove(node)
        check(u)
        check(y_and_z[0])
        check(y_and_z[1])

    def contract(self):
        attempts = len(self.good_vertices)
        while attempts > 0:
            v = self.good_vertices.pop(0)
            u, y_and_z = self.get_contractible_neighbour(v)
            if u == -1:
                self.good_vertices.append(v)
                attempts -= 1
                continue
            self.contractions.append({'v': v, 'u': u, 'y_and_z': y_and_z, 'v_nbr': np.where(self.matrix[v] == 1)[0]})
            self.update_adjacency_matrix(v, u)
            self.update_good_vertices(v, u, y_and_z)
            self.node_count -= 1
            self.edge_count -= 3
            return v, u
        return -1, -1

    def get_trivial_rel(self):
        for node in range(self.matrix.shape[0]):
            if self.matrix[self.north][node] == 1 and node not in [self.east, self.west]:
                self.matrix[node][self.north] = 2
                self.matrix[self.north][node] = 0

                self.matrix[self.south][node] = 2
                self.matrix[node][self.south] = 0

                self.matrix[node][self.east] = 3
                self.matrix[self.east][node] = 0

                self.matrix[self.west][node] = 3
                self.matrix[node][self.west] = 0

    def expand(self):
        contraction = self.contractions.pop()
        case = self.get_case(contraction)
        o = contraction['u']
        v = contraction['v']
        case(o, v, contraction['y_and_z'][0], contraction['y_and_z'][1], contraction['v_nbr'])
        self.node_position[o][0] = 2 * self.node_position[o][0] - self.node_position[v][0]
        self.node_position[o][1] = 2 * self.node_position[o][1] - self.node_position[v][1]

    def get_case(self, contraction):
        o = contraction['u']
        y_and_z = contraction['y_and_z']
        y = y_and_z[0]
        z = y_and_z[1]
        if self.matrix[o][y] == 2:
            if self.matrix[o][z] == 3:
                print("o->y : T1, o->z : T2, caseA")
                return self.caseA
            elif self.matrix[o][z] == 2:
                if self.get_ordered_neighbour_label(o, y, clockwise=False) == 3:
                	y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("o->y : T1, o->z : T1, caseB")
                return self.caseB
            elif self.matrix[z][o] == 3:
                print("o->y : T1, z->o : T2, caseD")
                return self.caseD
            elif self.matrix[z][o] == 2:
                print("o->y : T1, z->o : T1, caseF")
                return self.caseF
            else:
                print("ERROR")

        if self.matrix[y][o] == 2:
            if self.matrix[o][z] == 3:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T1, o->z : T2, caseE")
                return self.caseE
            elif self.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T1, o->z : T1, caseF")
                return self.caseF
            elif self.matrix[z][o] == 3:
                print("y->o : T1, z->0 : T2, caseH")
                return self.caseH
            elif self.matrix[z][o] == 2:
                if self.get_ordered_neighbour_label(o, y, clockwise=False) == 3:
                	y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T1, z->o : T1, caseI")
                return self.caseI
            else:
                print("ERROR")
                
        if self.matrix[o][y] == 3:
            if self.matrix[o][z] == 3:
                if self.get_ordered_neighbour_label(o, y, clockwise=False) == 2:
                	y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("o->y : T2, o->z : T2, caseC")
                return self.caseC
            elif self.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("o->y : T2,  o->z : T1, caseA swapped")
                return self.caseA
            elif self.matrix[z][o] == 3:
                print("o->y : T2, z->o : T2, caseG")
                return self.caseG
            elif self.matrix[z][o] == 2:
                print("o->y : T2, z->o : T1, caseE")
                return self.caseE
            else:
                print("ERROR")

        if self.matrix[y][o] == 3:
            if self.matrix[o][z] == 3:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T2, o->z : T2, caseG")
                return self.caseG
            elif self.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T2,  o->z : T1, caseD")
                return self.caseD
            elif self.matrix[z][o] == 3:
                if self.get_ordered_neighbour_label(o, y, clockwise=False) == 2:
                	y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T2,  z->o : T2, caseJ")
                return self.caseJ
            elif self.matrix[z][o] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                print("y->o : T2,  z->o : T1, caseH")
                return self.caseH
            else:
                print("ERROR")

    def handle_original_u_nbrs(self, o, v, y, z, v_nbr):
        for alpha in v_nbr:
            if alpha != y and alpha != z and alpha != o:
                if self.matrix[o][alpha] != 0:
                    self.matrix[v][alpha] = self.matrix[o][alpha]
                    self.matrix[o][alpha] = 0
                if self.matrix[alpha][o] != 0:
                    self.matrix[alpha][v] = self.matrix[alpha][o]
                    self.matrix[alpha][o] = 0

    def caseA(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour_label(o, y, clockwise=True) == 2:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[y][v] = 3
                self.matrix[v][z] = 3
                self.matrix[o][v] = 2
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 2
                self.matrix[v][z] = 3
                self.matrix[v][o] = 2
                self.matrix[o][y] = 0
                self.matrix[y][o] = 3
                self.matrix[z][o] = 0 # this line should be deleted
                self.matrix[o][z] = 3 # this line should be deleted
        else:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 2
                self.matrix[z][v] = 2
                self.matrix[o][v] = 3
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[o][z] = 0
                self.matrix[z][o] = 2
                self.matrix[v][o] = 3
                self.matrix[v][y] = 2
                self.matrix[v][z] = 3

    def caseB(self, o, v, y, z, v_nbr):
        self.handle_original_u_nbrs(o, v, y, z, v_nbr)
        self.matrix[v][y] = 3
        self.matrix[z][v] = 3
        self.matrix[o][v] = 2

    def caseC(self, o, v, y, z, v_nbr):
        self.handle_original_u_nbrs(o, v, y, z, v_nbr)
        self.matrix[y][v] = 2
        self.matrix[v][z] = 2
        self.matrix[o][v] = 3

    def caseD(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour_label(o, y, clockwise=False) == 2:
            if self.get_ordered_neighbour(o, y, False) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 3
                self.matrix[z][v] = 3
                self.matrix[o][v] = 2
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[o][y] = 0 # this line should be deleted
                self.matrix[o][y] = 3
                self.matrix[v][y] = 2
                self.matrix[z][v] = 3
                self.matrix[v][o] = 2
        else:
            if self.get_ordered_neighbour(o, y, False) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 2
                self.matrix[z][v] = 2
                self.matrix[v][o] = 3
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[z][o] = 0 # this line should be deleted
                self.matrix[z][o] = 2
                self.matrix[z][v] = 3
                self.matrix[v][y] = 2
                self.matrix[o][v] = 3

    def caseE(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour_label(o, y, clockwise=True) == 2:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 3
                self.matrix[z][v] = 3
                self.matrix[v][o] = 2
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[z][o] = 0 # this line should be deleted
                self.matrix[z][o] = 3
                self.matrix[z][v] = 2
                self.matrix[v][y] = 3
                self.matrix[o][v] = 2

        else:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 2
                self.matrix[z][v] = 2
                self.matrix[o][v] = 3
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[o][y] = 0 # this line should be deleted
                self.matrix[o][y] = 2
                self.matrix[v][o] = 3
                self.matrix[v][y] = 3
                self.matrix[z][v] = 2

    def caseF(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour(o, y, True) in v_nbr:
            self.handle_original_u_nbrs(o, v, y, z, v_nbr)
            self.matrix[v][y] = 2
            self.matrix[z][v] = 2
            self.matrix[o][v] = 3
        else:
            self.handle_original_u_nbrs(o, v, y, z, v_nbr)
            self.matrix[v][y] = 2
            self.matrix[z][v] = 2
            self.matrix[v][o] = 3

    def caseG(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour(o, y, True) in v_nbr:
            self.handle_original_u_nbrs(o, v, y, z, v_nbr)
            self.matrix[v][y] = 3
            self.matrix[z][v] = 3
            self.matrix[v][o] = 2
        else:
            self.handle_original_u_nbrs(o, v, y, z, v_nbr)
            self.matrix[v][y] = 3
            self.matrix[z][v] = 3
            self.matrix[o][v] = 2

    def caseH(self, o, v, y, z, v_nbr):
        if self.get_ordered_neighbour_label(o, y, clockwise=True) == 2:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[v][y] = 3
                self.matrix[z][v] = 3
                self.matrix[v][o] = 2
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[y][o] = 0
                self.matrix[o][y] = 3
                self.matrix[y][v] = 2
                self.matrix[z][v] = 3
                self.matrix[o][v] = 2
        else:
            if self.get_ordered_neighbour(o, y, True) in v_nbr:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[y][v] = 2
                self.matrix[v][z] = 2
                self.matrix[v][o] = 3
            else:
                self.handle_original_u_nbrs(o, v, y, z, v_nbr)
                self.matrix[z][o] = 0
                self.matrix[o][z] = 2
                self.matrix[y][v] = 2
                self.matrix[z][v] = 3
                self.matrix[o][v] = 3 

    def caseI(self, o, v, y, z, v_nbr):
        self.handle_original_u_nbrs(o, v, y, z, v_nbr)
        self.matrix[y][v] = 3
        self.matrix[v][z] = 3
        self.matrix[v][o] = 2

    def caseJ(self, o, v, y, z, v_nbr):
        self.handle_original_u_nbrs(o, v, y, z, v_nbr)
        self.matrix[v][y] = 2
        self.matrix[z][v] = 2
        self.matrix[v][o] = 3

    def get_ordered_neighbour_label(self, centre, y, clockwise=False):
        next = self.get_ordered_neighbour(centre, y, clockwise)
        if self.matrix[centre][next] == 2 or self.matrix[next][centre] == 2:
            return 2
        else:
            return 3

    def get_ordered_neighbour(self, centre, y, clockwise=False):
        ordered_neighbours = self.order_neighbours(centre, clockwise)
        return ordered_neighbours[(ordered_neighbours.index(y) + 1) % len(ordered_neighbours)]

    def order_neighbours(self, centre, clockwise=False):
        vertex_set = np.concatenate([np.where(np.logical_or(self.matrix[centre] == 2, self.matrix[centre] == 3))[0],
                                     np.where(np.logical_or(self.matrix[:, centre] == 2, self.matrix[:, centre] == 3))[0]]).tolist()
        ordered_set = [vertex_set.pop(0)]
        while len(vertex_set) != 0:
            for i in vertex_set:
                if self.matrix[ordered_set[len(ordered_set) - 1]][i] != 0 \
                        or self.matrix[i][ordered_set[len(ordered_set) - 1]] != 0:
                    ordered_set.append(i)
                    vertex_set.remove(i)
                    break
                elif self.matrix[ordered_set[0]][i] != 0 or self.matrix[i][ordered_set[0]] != 0:
                    ordered_set.insert(0, i)
                    vertex_set.remove(i)
                    break

        current = 0
        # case: centre is the South vertex
        if centre == self.south:
            if self.matrix[self.west][ordered_set[0]] != 0:
                ordered_set.reverse()

        # case: centre is the West vertex
        elif centre == self.west:
            if self.matrix[ordered_set[0]][self.north] != 0:
                ordered_set.reverse()

        # case: first vertex is in t1_leaving
        elif self.matrix[centre][ordered_set[0]] == 2:
            while self.matrix[centre][ordered_set[current]] == 2:
                current += 1
            if self.matrix[centre][ordered_set[current]] == 3:
                ordered_set.reverse()

        # case: first vertex is in t2_entering
        elif self.matrix[ordered_set[0]][centre] == 3:
            while self.matrix[ordered_set[current]][centre] == 3:
                current += 1
            if self.matrix[centre][ordered_set[current]] == 2:
                ordered_set.reverse()

        # case: first vertex is in t1_entering
        elif self.matrix[ordered_set[0]][centre] == 2:
            while self.matrix[ordered_set[current]][centre] == 2:
                current += 1
            if self.matrix[ordered_set[current]][centre] == 3:
                ordered_set.reverse()

        # case: first vertex is in t2_leaving
        elif self.matrix[centre][ordered_set[0]] == 3:
            while self.matrix[centre][ordered_set[current]] == 3:
                current += 1
            if self.matrix[ordered_set[current]][centre] == 2:
                ordered_set.reverse()

        if clockwise:
            ordered_set.reverse()
        return ordered_set

    def populate_t1_matrix(self):
        self.get_n_s_paths(self.south, [self.south])
        self.t1_matrix = np.empty((0, self.t1_longest_distance_value + 1), int)
        row_index = 0
        for path in self.n_s_paths:
            is_valid_path = True
            row = [-1] * (self.t1_longest_distance_value + 1)
            path_index = 0
            current_vertex = path[path_index]
            for distance in range(self.t1_longest_distance_value + 1):
                if path_index + 1 < len(path) and self.t1_longest_distance[path[path_index + 1]] <= distance:
                    path_index += 1
                    current_vertex = path[path_index]
                if row_index != 0 and self.t1_matrix[row_index - 1][distance] != current_vertex \
                        and current_vertex in self.t1_matrix[:, distance]:
                    is_valid_path = False
                    break
                row[distance] = current_vertex
            if is_valid_path:
                self.t1_matrix = np.append(self.t1_matrix, [row], axis=0)
                row_index += 1
        self.t1_matrix = self.t1_matrix.transpose()

    # while populating the t1_matrix we need N-S paths such that they are obtained in a DFS ordered manner with children
    # obtained in anticlockwise direction..... but in the REL we have S-N paths... so we construct the S-N path with
    # children obtained in clockwise direction and reverse the path when we reach N.
    def get_n_s_paths(self, source, path):
        if source == self.north: # base case of this recursive function as every S-N ends at N

            # making a deep copy of the path array as it changes during the recursive calls and wew want o save the
            # current state of this array
            path_deep_copy = [i for i in path]

            path_deep_copy.reverse() # reversing the array to get N-S path from the S-N path

            #iterating over the nodes in path and updating their longest distance from north
            for i in range(len(path_deep_copy)):
                node = path_deep_copy[i]
                self.t1_longest_distance[node] = max(self.t1_longest_distance[node], i) # index i represent the distance of node from north
                # updating the length of the longest N-S path
                self.t1_longest_distance_value = max(self.t1_longest_distance_value, self.t1_longest_distance[node])

            # adding this path in the n_s_paths
            self.n_s_paths.append(path_deep_copy)
            return

        # if we have not reached north yet then we get the children of the current source node and continue this DFS
        # to reach N from each children
        ordered_children = self.get_t1_ordered_children(source)
        for child in ordered_children:
            path.append(child)
            self.get_n_s_paths(child, path)
            path.remove(child)

    def get_t1_ordered_children(self, centre):
        ordered_neighbours = self.order_neighbours(centre, clockwise=True)
        index = 0
        ordered_children = []
        if centre == self.south:
            return ordered_neighbours
        while self.matrix[ordered_neighbours[index]][centre] != 3:
            index = (index + 1) % len(ordered_neighbours)
        while self.matrix[ordered_neighbours[index]][centre] == 3:
            index = (index + 1) % len(ordered_neighbours)
        while self.matrix[centre][ordered_neighbours[index]] == 2:
            ordered_children.append(ordered_neighbours[index])
            index = (index + 1) % len(ordered_neighbours)
        return ordered_children

    def populate_t2_matrix(self):
        self.get_w_e_paths(self.west, [self.west])
        self.t2_matrix = np.empty((0, self.t2_longest_distance_value + 1), int)
        row_index = 0
        for path in self.w_e_paths:
            is_valid_path = True
            row = [-1] * (self.t2_longest_distance_value + 1)
            path_index = 0
            current_vertex = path[path_index]
            for distance in range(self.t2_longest_distance_value + 1):
                if path_index + 1 < len(path) and self.t2_longest_distance[path[path_index + 1]] <= distance:
                    path_index += 1
                    current_vertex = path[path_index]
                if row_index != 0 and self.t2_matrix[row_index - 1][distance] != current_vertex \
                        and current_vertex in self.t2_matrix[:, distance]:
                    is_valid_path = False
                    break
                row[distance] = current_vertex
            if is_valid_path:
                self.t2_matrix = np.append(self.t2_matrix, [row], axis=0)
                row_index += 1

    def get_w_e_paths(self, source, path):
        self.t2_longest_distance[source] = max(self.t2_longest_distance[source], len(path) - 1)
        self.t2_longest_distance_value = max(self.t2_longest_distance_value, self.t2_longest_distance[source])
        if source == self.east:
            path_deep_copy = [i for i in path]
            self.w_e_paths.append(path_deep_copy)
            return
        ordered_children = self.get_t2_ordered_children(source)
        for child in ordered_children:
            path.append(child)
            self.get_w_e_paths(child, path)
            path.remove(child)

    def get_t2_ordered_children(self, centre):
        ordered_neighbours = self.order_neighbours(centre, clockwise=True)
        index = 0
        ordered_children = []
        if centre == self.west:
            return ordered_neighbours
        while self.matrix[centre][ordered_neighbours[index]] != 2:
            index = (index + 1) % len(ordered_neighbours)
        while self.matrix[centre][ordered_neighbours[index]] == 2:
            index = (index + 1) % len(ordered_neighbours)
        while self.matrix[centre][ordered_neighbours[index]] == 3:
            ordered_children.append(ordered_neighbours[index])
            index = (index + 1) % len(ordered_neighbours)
        return ordered_children

    def get_dimensions(self):
        for node in range(self.matrix.shape[0]):
            if node in [self.north, self.east, self.south, self.west]:
                continue
            row, col = np.where(self.t1_matrix[1:-1] == node)
            if row.shape[0] == 0:#remove this later
                continue
            counts = np.bincount(row)
            max_row = np.argmax(counts)
            indexes, = np.where(row == max_row)
            self.room_x[node] = col[indexes[0]]
            self.room_width[node] = col[indexes[-1]] - col[indexes[0]] + 1


            row, col = np.where(self.t2_matrix[:, 1:-1] == node)
            counts = np.bincount(col)
            max_col = np.argmax(counts)
            indexes, = np.where(col == max_col)
            self.room_y[node] = row[indexes[0]]
            self.room_height[node] = row[indexes[-1]] - row[indexes[0]] + 1
