import csv
import os
import pickle

import numpy as np

from .graph import Graph


class Node:
    n = 1

    def __init__(self, pos, direction, name="default"):
        if name == "default":
            self.name = str(Node.n)
            Node.n += 1
        else:
            self.name = name
        self.pos = pos
        self.direction = direction

    def flow(self, node):
        return (self.direction[0] + node.direction[0]) ** 2 + (self.direction[1] + node.direction[1]) ** 2

    def rotateDirection(self, theta):
        theta = np.deg2rad(theta)
        vec = np.array(self.direction, dtype=float)
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        res = np.dot(rotMatrix, vec)
        # Round small floating-point noise to nearest integer axis direction.
        res = np.rint(res).astype(int)
        return int(res[0]), int(res[1])

    def rotateAndTranslatePos(self, theta, x, y):
        theta = np.deg2rad(theta)
        vec = np.matrix(self.pos).transpose()
        vec = np.append(vec, [[1]], axis=0)
        rotMatrix = np.matrix(
            [[np.cos(theta), -np.sin(theta), x + 0.5], [np.sin(theta), np.cos(theta), y + 0.5], [0, 0, 1]]
        )

        res = np.dot(rotMatrix, vec)
        res = res[0:2, 0]
        res = res.tolist()

        return res[0][0], res[1][0]

    def globalPosAndDirection(self, theta, x, y, name="default"):
        return Node(self.rotateAndTranslatePos(theta, x, y), self.rotateDirection(theta), name)


class Tile:
    def __init__(self, csv_row):
        self.x = float(csv_row[0])
        self.y = float(csv_row[1])
        self.type = csv_row[2]
        self.rotation = float(csv_row[3])

    def create_nodes(self):
        return {}, []

    def create_edges(self, tile_map):
        return []

    def connect_node(self, node, tile_map):

        next_tile_pos_x = self.x + node.direction[0]
        next_tile_pos_y = self.y + node.direction[1]
        t = self.get_tile(next_tile_pos_x, next_tile_pos_y, tile_map)
        if t is None:
            return None

        while t is not None and t.type == "straight":
            next_tile_pos_x = next_tile_pos_x + node.direction[0]
            next_tile_pos_y = next_tile_pos_y + node.direction[1]
            t = self.get_tile(next_tile_pos_x, next_tile_pos_y, tile_map)

        if t is None:
            return None

        if t.type == "turn":
            if t.node1.flow(node) == 4:
                return [node.name, t.node1.name, "f"]
            elif t.node4.flow(node) == 4:
                return [node.name, t.node4.name, "f"]
        elif t.type == "3way":
            if t.node1.flow(node) == 4:
                return [node.name, t.node1.name, "f"]
            elif t.node3.flow(node) == 4:
                return [node.name, t.node3.name, "f"]
            elif t.node5.flow(node) == 4:
                return [node.name, t.node5.name, "f"]
        elif t.type == "4way":
            if t.node1.flow(node) == 4:
                return [node.name, t.node1.name, "f"]
            elif t.node3.flow(node) == 4:
                return [node.name, t.node3.name, "f"]
            elif t.node5.flow(node) == 4:
                return [node.name, t.node5.name, "f"]
            elif t.node7.flow(node) == 4:
                return [node.name, t.node7.name, "f"]

    def get_tile(self, x, y, tile_map):
        for tile in tile_map:
            if tile.x == x and tile.y == y:
                return tile


class TurnTile(Tile):
    name = 1000
    node1_default = Node((-0.25, 0.5), (0, -1), "node1_default")
    node2_default = Node((-0.5, 0.25), (-1, 0), "node2_default")

    node3_default = Node((0.25, 0.5), (0, 1), "node3_default")
    node4_default = Node((-0.5, -0.25), (1, 0), "node4_default")
    

    def create_nodes(self):
        x = self.x
        y = self.y
        theta = self.rotation
        self.node1 = TurnTile.node1_default.globalPosAndDirection(theta, x, y, self.getNodeName())
        self.node2 = TurnTile.node2_default.globalPosAndDirection(theta, x, y, self.getNodeName())
        self.node3 = TurnTile.node3_default.globalPosAndDirection(theta, x, y, self.getNodeName())
        self.node4 = TurnTile.node4_default.globalPosAndDirection(theta, x, y, self.getNodeName())                                            
        node_loc = {self.node1.name: self.node1.pos, self.node2.name: self.node2.pos, self.node3.name: self.node3.pos, self.node4.name: self.node4.pos}
        edges = [
            [self.node1.name, self.node2.name, "r"],
            [self.node4.name, self.node3.name, "l"],
        ]
        return node_loc, edges

    def create_edges(self, tile_map):
        edges = [ self.connect_node(self.node2, tile_map), self.connect_node(self.node3, tile_map) ]
        return edges

    def getNodeName(self):
        res = "turn" + str(TurnTile.name)
        TurnTile.name += 1
        return res


class ThreeWayTile(Tile):
    node1_default = Node((0.50, 0.25), (-1, 0), "node1_default")
    node2_default = Node((0.25, 0.50), (0, 1), "node2_default")
    node3_default = Node((-0.25, 0.50), (0, -1), "node3_default")
    node4_default = Node((-0.50, 0.25), (-1, 0), "node4_default")
    node5_default = Node((-0.50, -0.25), (1, 0), "node5_default")
    node6_default = Node((0.50, -0.25), (1, 0), "node6_default")

    def create_nodes(self):
        x = self.x
        y = self.y
        theta = self.rotation
        self.node1 = ThreeWayTile.node1_default.globalPosAndDirection(theta, x, y)
        self.node2 = ThreeWayTile.node2_default.globalPosAndDirection(theta, x, y)
        self.node3 = ThreeWayTile.node3_default.globalPosAndDirection(theta, x, y)
        self.node4 = ThreeWayTile.node4_default.globalPosAndDirection(theta, x, y)
        self.node5 = ThreeWayTile.node5_default.globalPosAndDirection(theta, x, y)
        self.node6 = ThreeWayTile.node6_default.globalPosAndDirection(theta, x, y)
        node_loc = {
            self.node1.name: self.node1.pos,
            self.node2.name: self.node2.pos,
            self.node3.name: self.node3.pos,
            self.node4.name: self.node4.pos,
            self.node5.name: self.node5.pos,
            self.node6.name: self.node6.pos,
        }

        edges = [
            [self.node1.name, self.node2.name, "r"],
            [self.node1.name, self.node4.name, "s"],
            [self.node3.name, self.node4.name, "r"],
            [self.node3.name, self.node6.name, "l"],
            [self.node5.name, self.node2.name, "l"],
            [self.node5.name, self.node6.name, "s"],
        ]
        return node_loc, edges

    def create_edges(self, tile_map):
        edges = [
            self.connect_node(self.node2, tile_map),
            self.connect_node(self.node4, tile_map),
            self.connect_node(self.node6, tile_map),
        ]
        return edges


class FourWayTile(Tile):
    node1_default = Node((0.50, 0.25), (-1, 0), "node1_default")
    node2_default = Node((0.25, 0.50), (0, 1), "node2_default")
    node3_default = Node((-0.25, 0.50), (0, -1), "node3_default")
    node4_default = Node((-0.50, 0.25), (-1, 0), "node4_default")
    node5_default = Node((-0.50, -0.25), (1, 0), "node5_default")
    node6_default = Node((-0.25, -0.50), (0, -1), "node6_default")
    node7_default = Node((0.25, -0.50), (0, 1), "node7_default")
    node8_default = Node((0.50, -0.25), (1, 0), "node8_default")

    def create_nodes(self):
        x = self.x
        y = self.y
        theta = self.rotation
        self.node1 = FourWayTile.node1_default.globalPosAndDirection(theta, x, y)
        self.node2 = FourWayTile.node2_default.globalPosAndDirection(theta, x, y)
        self.node3 = FourWayTile.node3_default.globalPosAndDirection(theta, x, y)
        self.node4 = FourWayTile.node4_default.globalPosAndDirection(theta, x, y)
        self.node5 = FourWayTile.node5_default.globalPosAndDirection(theta, x, y)
        self.node6 = FourWayTile.node6_default.globalPosAndDirection(theta, x, y)
        self.node7 = FourWayTile.node7_default.globalPosAndDirection(theta, x, y)
        self.node8 = FourWayTile.node8_default.globalPosAndDirection(theta, x, y)
        node_loc = {
            self.node1.name: self.node1.pos,
            self.node2.name: self.node2.pos,
            self.node3.name: self.node3.pos,
            self.node4.name: self.node4.pos,
            self.node5.name: self.node5.pos,
            self.node6.name: self.node6.pos,
            self.node7.name: self.node7.pos,
            self.node8.name: self.node8.pos,
        }

        edges = [
            [self.node1.name, self.node2.name, "r"],
            [self.node1.name, self.node4.name, "s"],
            [self.node1.name, self.node6.name, "l"],
            [self.node3.name, self.node4.name, "r"],
            [self.node3.name, self.node8.name, "l"],
            [self.node3.name, self.node6.name, "s"],
            [self.node5.name, self.node2.name, "l"],
            [self.node5.name, self.node8.name, "s"],
            [self.node5.name, self.node6.name, "r"],
            [self.node7.name, self.node8.name, "r"],
            [self.node7.name, self.node2.name, "s"],
            [self.node7.name, self.node4.name, "l"],
        ]

        return node_loc, edges

    def create_edges(self, tile_map):
        edges = []
        edges.append(self.connect_node(self.node2, tile_map))
        edges.append(self.connect_node(self.node4, tile_map))
        edges.append(self.connect_node(self.node6, tile_map))
        edges.append(self.connect_node(self.node8, tile_map))
        return edges


class StraightTile(Tile):
    pass


class graph_creator:
    def __init__(self):
        self.node_locations = {}
        self.edges = []
        self.tile_map = []
        self._mid_counter = 1
        self.INTERMEDIATE_COUNT = 4

    def add_node_locations(self, node_loc):
        node_loc = {k: (v[0]*0.6, v[1]*0.6) for k, v in node_loc.items()}
        self.node_locations.update(node_loc)

    def add_edges(self, ed):
        for edge in ed:
            if edge is None or len(edge) < 3:
                continue
            source = edge[0]
            target = edge[1]
            action = edge[2]
            src_pos = self.node_locations[source]
            tgt_pos = self.node_locations[target]
            manhattan_dist = abs(src_pos[0] - tgt_pos[0]) + abs(src_pos[1] - tgt_pos[1])

            mid_count = self.INTERMEDIATE_COUNT

            def bezier_pts(p0, p2, control, n):
                pts = []
                for i in range(0, n + 2):
                    t = float(i) / (n + 1)
                    a = (1 - t) * (1 - t)
                    b = 2 * (1 - t) * t
                    c = t * t
                    x = a * p0[0] + b * control[0] + c * p2[0]
                    y = a * p0[1] + b * control[1] + c * p2[1]
                    pts.append((x, y))
                return pts

            if action in ("l", "r"):
                v = (tgt_pos[0] - src_pos[0], tgt_pos[1] - src_pos[1])
                perp = (-v[1], v[0])
                norm = (perp[0] ** 2 + perp[1] ** 2) ** 0.5
                if norm == 0:
                    norm = 1.0
                perp_unit = (perp[0] / norm, perp[1] / norm)
                offset = 0.25
                # Invert side mapping so curves follow node turning direction visually
                side = -1.0 if action == "l" else 1.0
                control = ((src_pos[0] + tgt_pos[0]) / 2.0 + side * perp_unit[0] * offset,
                           (src_pos[1] + tgt_pos[1]) / 2.0 + side * perp_unit[1] * offset)
                points = bezier_pts(src_pos, tgt_pos, control, mid_count)
            else:
                points = [
                    (
                        src_pos[0] + (t / float(mid_count + 1)) * (tgt_pos[0] - src_pos[0]),
                        src_pos[1] + (t / float(mid_count + 1)) * (tgt_pos[1] - src_pos[1]),
                    )
                    for t in range(0, mid_count + 2)
                ]

            prev_name = source
            for i in range(1, len(points)):
                pt = points[i]
                if i < len(points) - 1:
                    mid_name = "mid" + str(self._mid_counter)
                    self._mid_counter += 1
                    self.node_locations[mid_name] = (pt[0], pt[1])
                    curr_name = mid_name
                else:
                    curr_name = target

                prev_pos = self.node_locations[prev_name]
                curr_pos = self.node_locations[curr_name]
                seg_weight = abs(prev_pos[0] - curr_pos[0]) + abs(prev_pos[1] - curr_pos[1])
                self.edges.append([prev_name, curr_name, seg_weight, action])
                prev_name = curr_name

    def pickle_save(self, name="duckietown_map.pkl"):
        afile = open(r"maps/duckietown_map.pkl", "w+")
        pickle.dump([self.edges, self.node_locations], afile)
        afile.close()

    def build_graph_from_csv(self, csv_filename="map.csv"):
        script_dir = os.path.dirname(__file__)
        map_path = script_dir + "/../../src/maps/" + csv_filename
        with open(map_path + ".csv", "r", newline="") as f:
            spamreader = csv.reader(f, skipinitialspace=True)
            for i, row in enumerate(spamreader):
                if i != 0:
                    row_ = [element.strip() for element in row]  # remove white spaces
                    if len(row_) < 4 or not any(row_):
                        continue
                    if row_[2] == "turn":
                        self.tile_map.append(TurnTile(row_))
                    elif row_[2] == "3way":
                        self.tile_map.append(ThreeWayTile(row_))
                    elif row_[2] == "4way":
                        self.tile_map.append(FourWayTile(row_))
                    elif row_[2] == "straight":
                        self.tile_map.append(StraightTile(row_))
        self.generate_node_locations()
        self.generate_edges()

        duckietown_graph = Graph()
        for edge in self.edges:
            duckietown_graph.add_edge(edge[0], edge[1], edge[2], edge[3])
        duckietown_graph.set_node_positions(self.node_locations)

        return duckietown_graph

    def generate_node_locations(self):
        for tile in self.tile_map:
            node_loc, edges = tile.create_nodes()
            self.add_node_locations(node_loc)
            self.add_edges(edges)

    def generate_edges(self):
        for tile in self.tile_map:
            edges = tile.create_edges(self.tile_map)
            self.add_edges(edges)



if __name__ == "__main__":
    gc = graph_creator()
    duckietown_graph = gc.build_graph_from_csv(csv_filename="map")
    # Node locations (for visual representation) and heuristics calculation
    # node_locations, edges = gc.get_map_226()
    # gc.add_node_locations(node_locations)
    # gc.add_edges(edges)
    # gc.pickle_save()
    print(duckietown_graph._edges)
    print(duckietown_graph.node_positions)
    duckietown_graph.draw(script_dir="out/generate_duckietown_map", map_name="duckietown_226")
