'''
Author: AngieJC htk90uggk@outlook.com
Date: 2023-11-02 19:57:04
LastEditors: AngieJC htk90uggk@outlook.com
LastEditTime: 2023-11-24 22:08:23
'''

from generic import find_mitm_attack, EXTENDED_SETTING, CLASSICAL_COMPUTATION, SINGLE_SOLUTION, ALL_SOLUTIONS
from util import PresentConstraints
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

keccak_state_size = 25
keccak_lane_size = 2
width_between_layers = 0.9
cell_width = 0.1
bleed = 0.1
CELL_BLUE = "blue"
CELL_RED = "red"
EDGE_WIDTH = 0.5
EDGE_BLUE = "blue"
EDGE_RED = "red"
EDGE_MATCH = "cyan"
EDGE_GLOBAL = "green"   # color of global edge
BOARD_COLOR = "black"   # color of cell board
BOARD_WIDTH = 0.5       # width of cell board

def get_xyz(index: int, dim: int = 2):
    if dim == 3:
        x = (index % (5 * keccak_lane_size)) // keccak_lane_size
        y = index // (5 * keccak_lane_size)
        z = index % keccak_lane_size
        return x, y, z
    elif dim == 2:
        x = index % 5
        y = index // 5
        return x, y

def get_index(x: int, y: int, z: int = None) -> int:
    x = (x + 5) % 5
    y = (y + 5) % 5
    if z == None:
        return x + y * 5
    z = (z + keccak_lane_size) % keccak_lane_size
    return (x + y * 5) * keccak_lane_size + z

def init_Sbox():
    S = [0 for i in range(2**5)]
    S_inv = [0 for i in range(2**5)]
    for x0 in range(2):
        for x1 in range(2):
            for x2 in range(2):
                for x3 in range(2):
                    for x4 in range(2):
                        y0 = x0 ^ (x1 ^ 1) * x2
                        y1 = x1 ^ (x2 ^ 1) * x3
                        y2 = x2 ^ (x3 ^ 1) * x4
                        y3 = x3 ^ (x4 ^ 1) * x0
                        y4 = x4 ^ (x0 ^ 1) * x1
                        x = x4 + 2 * (x3 + 2 * (x2 + 2 * (x1 + 2 * x0)))
                        y = y4 + 2 * (y3 + 2 * (y2 + 2 * (y1 + 2 * y0)))
                        S[x] = y
                        S_inv[y] = x
    return S, S_inv

def init_rho_offset():
    rho_offset = np.zeros(25, dtype = int)
    rho_offset[get_index(0, 0)] = 0
    x = 1
    y = 0
    for t in range(25 - 1):
        rho_offset[get_index(x, y)] = ((t + 1) * (t + 2) // 2) % keccak_lane_size
        x, y = y, (2 * x + 3 * y) % 5
    return rho_offset

def init_keccak(lane_size = 2):
    global keccak_state_size, keccak_lane_size
    keccak_lane_size = lane_size
    keccak_state_size = keccak_state_size * keccak_lane_size

    matrix = np.zeros((keccak_state_size, keccak_state_size), dtype = int)

    # 1. theta
    for i in range(keccak_state_size):
        matrix[i, i] = 1
        x, y, z = get_xyz(i, dim = 3)
        for j in range(5):
            matrix[i][get_index(x - 1, j, z)] = 1
            matrix[i][get_index(x + 1, j, z - 1)] = 1
    
    # 2. rho
    rho_offset = init_rho_offset()
    tmp = matrix.copy()
    for x in range(5):
        for y in range(5):
            offset = rho_offset[get_index(x, y)]
            for z in range(keccak_lane_size):
                matrix[get_index(x, y, z)] = tmp[get_index(x, y, z - offset)]
    
    # 3. pi
    tmp = matrix.copy()
    for x in range(5):
        for y in range(5):
            for z in range(keccak_lane_size):
                matrix[get_index(y, 2 * x + 3 * y, z)] = tmp[get_index(x, y, z)]
    
    # 4. compress matrix
    tmp = matrix.copy()
    matrix = []
    for i in range(keccak_state_size):
        row = []
        for j in range(keccak_state_size):
            if tmp[i][j] == 1:
                row.append(j)
        matrix.append(row)

    return tmp, matrix

def init_coord(nrounds = 2, nblocks = 2, toy = False):
    y_max = keccak_state_size * cell_width + bleed
    cell_name_prefix = ["A_", "L_", "S_"]
    cell_coord = {}
    for block in range(nblocks):
        for round in range(nrounds):
            for layer in range(3):
                x = (3 * block * nrounds + 3 * round + layer) * (width_between_layers + cell_width) + bleed
                cell_count = keccak_state_size
                cell_high = cell_width
                if layer == 2:
                    cell_count = cell_count // 5
                    cell_high = cell_high * 5
                for i in range(cell_count):
                    y = y_max - (i + 1) * cell_high
                    cell_name = cell_name_prefix[layer] + str(block) + "m_" + str(round) + "r_" + str(i) + "i"
                    cell_coord[cell_name] = {}
                    cell_coord[cell_name]["x"] = x
                    cell_coord[cell_name]["y"] = y
                    cell_coord[cell_name]["high"] = cell_high
    if toy:
        x = (3 * nrounds) * (width_between_layers + cell_width) + bleed
        for h in range(hash_size):
            y = y_max - (h + 1) * cell_width
            cell_name = "B_0m_" + str(nrounds) + "r_" + str(h) + "i"
            cell_coord[cell_name] = {}
            cell_coord[cell_name]["x"] = x
            cell_coord[cell_name]["y"] = y
            cell_coord[cell_name]["high"] = cell_width
    return cell_coord

def draw_model(nrounds, cons, cell_var_covered, global_lincons, nblocks, toy = False):
    cell_coords = init_coord(nrounds, nblocks, toy)
    x_max = (3 * nrounds * nblocks - 2) * (width_between_layers + cell_width) + cell_width + bleed
    if toy:
        x_max += (width_between_layers + cell_width) * 2
    y_max = keccak_state_size * cell_width + bleed
    fig, ax = plt.subplots(figsize = (15, 15))

    # draw cells
    for cell_name in cell_coords:
        if cell_name in cell_var_covered["forward"] and cell_var_covered["forward"][cell_name] == 1:
            cell_color = CELL_BLUE
        elif cell_name in cell_var_covered["backward"] and cell_var_covered["backward"][cell_name] == 1:
            cell_color = CELL_RED
        else:
            continue
        cell_square = Rectangle((cell_coords[cell_name]["x"], cell_coords[cell_name]["y"]), cell_width, cell_coords[cell_name]["high"], 
                                facecolor = cell_color, edgecolor = BOARD_COLOR, linewidth = BOARD_WIDTH)
        ax.add_patch(cell_square)
    
    # draw edges
    for edge_name in cons.edge_name_to_data:
        cell_name1 = cons.edge_name_to_data[edge_name][0]
        cell_name2 = cons.edge_name_to_data[edge_name][1]
        if "A_0m_0r_" in cell_name2:
            continue
        if cell_var_covered["forward"][cell_name1] == 1 and cell_var_covered["forward"][cell_name2] == 1:
            edge_color = EDGE_BLUE
        elif cell_var_covered["backward"][cell_name1] == 1 and cell_var_covered["backward"][cell_name2] == 1:
            edge_color = EDGE_RED
        elif (cell_var_covered["forward"][cell_name1] == 1 and cell_var_covered["backward"][cell_name2] == 1) or \
                (cell_var_covered["backward"][cell_name1] == 1 and cell_var_covered["forward"][cell_name2] == 1):
            if global_lincons[edge_name] == 1:
                edge_color = EDGE_GLOBAL
            else:
                edge_color = EDGE_MATCH
        else:
            continue
        x1 = cell_coords[cell_name1]["x"] + cell_width
        x2 = cell_coords[cell_name2]["x"]
        if "S" in cell_name1:
            x, y, z = get_xyz(int(cell_name2.split("_")[-1][:-1]), 3)
            y1 = cell_coords[cell_name1]["y"] + cell_coords[cell_name1]["high"] - (x + 0.5) * cell_width
        else:
            y1 = cell_coords[cell_name1]["y"] + cell_coords[cell_name1]["high"] / 2
        if "S" in cell_name2:
            x, y, z = get_xyz(int(cell_name1.split("_")[-1][:-1]), 3)
            y2 = cell_coords[cell_name2]["y"] + cell_coords[cell_name2]["high"] - (x + 0.5) * cell_width
        else:
            y2 = cell_coords[cell_name2]["y"] + cell_coords[cell_name2]["high"] / 2
        ax.plot([x1, x2], [y1, y2], color = edge_color, linewidth = EDGE_WIDTH)


    fig.set_size_inches(x_max + bleed, y_max + bleed)
    ax.set_xlim(0, x_max + bleed)
    ax.set_ylim(0, y_max + bleed)
    ax.axis("off")
    plt.savefig("imgs/Keccak_" + str(nrounds) + "r_" + str(lane_size) + "l_" + str(hash_size) + "h.pdf", bbox_inches = "tight", pad_inches = 0.0)
    return
            
'''
Each round has 3 layers:
* layer 0: "A", input of this round, 11-branch cells
* layer 1: "L", state after linear transformation, 11-XOR cells
* layer 2: "S", S-box, 5-bit cells

There are *global* edges between hash and the first h bits of c(apacity).
'''
def gen_toy_keccak_model(nrounds = 1, lane_size = 2, hash_size = 16):
    cons = PresentConstraints(nrounds = 3 * nrounds + 1)
    raw_matrix, matrix = init_keccak(lane_size = lane_size)

    for round in range(nrounds):
        # layer 0
        for i in range(25 * keccak_lane_size):
            if round == 0 and i >= keccak_state_size - hash_size:
                continue
            cell_name = "A_0m_" + str(round) + "r_" + str(i) + "i"
            cons.add_cell(r = 3 * round, w = 1, name = cell_name)
        # layer 1
        for i in range(25 * keccak_lane_size):
            cell_name = "L_0m_" + str(round) + "r_" + str(i) + "i"
            width = 11
            if round == 0:
                for pos in matrix[i]:
                    if pos >= keccak_state_size - hash_size:
                        width -= 1
            cons.add_cell(r = 3 * round + 1, w = width, name = cell_name)
            # edges between layer 0 and layer 1
            for j in matrix[i]:
                if round == 0 and j >= keccak_state_size - hash_size:
                    continue
                cons.add_edge(c1 = "A_0m_" + str(round) + "r_" + str(j) + "i", 
                              c2 = cell_name, w = 1)
        # layer 2
        for i in range(5 * keccak_lane_size):
            cell_name = "S_0m_" + str(round) + "r_" + str(i) + "i"
            cons.add_cell(r = 3 * round + 2, w = 5, name = cell_name)
            # edges between layer 1 and layer 2
            y = i // keccak_lane_size
            z = i % keccak_lane_size
            for x in range(5):
                cons.add_edge(c1 = "L_0m_" + str(round) + "r_" + str(get_index(x, y, z)) + "i", 
                              c2 = cell_name, w = 1)
    # for each round, connect "S%i_%i" to "A%i_%i"
    for i in range(nrounds - 1):
        for x in range(5):
            for y in range(5):
                for z in range(keccak_lane_size):
                    cons.add_edge(c1 = "S_0m_" + str(i) + "r_" + str(y * keccak_lane_size + z) + "i", 
                                  c2 = "A_0m_" + str(i + 1) + "r_" + str(get_index(x, y, z)) + "i", 
                                  w = 1)
    # for last round, h 2-branch cells, 
    # connect to first and last h bits of c.
    for h in range(hash_size):
        cell_name = "B_0m_" + str(nrounds) + "r_" + str(h) + "i"
        cons.add_cell(r = 3 * nrounds, w = 1, name = cell_name)
        # connect 5-bit cells to 2-branch cells
        x, y, z = get_xyz(h, dim = 3)
        cons.add_edge(c1 = "S_0m_" + str(nrounds - 1) + "r_" + str(y * keccak_lane_size + z) + "i", 
                      c2 = cell_name, w = 1)
        # connect 2-branch cells to first h bits of c
        edge_name = cons.add_edge(c1 = cell_name, 
                                  c2 = "A_0m_0r_" + str(keccak_state_size - 2 * hash_size + h) + "i", 
                                  w = 1)
        cons.set_global(edge_name)
    
    possible_middle_layers = [i for i in range(3 * nrounds)]

    return cons, possible_middle_layers

'''
Each round has 3 layers:
 * layer 0: "A", input of this round, n-branch cells
 * layer 1: "L", state after linear transformation, n-XOR cells
 * layer 2: "S", S-box, 5-bit cells

Unecessary cells:
 1 the last h A cells of the first block's first round
 2 the first h L cells of the last round of 1 to n-1 blocks
 3 the first h/5 S cells of the last round of 1 to n-1 blocks
 4 the first h A cells of the first round of 2 to n blocks
 5 the last t-h L cells of the last block's last round
 6 all S cells of the last block's last round
 * if nblocks = 1, then 2, 3, 4 should be removed
'''
def gen_keccak_model(nrounds = 2, lane_size = 8, hash_size = 80, nblocks = 2):
    cons = PresentConstraints(nrounds = 3 * nrounds * nblocks - 1)
    matrix, matrix_compress = init_keccak(lane_size = lane_size)
    block_size = keccak_state_size - 2 * hash_size

    for block in range(nblocks):
        for round in range(nrounds):
            # layer 0, A
            for index in range(keccak_state_size):
                if round == 0:
                    if block == 0 and index >= keccak_state_size - hash_size:
                        continue
                width = 1
                if block > 0 and index < block_size and round == 0:
                    width = 2
                cell_name = "A_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                cons.add_cell(r = 3 * (nrounds * block + round), w = width, name = cell_name)
                # edges between layer S and layer A
                if round == 0:
                    S_block = block - 1
                    S_round = nrounds - 1
                else:
                    S_block = block
                    S_round = round - 1
                x, y, z = get_xyz(index, dim = 3)
                S_index = y * keccak_lane_size + z
                cell_name_ = "S_" + str(S_block) + "m_" + str(S_round) + "r_" + str(S_index) + "i"
                if cell_name_ in cons.cell_name_to_data:
                    cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
            # layer 1, L
            for index in range(keccak_state_size):
                if round == nrounds - 1 and index >= hash_size and block == nblocks - 1:
                    continue
                cell_name = "L_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                width = 11
                if round == 0:
                    if block == 0:
                        for pos in matrix_compress[index]:
                            if pos >= keccak_state_size - hash_size:
                                width -= 1
                cons.add_cell(r = 3 * (nrounds * block + round) + 1, w = width, name = cell_name)
                # edges between layer 0 and layer 1
                for pos in matrix_compress[index]:
                    cell_name_ = "A_" + str(block) + "m_" + str(round) + "r_" + str(pos) + "i"
                    if cell_name_ in cons.cell_name_to_data:
                        cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
            # layer 2, S
            for index in range(5 * keccak_lane_size):
                if round == nrounds - 1 and block == nblocks - 1:
                    continue
                cell_name = "S_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                cons.add_cell(r = 3 * (nrounds * block + round) + 2, w = 5, name = cell_name)
                # edges between layer 1 and layer 2
                y = index // keccak_lane_size
                z = index % keccak_lane_size
                for x in range(5):
                    cell_name_ = "L_" + str(block) + "m_" + str(round) + "r_" + str(get_index(x, y, z)) + "i"
                    if cell_name_ in cons.cell_name_to_data:
                        cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
    
    # for last rount of last block, 
    # connect n-XOR cells with first h bits of c of the first block's first round
    for index in range(hash_size):
        cell_name = "L_" + str(nblocks - 1) + "m_" + str(round) + "r_" + str(index) + "i"
        cell_name_ = "A_0m_0r_" + str(keccak_state_size - 2 * hash_size + index) + "i"
        if cell_name in cons.cell_name_to_data and cell_name_ in cons.cell_name_to_data:
            edge_name = cons.add_edge(c1 = cell_name, c2 = cell_name_, w = 1)
            cons.set_global(edge_name)
    
    # full layers
    possible_middle_layers = [i for i in range(3 * nrounds * nblocks - 1)]

    return cons, possible_middle_layers[:-1]

def gen_tidy_keccak_model(nrounds = 2, lane_size = 8, hash_size = 80, nblocks = 2):
    cons = PresentConstraints(nrounds = 3 * nrounds * nblocks - 1)
    matrix, matrix_compress = init_keccak(lane_size = lane_size)
    block_size = keccak_state_size - 2 * hash_size

    for block in range(nblocks):
        for round in range(nrounds):
            # layer 0, A
            for index in range(keccak_state_size):
                if round == 0:
                    if block == 0 and (index >= keccak_state_size - hash_size or index < block_size):
                        continue
                    elif block != 0 and index < block_size:
                        continue
                cell_name = "A_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                cons.add_cell(r = 3 * (nrounds * block + round), w = 1, name = cell_name)
                # edges between layer S and layer A
                if round == 0:
                    S_block = block - 1
                    S_round = nrounds - 1
                else:
                    S_block = block
                    S_round = round - 1
                x, y, z = get_xyz(index, dim = 3)
                S_index = y * keccak_lane_size + z
                cell_name_ = "S_" + str(S_block) + "m_" + str(S_round) + "r_" + str(S_index) + "i"
                if cell_name_ in cons.cell_name_to_data:
                    cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
            # layer 1, L
            for index in range(keccak_state_size):
                if round == nrounds - 1 and ((index < block_size and block < nblocks - 1) or \
                    index >= hash_size and block == nblocks - 1):
                    continue
                cell_name = "L_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                width = 11
                if round == 0:
                    if block == 0:
                        for pos in matrix_compress[index]:
                            if pos >= keccak_state_size - hash_size or pos < block_size:
                                width -= 1
                        if index < block_size:
                            width += 1
                    else:
                        for pos in matrix_compress[index]:
                            if pos < block_size:
                                width -= 1
                        if index < block_size:
                            width += 1
                cons.add_cell(r = 3 * (nrounds * block + round) + 1, w = width, name = cell_name)
                # edges between layer 0 and layer 1
                for pos in matrix_compress[index]:
                    cell_name_ = "A_" + str(block) + "m_" + str(round) + "r_" + str(pos) + "i"
                    if cell_name_ in cons.cell_name_to_data:
                        cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
            # layer 2, S
            for index in range(5 * keccak_lane_size):
                if (round == nrounds - 1 and block < nblocks - 1 and index < block_size // 5 \
                    and block != nblocks # this condiction will never be satisfied since block is 0 to nblocks - 1
                    ) \
                    or (round == nrounds - 1 and block == nblocks - 1):
                    continue
                cell_name = "S_" + str(block) + "m_" + str(round) + "r_" + str(index) + "i"
                cons.add_cell(r = 3 * (nrounds * block + round) + 2, w = 5, name = cell_name)
                # edges between layer 1 and layer 2
                y = index // keccak_lane_size
                z = index % keccak_lane_size
                for x in range(5):
                    cell_name_ = "L_" + str(block) + "m_" + str(round) + "r_" + str(get_index(x, y, z)) + "i"
                    if cell_name_ in cons.cell_name_to_data:
                        cons.add_edge(c1 = cell_name_, c2 = cell_name, w = 1)
    
    # for last rount of last block, 
    # connect n-XOR cells with first h bits of c of the first block's first round
    for index in range(hash_size):
        cell_name = "L_" + str(nblocks - 1) + "m_" + str(round) + "r_" + str(index) + "i"
        cell_name_ = "A_0m_0r_" + str(keccak_state_size - 2 * hash_size + index) + "i"
        if cell_name in cons.cell_name_to_data and cell_name_ in cons.cell_name_to_data:
            edge_name = cons.add_edge(c1 = cell_name, c2 = cell_name_, w = 1)
            cons.set_global(edge_name)
    
    # full layers
    possible_middle_layers = [i for i in range(3 * nrounds * nblocks)]
    for block in range(nblocks):
        possible_middle_layers.remove(3 * nrounds * block)
        possible_middle_layers.remove(3 * nrounds * block + 3 * nrounds - 1)
        possible_middle_layers.remove(3 * nrounds * block + 3 * nrounds - 2)

    return cons, possible_middle_layers

_HELP = '''
Usage: python3 keccak.py rounds lane_size hash_size
'''

if __name__ == "__main__":
    import sys

    argc = len(sys.argv)
    if argc != 4:
        print(_HELP)
        exit(1)
    
    computation_model = CLASSICAL_COMPUTATION

    cut_forward = []
    cut_backward = []
    covered_round = None
    optimize_with_mem = False
    time_target = None

    nrounds = int(sys.argv[1])
    lane_size = int(sys.argv[2])
    hash_size = int(sys.argv[3])
    nblocks = (hash_size // (25 * lane_size - 2 * hash_size)) + (hash_size % (25 * lane_size - 2 * hash_size) != 0)
    func_name = sys.argv[1] + "r_" + sys.argv[2] + "l_" + sys.argv[3] + "h"
    if func_name == "1r_2l_16h":
        cons, possible_middle_layers = gen_toy_keccak_model(nrounds = nrounds, lane_size = lane_size, hash_size = hash_size)
    else:
        cons, possible_middle_layers = gen_keccak_model(nrounds = nrounds, lane_size = lane_size, hash_size = hash_size, nblocks = nblocks)

    cell_var_covered, global_lincons = find_mitm_attack(
        cons, 
        time_target = time_target, 
        setting = EXTENDED_SETTING, 
        computation_model = computation_model, 
        flag = SINGLE_SOLUTION, 
        optimize_with_mem = optimize_with_mem, 
        covered_round = covered_round, 
        cut_forward = cut_forward, 
        cut_backward = cut_backward, 
        possible_middle_rounds = possible_middle_layers, 
        nrounds = nrounds, 
        lane_size = lane_size, 
        hash_size = hash_size
    )

    if func_name == "1r_2l_16h":
        draw_model(nrounds, cons, cell_var_covered, global_lincons, 1, True)
    else:
        draw_model(nrounds, cons, cell_var_covered, global_lincons, nblocks, False)
