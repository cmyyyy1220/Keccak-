'''
Author: AngieJC htk90uggk@outlook.com
Date: 2023-11-16 09:43:54
LastEditors: AngieJC htk90uggk@outlook.com
LastEditTime: 2023-11-17 17:08:11
'''

import keccak
import numpy as np
import random
import time

hash_value = 0x1234

# initialize matrix of linear layer
matrix, matrix_compress = keccak.init_keccak(lane_size = 2)

# initialize S-box
S, S_inv = keccak.init_Sbox()

# reture bin list of n, if list len < size, add 0 to the front
def int_to_bin_list(n, size):
    bin_list = list(bin(n)[2:])
    bin_list = [int(i) for i in bin_list]
    if len(bin_list) < size:
        bin_list = [0] * (size - len(bin_list)) + bin_list
    return bin_list

def bin_list_to_int(bin_list):
    ret = 0
    for i in bin_list:
        ret = ret * 2 + i
    return ret

# calculate matrix * X for partial index
def linear_layer(matrix, X, X_set, Y_set):
    Y = [0] * len(Y_set)
    for i in Y_set:
        for index in matrix[i]:
            if index <= X_set[-1]:
                Y[Y_set.index(i)] = Y[Y_set.index(i)] ^ X[X_set.index(index)]
    return Y

# match
def keccak_match(A, B, index):
    for b in B:
        for i in range(len(A)):
            if(A[i] == b[index[i]]):
                if i == len(A) - 1:
                    return True
                continue
            else:
                break
    return False

# embed list B to A with specific index
def embed_list(A, index, B):
    for i in range(len(index)):
        A[index[i]] = B[i]
    return A

# keccak encode
def super_Sbox(A):
    for Sbox_index in range(len(A) // 5):
        y = Sbox_index // 2
        z = Sbox_index % 2
        Sbox_in = 0
        for x in range(5):
            Sbox_in = Sbox_in * 2 + A[keccak.get_index(x, y, z)]
        Sbox_out = int_to_bin_list(S[Sbox_in], 5)
        for x in range(5):
            A[keccak.get_index(x, y, z)] = Sbox_out[x]
    return A
def keccak_enc(A, R, hash_size):
    A += [0] * (keccak.keccak_state_size - len(A))
    for r in range(R):
        A = np.dot(matrix, np.array(A)).tolist()
        for i in range(len(A)):
            A[i] %= 2
        A = super_Sbox(A)
    return bin_list_to_int(A[:hash_size])

'''Pseudocode
calculate S{0, 1} and 4 S[2]s from hash value (before Sbox)                             <2^2>
for g in range(2):                                                                      <2^1>
    A[16] = g
    for A{1, 2, 4, 6, 9, 11, 12, 13, 14} in range(2**9):                                <2^9>
        calculate to 8 bits L{1, 3, 7, 8, 10, 12, 14, 16}
        match L{1, 3, 7, 8, 10, 12, 14, 16} with S{0, 1, 2}                             Blue set remain 2^1
    for A[0] in range(2):                                                               <2^1>
        get all possible value of blue set (A[0]||A{1, 2, 4, 6, 9, 11, 13, 14})         Blue set remain 2^2
    for A{3, 5, 7, 8, 10, 15, 17} in range(2**7):
        get value of A
        match Keccak(A) with hash value
'''
def keccak_attack_1r_2l_16h():
    hash_array = int_to_bin_list(hash_value, 16)
    S0 = S_inv[hash_array[8] + 2 * (hash_array[6] + 2 * (hash_array[4] + 2 * (hash_array[2] + 2 * hash_array[0])))]
    S1 = S_inv[hash_array[9] + 2 * (hash_array[7] + 2 * (hash_array[5] + 2 * (hash_array[3] + 2 * hash_array[1])))]
    S0 = int_to_bin_list(S0, 5)
    S1 = int_to_bin_list(S1, 5)
    S2 = []
    S2_0 = hash_array[10]
    S2_1 = hash_array[12]
    S2_2 = hash_array[14]
    preimages = []
    for S2_3 in range(2): # 2^1
        for S2_4 in range(2): # 2^1
            S2.append(int_to_bin_list(S_inv[S2_4 + 2 * (S2_3 + 2 * (S2_2 + 2 * (S2_1 + 2 * S2_0)))], 5))
    S012 = []
    for s2 in S2: # 2^2
        S012.append(S0 + S1 + s2)

    for g in range(2): # 2^1
        A_set = []
        A = [0] * 18
        A[16] = g
        for B in range(2**9): # 2^9
            B = int_to_bin_list(B, 9)
            L = linear_layer(matrix_compress, B, [1, 2, 4, 6, 9, 11, 12, 13, 14], [1, 3, 7, 8, 10, 12, 14, 16])
            L[3] = L[3] ^ A[16]
            L[5] = L[5] ^ A[16]
            if keccak_match(L, S012, [5, 6, 8, 4, 10, 11, 12, 13]):
                A = embed_list(A, [1, 2, 4, 6, 9, 11, 12, 13, 14], B)
                for A0 in range(2): # 2^1
                    A[0] = A0
                    A_set.append(A.copy())
        # print(len(A_set))
        for R in range(2**7): # 2^7
            R = int_to_bin_list(R, 7)
            for A in A_set: # about 2^4
                A = embed_list(A, [3, 5, 7, 8, 10, 15, 17], R)
                hv = keccak_enc(A, 1, 16)
                if hv == hash_value and A[16] == 1 and A[17] == 1:
                    # print(A[:18])
                    print("0x{:04x}".format(bin_list_to_int(A[:16])))
                    preimages.append(A[:16])
    return preimages

def keccak_attack_2r_8l_80h():
    return 0

_HELP = '''
Usage: python3 keccak_attack.py rounds lane_size hash_size
'''

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print(_HELP)
        exit(1)
    nround = int(sys.argv[1])
    lane_size = int(sys.argv[2])
    hash_size = int(sys.argv[3])

    preimage = random.randint(0, 2**hash_size - 1)
    hash_value = keccak_enc(int_to_bin_list(preimage, hash_size) + [1, 1], nround, hash_size)
    print("preimage:", "0x{:04x}".format(preimage))
    print("hash:    ", "0x{:04x}".format(hash_value))
    print("searching preimages:")

    func_name = sys.argv[1] + "r_" + sys.argv[2] + "l_" + sys.argv[3] + "h"
    attack_start_time = time.time()
    if func_name == "1r_2l_16h":
        keccak_attack_1r_2l_16h()
    elif func_name == "2r_8l_80h":
        keccak_attack_2r_8l_80h()
    else:
        print("Err: no such configuration")
        exit(1)
    attack_end_time = time.time()
    print("attack time:", attack_end_time - attack_start_time, "seconds")

    # preimage = random.randint(0, 2**hash_size - 1)
    # print("---------------------- metadata -----------------------")
    # print("preimage:", "0x{:04x}".format(preimage))
    # hash_value = keccak_enc(int_to_bin_list(preimage, hash_size) + [1, 1], 1, hash_size)
    # print("hash:    ", "0x{:04x}".format(hash_value))
    # print("---------------------- attacking ----------------------")
    # print("Searching preimages for", "0x{:04x}".format(hash_value))
    # attack_start_time = time.time()
    # preimages = keccak_attack()
    # attack_end_time = time.time()
    # print("---------------------- checking -----------------------")
    # for preimage in preimages:
    #     print("checking", "0x{:04x}".format(bin_list_to_int(preimage)), end = " ")
    #     if keccak_enc(preimage + [1, 1], 1, hash_size) == hash_value:
    #         print("\033[32mpassed\033[0m")
    #     else:
    #         print("\033[31mfailed\033[0m")
    # print("----------------------- time --------------------------")
    # print("attack time:", attack_end_time - attack_start_time, "seconds")
    # print("start exhaustive searching...")
    # exhaustive_start_time = time.time()
    # for preimage in range(2**hash_size):
    #     if keccak_enc(int_to_bin_list(preimage, hash_size) + [1, 1], 1, hash_size) == hash_value:
    #         print("preimage found:", "0x{:04x}".format(preimage))
    # exhaustive_end_time = time.time()
    # print("exhaustive time:", exhaustive_end_time - exhaustive_start_time, "seconds")
