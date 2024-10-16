src/：          源代码
    imgs/：     生成的中间相遇攻击方案
    lp/：       lp 文件和 sol 文件

mitm-milp/：    论文提供的源代码

依赖：
	1. pyscipopt
	2. gurobipy
	3. numpy
	4. matplotlib

src/：
keccak.py
    描述：  寻找特定版本 keccak 的中间相遇攻击
    用法：  python3 keccak.py <round> <lane size> <hash size>
    函数：
     29  行 get_xyz：               根据一维索引返回三维坐标
     40  行 get_index：             根据三维坐标返回一维索引
     48  行 init_Sbox：             初始化 S 盒
     67  行 init_rho_offset：       初始化 \rho 操作的偏移量
     77  行 init_keccak：           初始化线性层（\theta + \rho + \pi）矩阵，
                                    返回的第一个为原始矩阵，每行只有 11 个 1，
                                    其余为 0；第二个为压缩矩阵，只保存第一个
                                    矩阵中为 1 的位置，每行只有 11 个数
     120 行 init_coord：            计算所有节点在导出的 PDF 上的坐标，方便画图
     151 行 draw_model：            将结果转成 PDF
     219 行 gen_toy_keccak_model：  生成 1 轮 keccak[r=18, c=32, h=16] 的攻击
                                    模型，由于有些东西写死了又不好改，这个函数
                                    就单独放在这了
     297 行 gen_keccak_model：      生成 MILP 模型，不删除中间节点
     371 行 gen_tidy_keccak_model： 生成 MILP 模型，删除中间节点，两者区别可参考
                                    Keccak_3r_8l_80h.pdf 和 Keccak_3r_8l_80h_tidy.pdf
     464 行，主要逻辑：
        1. 488 行调用 gen_keccak_model，可换成 gen_tidy_keccak_model
        2. 490 行寻找中间相遇攻击
        3. 506 行画图

keccak_attack.py
    描述：  实际中间相遇攻击寻找原像（未考虑轮常量）
    用法：  python3 keccak_attack.py <round> <lane size> <hash size>
    限制：  目前只有 1 轮 keccak[r=18, c=32, h=16] 是实际找到了中间相
            遇攻击的，即只有 python3 keccak_attack.py 1 2 16 是有用的

generic.py
     585 行至 593 行：              将 SCIP 模型输出成 lp 文件，调用 Gurobi 求解，
                                    将结果写入 sol 文件，SCIP 模型读取 sol 文件。
                                    这是为了加快求解速度，keccak_attack.py 中关于
                                    keccak[r=18, c=32, h=16] 的攻击是 SCIP 给出的
                                    解，与 Gurobi 给出的解略有不同，但复杂度是一致的

imgs/：
Keccak_3r_8l_80h.pdf：
    1. 第 0 列最下面是初始为 0 的 capacity
    2. 最后一列是哈希值经过 S 盒的逆变换之后的节点，下面的节点并不关心
    3. 除 1 和 2 外其余节点均未省略

Keccak_3r_8l_80h_tidy.pdf：
    1. 在 Keccak_3r_8l_80h.pdf 的基础之上还删除了第 0，7，8，9 列最上方的节点
    2. 第 1 列最上方的节点并未省略，只是模型的解不涉及这些节点，没有画出来
