import copy
# import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
# from Arguments import Arguments
import numpy as np

# args = Arguments()

n_qubits = 8
n_layers = 4

def gen_arch(change_code, base_code):  # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    # n_qubits = base_code[0]
    if n_qubits == 7:
        arch_code = [2, 3, 4, 5, 6, 7, 1] * base_code[1]  # qubits * layers
    else:
        # arch_code = [2, 3, 4, 1] * base_code[1]
        arch_code = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1] * base_code[1]  # for MNIST 10
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code


def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:, 0] - 1
        change_code = change_code.reshape(-1, length)
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1, 0)
            j += 1
    return single_dict


def translator(n_qubits, n_layers):
    updated_design = {}

    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):

        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i) + '0'] = 'RZ'
            updated_design['rot' + str(layer) + str(i) + '1'] = 'RY'
            updated_design['rot' + str(layer) + str(i) + '2'] = 'RZ'
        for j in range(n_qubits):
            updated_design['enta' + str(layer) + str(j) + '0'] = ('CNOT', [j, 0])
            updated_design['enta' + str(layer) + str(j) + '1'] = ('CNOT', [j, 1])
            updated_design['enta' + str(layer) + str(j) + '2'] = ('CNOT', [j, 2])
            updated_design['enta' + str(layer) + str(j) + '3'] = ('CNOT', [j, 3])
            updated_design['enta' + str(layer) + str(j) + '4'] = ('CNOT', [j, 4])
            updated_design['enta' + str(layer) + str(j) + '5'] = ('CNOT', [j, 5])
            updated_design['enta' + str(layer) + str(j) + '6'] = ('CNOT', [j, 6])
            updated_design['enta' + str(layer) + str(j) + '7'] = ('CNOT', [j, 7])

        # # categories of single-qubit parametric gates
        # for i in range(n_qubits):
        #     updated_design['rot' + str(layer) + str(i)] = 'U3'
        # # categories and positions of entangled gates
        # for j in range(n_qubits):
        #     if net[j + layer * n_qubits] > 0:
        #         updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits] - 1])
        #     else:
        #         updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits]) - 1, j])

    # updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design


def cir_to_matrix(x, y, arch_code):
    qubits = arch_code[0]
    layers = arch_code[1]
    entangle = gen_arch(y, arch_code)
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1, 0)
    single = np.ones((qubits, 2 * layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers + 1)], entangle, axis=1)
    return arch.transpose(1, 0)


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        # self.design = design
        self.n_wires = self.args.n_qubits
        self.n_layers = self.args.n_layers
        # self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])
        # self.uploading = [tq.GeneralEncoder(encoder_op_list_name_dict['{}x4_ryzxy'.format(i)]) for i in range(4)]
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(self.n_wires)]

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []

        for i in range(self.n_layers):
            for j in range(self.args.n_qubits):
                self.q_params_rot.append(pi * torch.rand(1))    # RZ
                self.q_params_rot.append(pi * torch.rand(1))    # RY
                self.q_params_rot.append(pi * torch.rand(1))    # RZ
            # self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3))  # each CU3 gate needs 3 parameters

        for layer in range(self.n_layers):
            for q in range(self.n_wires):
                # 'trainable' option
                # if self.design['change_qubit'] is None:
                #     rot_trainable = True
                #     enta_trainable = True
                # elif q == self.design['change_qubit']:
                #     rot_trainable = True
                #     enta_trainable = True
                # else:
                #     rot_trainable = False
                #     enta_trainable = False
                # single-qubit parametric gates
                rot_trainable = True
                self.rots.append(tq.RZ(has_params=True, trainable=rot_trainable,
                                        init_params=self.q_params_rot[24 * layer + 3 * q + 0]))
                self.rots.append(tq.RY(has_params=True, trainable=rot_trainable,
                                        init_params=self.q_params_rot[24 * layer + 3 * q + 1]))
                self.rots.append(tq.RZ(has_params=True, trainable=rot_trainable,
                                        init_params=self.q_params_rot[24 * layer + 3 * q + 2]))
                for j in range(self.n_wires):
                    if q == j:
                        self.entas.append(tq.I(has_params=False, trainable=False, init_params=None, wires=q))
                    else:
                        self.entas.append(tq.CNOT(has_params=False, trainable=False, init_params=None, wires=[q, j]))

                # entangled gates
                # if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                #     self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                #                              init_params=self.q_params_enta[q][layer]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            # {"input_idx": [2], "func": "rx", "wires": [qubit]},
            # {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x, hardwts):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)  # 'down_sample_kernel_size' = 6
        x = x.view(bsz, 8, 2)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        for layer in range(self.args.n_layers):
            for j in range(self.n_wires):
                if layer==0:
                    self.uploading[j](qdev, x[:, j])

                for k in range(self.n_wires+1):
                    if hardwts[j][k]==1:
                        if k == 8:
                            self.rots[(j + layer * self.n_wires) * 3](qdev, wires=j)
                            self.rots[(j + layer * self.n_wires) * 3 + 1](qdev, wires=j)
                            self.rots[(j + layer * self.n_wires) * 3 + 2](qdev, wires=j)
                        elif k == j:
                            self.entas[(j + layer * self.n_wires) * 8 + k](qdev, wires=j)
                        else:
                            self.entas[(j + layer * self.n_wires) * 8 + k](qdev, wires=[j, k])

        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments):
        super(QNet, self).__init__()
        self.args = arguments
        # self.design = design
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self.args.n_qubits, self.args.n_qubits+1)    # sampling requires a 8x9 matrix
        )
        self.QuantumLayer = TQLayer(self.args)

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_alphas(self):
        return [self.arch_parameters]
    #
    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self.arch_parameters, dim=-1).cpu()
            )


    def forward(self, x_image):
        while True:
            while True:
                gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
                logits = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if torch.max(torch.argmax(hardwts,dim=1)) == 8:     # To make sure at least one rot gate is sampled
                    break
            if (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        exp_val = self.QuantumLayer(x_image, hardwts)
        output = F.log_softmax(exp_val, dim=1)
        return output
