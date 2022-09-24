
import numpy as np



def encode_in_6_dimensions(on_target_seq, off_target_seq):
    bases_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
    positions_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}

    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "-"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "-"*(tlen-len(off_target_seq)) + off_target_seq
    # 碱基，indel和空的 编码，转换
    on_target_seq_code = np.array([bases_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([bases_dict[base] for base in list(off_target_seq)])

    pair_dim5_codes = []
    for i in range(len(on_target_seq)):
        bases_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i]) # 前四维
        dir_code = np.zeros(1) # 表示两个碱基在on off target上的可能位置，1，0，-1

        if positions_dict[on_target_seq[i]] == positions_dict[off_target_seq[i]]:
            bases_code = bases_code*-1
        elif positions_dict[on_target_seq[i]] < positions_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif positions_dict[on_target_seq[i]] > positions_dict[off_target_seq[i]]:
            dir_code[0] = -1
        else:
            raise Exception("Invalid seq!", on_target_seq, off_target_seq)

        pair_dim5_codes.append(np.concatenate((bases_code, dir_code)))
    pair_dim5_codes = np.array(pair_dim5_codes)

    isPAM = np.zeros((24,1))
    isPAM[-3:, :] = 1
    pair_code = np.concatenate((pair_dim5_codes, isPAM), axis=1)
    return pair_code

def encode_by_base_pair_vocabulary(on_target_seq, off_target_seq):
    BASE_PAIR_VOCABULARY_v1 = {
        "AA":0,  "TT":1,  "GG":2,  "CC":3,
        "AT":4,  "AG":5,  "AC":6,  "TG":7,  "TC":8,  "GC":9,
        "TA":10, "GA":11, "CA":12, "GT":13, "CT":14, "CG":15,
        "A_":16, "T_":17, "G_":18, "C_":19,
        "_A":20, "_T":21, "_G":22, "_C":23,
        "AAP":24,  "TTP":25,  "GGP":26,  "CCP":27,
        "ATP":28,  "AGP":29,  "ACP":30,  "TGP":31,  "TCP":32,  "GCP":33,
        "TAP":34, "GAP":35, "CAP":36, "GTP":37, "CTP":38, "CGP":39,
        "A_P":40, "T_P":41, "G_P":42, "C_P":43,
        "_AP":44, "_TP":45, "_GP":46, "_CP":47,
        "__":48, "__P":49
    }
    BASE_PAIR_VOCABULARY_v2 = {
        "AA": 0,    "TT": 1,    "GG": 2,    "CC": 3,
        "AAP":4,    "TTP":5,    "GGP":6,    "CCP":7,
        "AT": 8,    "AG": 9,    "AC": 10,   "TG": 11,   "TC": 12,   "GC": 13,
        "TA": 14,   "GA": 15,   "CA": 16,   "GT": 17,   "CT": 18,   "CG": 19,
        "ATP":20,   "AGP":21,   "ACP":22,   "TGP":23,   "TCP":24,   "GCP":25,
        "TAP":26,   "GAP":27,   "CAP":28,   "GTP":29,   "CTP":30,   "CGP":31,
        "A_": 32,   "T_": 33,   "G_": 34,   "C_": 35,
        "_A": 36,   "_T": 37,   "_G": 38,   "_C": 39,
        "A_P":40,   "T_P":41,   "G_P":42,   "C_P":43,
        "_AP":44,   "_TP":45,   "_GP":46,   "_CP":47,
        "__": 48,   "__P":49
    }
    BASE_PAIR_VOCABULARY_v3 = {
        "AA": 0,    "TT": 1,    "GG": 2,    "CC": 3,
        "AT": 4,    "AG": 5,    "AC": 6,   "TG": 7,   "TC": 8,   "GC": 9,
        "TA": 10,   "GA": 11,   "CA": 12,   "GT": 13,   "CT": 14,   "CG": 15,
        "A_": 16,   "T_": 17,   "G_": 18,   "C_": 19,
        "_A": 20,   "_T": 21,   "_G": 22,   "_C": 23,
        "__": 24
    }
    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "_"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "_"*(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq = on_target_seq.replace("-", "_")
    off_target_seq = off_target_seq.replace("-", "_")

    pair_vector = list()
    for i in range(tlen):
        base_pair = on_target_seq[i]+off_target_seq[i]
        # if i > 20:
        #     base_pair += "P"
        pair_vector.append(BASE_PAIR_VOCABULARY_v3[base_pair])
    pair_vector = np.array(pair_vector)
    return pair_vector

def encode_by_base_vocabulary(seq):
    BASE_VOCABULARY_v1 = {
        "A": 50, "T": 51, "G": 52, "C": 53, "_": 54
    }
    BASE_VOCABULARY_v3 = {
        "A": 25, "T": 26, "G": 27, "C": 28, "_": 29
    }
    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    seq = "_"*(tlen-len(seq)) + seq
    seq = seq.replace("-", "_")

    seq_vector = list()
    for i in range(tlen):
        base = seq[i]
        seq_vector.append(BASE_VOCABULARY_v3[base])
    seq_vector = np.array(seq_vector)
    return seq_vector

def encode_by_one_hot(on_target_seq, off_target_seq):
    bases_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 1]}

    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "-"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "-"*(tlen-len(off_target_seq)) + off_target_seq

    pair_dim11_codes = []
    for i in range(len(on_target_seq)):
        base_code = bases_dict[on_target_seq[i]]+bases_dict[off_target_seq[i]]
        if i in [21, 22, 23]:
            base_code.append(1)
        elif 0 <= i <= 20:
            base_code.append(0)
        else:
            raise Exception("base code error")
        pair_dim11_codes.append(base_code)
    pair_dim11_codes = np.array(pair_dim11_codes)
    return pair_dim11_codes

def encode_by_crispr_net_method(on_target_seq, off_target_seq):
    bases_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                          'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
                          '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
    direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}

    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "-"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "-"*(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq_code = np.array([bases_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([bases_dict[base] for base in list(off_target_seq)])

    on_off_dim7_codes = []
    for i in range(len(on_target_seq)):
        diff_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i]) # 前5维or
        on_b = on_target_seq[i]
        off_b = off_target_seq[i]
        dir_code = np.zeros(2)
        if on_b == "-" or off_b == "-" or direction_dict[on_b] == direction_dict[off_b]:
            pass
        else:
            if direction_dict[on_b] > direction_dict[off_b]:
                dir_code[0] = 1
            else:
                dir_code[1] = 1
        on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim7_codes = np.array(on_off_dim7_codes)
    return on_off_dim7_codes


def encode_by_crispr_net_method_with_isPAM(on_target_seq, off_target_seq):
    bases_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                          'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
                          '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
    direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}

    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "-"*(tlen-len(on_target_seq)) + on_target_seq
    off_target_seq = "-"*(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq_code = np.array([bases_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([bases_dict[base] for base in list(off_target_seq)])

    on_off_dim7_codes = []
    for i in range(len(on_target_seq)):
        diff_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i]) # 前5维or
        on_b = on_target_seq[i]
        off_b = off_target_seq[i]
        dir_code = np.zeros(2)
        if on_b == "-" or off_b == "-" or direction_dict[on_b] == direction_dict[off_b]:
            pass
        else:
            if direction_dict[on_b] > direction_dict[off_b]:
                dir_code[0] = 1
            else:
                dir_code[1] = 1
        on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim7_codes = np.array(on_off_dim7_codes)
    isPAM = np.zeros((24,1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim7_codes, isPAM), axis=1)
    return on_off_code

def encode_by_crispr_ip_method(on_target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
    pos_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}

    tlen = 24
    on_target_seq = "-" *(tlen-len(on_target_seq)) + on_target_seq
    
    off_target_seq = "-" *(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq_code = np.array([encoded_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(on_target_seq)):
        diff_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[on_target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code*-1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[on_target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[on_target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", on_target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24,1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code

def encode_by_crispr_ip_method_without_minus(on_target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
    pos_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}

    tlen = 24
    on_target_seq = "-" *(tlen-len(on_target_seq)) + on_target_seq
    
    off_target_seq = "-" *(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq_code = np.array([encoded_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(on_target_seq)):
        diff_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[on_target_seq[i]] == pos_dict[off_target_seq[i]]:
            # remove a command against crispr_ip: diff_code = diff_code*-1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[on_target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[on_target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", on_target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24,1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code

def encode_by_crispr_ip_method_without_isPAM(on_target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
    pos_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}

    tlen = 24
    on_target_seq = "-" *(tlen-len(on_target_seq)) + on_target_seq
    
    off_target_seq = "-" *(tlen-len(off_target_seq)) + off_target_seq
    on_target_seq_code = np.array([encoded_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(on_target_seq)):
        diff_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[on_target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code*-1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[on_target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[on_target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", on_target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    return on_off_dim6_codes

def encode_by_r_crispr_method(on_target_seq, off_target_seq):
    on_target_seq = "-" * (24-len(on_target_seq)) + on_target_seq
    off_target_seq = "-" * (24-len(off_target_seq)) + off_target_seq
    base_dict = {'A':[1,0,0,0,0],
                'C':[0,1,0,0,0],
                'G':[0,0,1,0,0],
                'T':[0,0,0,1,0],
                '_':[0,0,0,0,1],
                '-':[0,0,0,0,0]}
    direct = {'A':1, 'C':2, 'G':3, 'T':4, '_':5}
    # on_seq encoder
    seq_on = []
    on_base_list = list(on_target_seq)
    for i in range(len(on_base_list)):
        if on_base_list[i] == "N":
            on_base_list[i] = list(off_target_seq)[i]
        seq_on.append(base_dict[on_base_list[i]])
    on_seq = np.array(seq_on)

    # off_seq_encoder
    seq_off = []
    off_base_list = list(off_target_seq)
    for i in range(len(off_base_list)):
        seq_off.append(base_dict[off_base_list[i]])
    off_seq = np.array(seq_off)

    # pair_encoder
    pair_encode = []

    for i in range(len(on_base_list)):
        base_code = np.bitwise_or(on_seq[i],off_seq[i])
        direct_code = np.zeros(2)
        if on_base_list[i] == "N":
            on_base_list[i] = off_base_list[i]
        if on_base_list[i] == "-" or off_base_list[i] == "-" or direct[on_base_list[i]] == direct[off_base_list[i]]:
            pass
        else:
            if direct[on_base_list[i]] < direct[off_base_list[i]]:
                direct_code[0] = 1
            else:
                direct_code[1] = 1
        pair_encode.append(np.concatenate((base_code,direct_code)))
    pair_code = np.array(pair_encode)
    return pair_code

# if __name__ == "__main__":
#     on_target_seq  = "GAGTC_CGAGCAGAAGAAGAAAGG"
#     off_target_seq = "GAGTCGCGAGTAGAAG_AGAACGG"
#     pair_code = encode_in_6_dimensions(on_target_seq, off_target_seq)
#     print(pair_code)


