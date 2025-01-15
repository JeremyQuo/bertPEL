from collections import OrderedDict

def read_fasta_to_dic(filename):
    """
    function used to parser small fasta
    still effective for genome level file
    """
    fa_dic = OrderedDict()

    with open(filename, "r") as f:
        for n, line in enumerate(f.readlines()):
            if line.startswith(">"):
                if n > 0:
                    fa_dic[short_name] = "".join(seq_l)  # store previous one

                full_name = line.strip().replace(">", "")
                short_name_list = full_name.split(";")
                status = False
                for item in short_name_list:
                    if 'locus' in item:
                        short_name=item.split("=")[1]
                        status = True
                if not status:
                    print(1)
                seq_l = []
            else:  # collect the seq lines
                if len(line) > 8:  # min for fasta file is usually larger than 8
                    seq_line1 = line.strip()
                    seq_l.append(seq_line1)

        fa_dic[short_name] = "".join(seq_l)  # store the last one
    return fa_dic

def save_fasta_dict(fasta_dict,path):
    f=open(path,'w+')
    for key,value in fasta_dict.items():
        f.write('>'+key+'\n')
        line = len(value) // 80 + 1
        for i in range(0, line):
            f.write(value[i*80:(i+1)*80]+'\n')
    f.close()

sequence_dic = read_fasta_to_dic("data/pa.ffn")
save_fasta_dict(sequence_dic,"data/pao_mrna.fasta")
print(1)