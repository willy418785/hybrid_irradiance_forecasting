from pyimagesearch import time_embedding

time_embedding_list = [None, "t2v", "learnable", 'sin_cos']


class TEFac:
    def __init__(self):
        pass

    @staticmethod
    def get_te_mode(command):
        if type(command) is int:
            if command < 0 or command >= len(time_embedding_list):
                command = None
            else:
                command = time_embedding_list[command]
        else:
            if command not in time_embedding_list:
                command = None
        return command

    @staticmethod
    def new_te_module(command, tar_dim: int, seq_structure: tuple):
        # command is expected to be an element of time_embedding_list or an integer for indexing bypass_list

        input_width, shift, label_width = seq_structure
        command = TEFac.get_te_mode(command)

        if command is None:
            te = None
        elif command == "t2v":
            te = time_embedding.Time2Vec(output_dims=tar_dim,
                                         input_len=input_width,
                                         shift_len=shift,
                                         label_len=label_width)
        elif command == "learnable":
            te = time_embedding.TimeEmbedding(output_dims=tar_dim,
                                              input_len=input_width,
                                              shift_len=shift,
                                              label_len=label_width)
        elif command == "sin_cos":
            te = time_embedding.SinCosTimeEncoding(output_dims=tar_dim,
                                                   input_len=input_width,
                                                   shift_len=shift,
                                                   label_len=label_width)
        else:
            te = None

        return te
