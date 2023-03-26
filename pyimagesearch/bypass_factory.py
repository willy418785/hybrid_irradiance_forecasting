from pyimagesearch import model_AR, model_baseline

bypass_list = [None, "LR", "MA"]


class BypassFac:
    def __init__(self):
        pass

    @staticmethod
    def get_bypass_mode(command):
        if type(command) is int:
            if command < 0 or command >= len(bypass_list):
                command = None
            else:
                command = bypass_list[command]
        else:
            if command not in bypass_list:
                command = None
        return command

    @staticmethod
    def new_bypass_module(command, out_width: int, order=None, in_dim=None, window_len=None, is_within_day=None,
                          samples_per_day=None):
        # command is expected to be an element of bypass_list or an integer for indexing bypass_list

        command = BypassFac.get_bypass_mode(command)
        if command is None:
            bypass = None
        elif command == "LR":
            assert order is not None and type(order) is int
            assert in_dim is not None and type(in_dim) is int
            bypass = model_AR.TemporalChannelIndependentLR(order, out_width, in_dim)
        elif command == "MA":
            assert window_len is not None and type(window_len) is int
            assert is_within_day is not None and type(is_within_day) is bool
            assert samples_per_day is not None and type(samples_per_day) is int
            bypass = model_baseline.MA(window_len=window_len,
                                       is_within_day=is_within_day,
                                       samples_per_day=samples_per_day,
                                       output_width=out_width)
        else:
            bypass = None
        return bypass


