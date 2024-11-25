class ProfileTimeInfo():
    def __init__(self):
        # Profile source information
        self.total_comm_time = 0
        self.wait_comm_time = 0
        self.overlap_comm_time = 0


class TpProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(TpProfileTimeInfo, self).__init__()
        # Total time when communication hiding is not performed
        self.fixedtotal_tp_time = 0
        self.fixedwait_tp_time = 0


class Mc2ProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(Mc2ProfileTimeInfo, self).__init__()
        self.matmul_compute_time = 0


class CpProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(CpProfileTimeInfo, self).__init__()
        # Total time when communication hiding is not performed
        self.attn_cp_time = 0
        self.attn_cpbw_time = 0
        self.vector_cp_time = 0


class DpProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(DpProfileTimeInfo, self).__init__()
        # Total time when communication hiding is not performed
        self.overlap_grad_reduce = 0
        self.overlap_param_gather = 0
        self.overlap = 0
        self.total_mlpzero_time = 0
        self.total_otherzero_time = 0
        self.mlp_ag_time = 0
        self.mlp_rs_time = 0
        self.other_ag_time = 0
        self.other_rs_time = 0


class EpProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(EpProfileTimeInfo, self).__init__()
        self.min_time = 0


class PpProfileTimeInfo(ProfileTimeInfo):
    def __init__(self):
        super(PpProfileTimeInfo, self).__init__()
        # Total time when communication hiding is not performed
        self.each_pp_time = 0
        self.bubble_end_time = 0
        self.bubble_start_time = 0


class TotalProfileTimeInfo():
    def __init__(self):
        # Profile source information
        self.tp_profile_time_info = TpProfileTimeInfo()
        self.cp_profile_time_info = CpProfileTimeInfo()
        self.dp_profile_time_info = DpProfileTimeInfo()
        self.ep_profile_time_info = EpProfileTimeInfo()
        self.pp_profile_time_info = PpProfileTimeInfo()
        self.mc2_profile_time_info = Mc2ProfileTimeInfo()
