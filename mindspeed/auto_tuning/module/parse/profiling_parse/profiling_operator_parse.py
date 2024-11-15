from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_constant import OperatorDetails


class AnalyseOperatorMsg:
    """ Analyse operator message. """

    def __init__(self, operator_details):
        self._operator_details = operator_details

    def analyse_embedding(self, start_idx, end_idx):
        return self._analyse_operators(start_idx, end_idx)

    def analyse_forward(self, start_idx, end_idx):
        return self._analyse_operators(start_idx, end_idx)

    def analyse_loss(self, start_idx, end_idx):
        return self._analyse_operators(start_idx, end_idx)

    def analyse_backward(self, start_idx, end_idx):
        return self._analyse_operators(start_idx, end_idx)

    def analyse_optimizer(self, start_idx, end_idx):
        return self._analyse_operators(start_idx, end_idx)

    def _analyse_operators(self, start_idx, end_idx):
        details_list = []
        for i in range(start_idx, end_idx):
            detail = self._operator_details[i]
            op_detail = OperatorDetails(
                name=detail['Name'],
                type_=detail['Type'],
                input_shapes=detail['Input Shapes'],
                output_shapes=detail['Output Shapes'],
                duration_us=detail['Duration(us)'],
                wait_time_us=detail['Wait Time(us)'],
                accelerator_core=detail['Accelerator Core']
            )
            details_list.append(op_detail)
        return details_list
