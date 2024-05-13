from ascendspeed.op_builder import LcalOpBuilder


class CoCOperations:
    ascendspeed_ops = LcalOpBuilder().load()

    def matmul_all_reduce(self, input1, input2, output, bias=None):
        device = input1.device.index
        CoCOperations.ascendspeed_ops.matmul_all_reduce(input1, input2, bias, output, device)
        return output

    def all_gather_matmul(self, input1, input2, output, bias=None):
        device = input1.device.index
        CoCOperations.ascendspeed_ops.all_gather_matmul(input1, input2, bias, output, device)
        return output

    def all_gather_matmul_v2(self, input1, input2, output, comm_output, bias=None):
        device = input1.device.index
        CoCOperations.ascendspeed_ops.all_gather_matmul_v2(input1, input2, bias, output, comm_output, device)
        return output, comm_output

    def matmul_reduce_scatter(self, input1, input2, output, bias=None):
        device = input1.device.index
        CoCOperations.ascendspeed_ops.matmul_reduce_scatter(input1, input2, bias, output, device)
        return output

    def pure_matmul(self, input1, input2, output, bias=None):
        device = input1.device.index
        CoCOperations.ascendspeed_ops.pure_matmul(input1, input2, bias, output, device)
        return output

coc_ops = CoCOperations()
