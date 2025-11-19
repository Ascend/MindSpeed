class Fp8Padding:
    def __init__(self, num_local_experts):
        pass

    def __call__(self, inp, m_split):
        return inp, m_split


class Fp8Unpadding:
    def __init__(self, num_local_experts):
        pass

    def __call__(self, inp, m_split):
        return inp
