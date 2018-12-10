class Calculation(object):
    def get_number_of_parameters(self, K, F, D_in):
        """get cnn parameters

        function untuk menghitung jumlah parameter cmm

        Arguments:
            K {int} -- jumlah filter
            F {int} -- kernel size
            D_in {int} -- dept dari layer sebelumnya
        """
        return (K * F * F * D_in) + K

    def get_shape_from_convolutional(self, W_in, F, P=0, S=1):
        """get shape from convolutional layer

        function untuk menghitung shape yang dihasilkan convolutional layer

        Arguments:
            W_in {int} -- Panjang/tinggi dari layer sebelumnya
            F {int} -- kernel size

        Keyword Arguments:
            P {int} -- padding (default: {0})
            S {stride} -- stride (default: {1})
        """
        return (W_in - F + (2 * P)) / S + 1


if __name__ == '__main__':
    calc = Calculation()
    conv1 = calc.get_shape_from_convolutional(W_in=32, F=3, S=1, P=1)
    print(conv1)
    conv2 = calc.get_shape_from_convolutional(W_in=32, F=3, S=1, P=1)
    print(conv2)
    conv3 = calc.get_shape_from_convolutional(W_in=16, F=3, S=1, P=1)
    print(conv3)