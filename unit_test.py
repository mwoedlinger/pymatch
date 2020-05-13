def grad_test():
    a = Tensor(2, name='a')
    b = Tensor(3, name='b')
    c = Tensor(4, name='c')

    x = (a + c*(a+b))**2

    x.backward()

    assert x.val == 484
    assert a.grad == 220

    return 'passed'

if __name__ == '__main__':
    print('grad test: {}'.format(str(grad_test)))