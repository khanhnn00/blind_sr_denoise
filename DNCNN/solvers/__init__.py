from .ESolver import ESolver

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = ESolver(opt)
    else:
        raise NotImplementedError

    return solver