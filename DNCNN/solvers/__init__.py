from .ESolver import ESolver
from .CBDSolver import CBDSolver

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = ESolver(opt)
    else:
        raise NotImplementedError

    return solver

def create_cbd_solver(opt):
    if opt['mode'] == 'sr':
        solver = CBDSolver(opt)
    else:
        raise NotImplementedError

    return solver