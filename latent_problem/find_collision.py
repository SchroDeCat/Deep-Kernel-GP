from target_funcs import *
from collision_problem import Collision_Problem
from argparse import ArgumentParser


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--n', default=100, help='training points')
    parser.add_argument('--i', default=10000, help='Iterations')
    args = parser.parse_args()

    test = Collision_Problem(lambda x:target_func2(x), train_num=int(args.n))
    if test.find_collision(iter_num=int(args.i)):
        print("Collision Found!")
    else:
        test.plot_collision() # plot result at the end if collision not found