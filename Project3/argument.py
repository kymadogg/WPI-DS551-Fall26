def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    '''
    Things to parameterize:
    ---

    '''
    parser.add_argument('--trials', type=int, default=50_000, help='number of games to play for training')
    parser.add_argument('--name', type=str, default="default", help='name of the log file and model weights')
    return parser
