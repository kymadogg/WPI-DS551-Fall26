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
    EPSILON_START = 0.9
    EPSILON_END = 0.025
    NUM_GAMES = 1000000
    EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_GAMES
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    GAMMA = 0.99
    REPLAY_MEMORY_SIZE = 10000
    START_REPLAY_MEMORY_SIZE = 5000
    STEPS_FOR_UPDATE = 5000
    '''

    parser.add_argument('--')
    return parser
