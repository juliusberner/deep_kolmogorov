from deep_kolmogorov.trainer import main, get_args

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    main(vars(args))
