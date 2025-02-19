from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    main_fn = execute_function(args.method, args.mode)

    args.save_dir = f"{args.save_dir}/{args.concepts.replace(' ', '-')}/{args.method}"
    
    main_fn(args)
