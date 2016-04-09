from pixelqueer import *

def run(args):
    if args.action == "learn":
        male_eigenfaces, female_eigenfaces, mtf_matrix, ftm_matrix = learn(args.source, args.maxEigenfaces, args.maxImages)
        learn_file = open(args.brain, "wb")
        pickle.dump(male_eigenfaces, learn_file)
        pickle.dump(female_eigenfaces, learn_file)
        pickle.dump(mtf_matrix, learn_file)
        pickle.dump(ftm_matrix, learn_file)

    elif args.action == "run":
        learn_file = open(args.brain, "rb")
        male_eigenfaces = pickle.load(learn_file)
        female_eigenfaces = pickle.load(learn_file)
        mtf_matrix = pickle.load(learn_file)
        ftm_matrix = pickle.load(learn_file)

        original_image = args.image
        altered_image = None
        print args.direction
        if args.direction == "mtf":
            print "MTF Mode"
            altered_image = alter_image(male_eigenfaces, female_eigenfaces, mtf_matrix, original_image)
        elif args.direction == "ftm":
            print "FTM Mode"
            altered_image = alter_image(female_eigenfaces, male_eigenfaces, ftm_matrix, original_image)
        else:
            print "Non-binary gender not yet supported :("
            return

        if args.out is not None:
            altered_image_to_save = Image.fromarray(((altered_image+1.0)/2.0 * 255).astype(np.uint8))
            altered_image_to_save.save(args.out)
        else:
            plot_image(original_image, altered_image)


print "Hello world!"

parser = argparse.ArgumentParser(description="Genderbending!")
parser.add_argument("action", choices=["learn", "run"])
parser.add_argument("brain", help="Destination file for learned data")
parser.add_argument("--source", help="Source image folder containing Male and Female subfolders")
parser.add_argument("--direction", choices=["mtf", "ftm"], help="Direction to bend gender")
parser.add_argument("--maxEigenfaces", help="Maximum number of eigenfaces to use", default=max, type=int)
parser.add_argument("--maxImages", help="Maximum number of images to use (per gender)", default=max, type=int)
parser.add_argument("--image", help="Image to load (run mode only)")
parser.add_argument("--out", help="Output file path")

args = parser.parse_args(sys.argv[1:])
run(args)

