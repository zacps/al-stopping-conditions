import argparse
import dill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")

    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        active_learner = dill.load(f)
        print(f"Progress {active_learner.learner.X_training.shape[0]}/1000")


if __name__ == "__main__":
    main()
