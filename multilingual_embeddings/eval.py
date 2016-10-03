"""
eval.py

Visualize and evaluate the quality of the multilingual embeddings.
"""
import matplotlib.pyplot as plt
import os

EVAL_DIR = "data/eval"
PAIRS = ["de-en", "en-es", "en-fr"]
NUM_BUCKETS = 3


def main():
    """
    Main function, generate all graphs and necessary evaluation data.
    """
    graph_dict = {'cos': {}, 'euc': {}}

    for type in ['cos', 'euc']:
        for p in PAIRS:
            x, y = [], []
            fill, count = False, 0
            for i in range(NUM_BUCKETS - 1):
                with open(os.path.join(EVAL_DIR, str(i) + "_" + p + "." + type), 'r') as f:
                    for line in f:
                        if not fill:
                            x.append(int(line.split('\t')[0]))
                            y.append(float(line.split('\t')[1]))
                        else:
                            y[count] += float(line.split('\t')[1])
                        count += 1
                fill, count = True, 0
            x = map(lambda z: z * 10, x)
            y = map(lambda z: z / float(NUM_BUCKETS), y)
            graph_dict[type][p] = (x, y)

    # PLOT
    for type in ['cos', 'euc']:
        data = graph_dict[type]
        print data[PAIRS[0]][0]
        print data[PAIRS[0]][1]
        print ""
        plt.plot(data[PAIRS[0]][0], data[PAIRS[0]][1], 'ro-', label=PAIRS[0])
        plt.plot(data[PAIRS[1]][0], data[PAIRS[1]][1], 'bo-', label=PAIRS[1])
        plt.plot(data[PAIRS[2]][0], data[PAIRS[2]][1], 'go-', label=PAIRS[2])
        plt.xlabel('Batch Number')
        if type == 'cos':
            plt.ylabel('Cosine Distance')
            plt.title('Average Cosine Distance Between Embeddings of Different Language Pairs')
            plt.ylim([-.2, .8])
        else:
            plt.ylabel('Euclidean Distance')
            plt.title('Average Euclidean Distance Between Embeddings of Different Language Pairs')
            plt.ylim([-2, 150])

        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()