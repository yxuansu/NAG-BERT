import ose
import argparse
from evaluation import write_results, get_rouge_scores

def parse_config():
    parser = argparse.ArgumentParser()
    # model parameters path
    parser.add_argument('--inference_path', type=str)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    # path to store generated summary
    summary_dir = args.summary_dir
    try:
        os.stat(summary_dir)
    except:
        os.mkdir(summary_dir)

    # path to store reference summary
    model_dir = args.model_dir
    try:
        os.stat(model_dir)
    except:
        os.mkdir(model_dir)
    reference_list, hypothesis_list = [], []
    with open(args.inference_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split('\t')
            one_reference, one_hypothesis = content_list[1].strip(), content_list[2].strip()
            reference_list.append(one_reference)
            hypothesis_list.append(one_hypothesis)

    write_results(args.summary_dir, hypothesis_list, mode = r'decoded')
    write_results(args.model_dir, reference_list, mode = r'reference')
    rogue_1_score, rogue_2_score, rogue_l_score = get_rouge_scores(summary_dir, model_dir)
    print ('rogue 1 is %.5f, rogue 2 is %.5f, rogue l is %.5f' % (rogue_1_score, rogue_2_score, rogue_l_score))

