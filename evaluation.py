from pyrouge import Rouge155
def get_one_line_result(line):
    return float(line.strip().split()[3]) * 100
    
def get_rouge_scores(summary_dir, model_dir):
    r = Rouge155()
    r.system_dir = summary_dir
    r.model_dir = model_dir
    r.system_filename_pattern = r'(\d+)_decoded.txt'
    r.model_filename_pattern = r'#ID#_reference.txt'
    output = r.convert_and_evaluate()
    output_list = output.split('\n')
    rogue_1_score = get_one_line_result(output_list[3])
    rogue_2_score = get_one_line_result(output_list[7])
    rogue_l_score = get_one_line_result(output_list[19])
    return rogue_1_score, rogue_2_score, rogue_l_score

def write_results(dir, text_list, mode):
    if mode == 'decoded':
        pass
    elif mode == 'reference':
        pass
    else:
        raise Exception('Wrong Result Mode!!!')

    data_num = len(text_list)
    for i in range(data_num):
        one_text = text_list[i]
        one_out_f = dir + '/' + str(i) + '_' + mode + '.txt'
        with open(one_out_f, 'w', encoding = 'utf8') as o:
            o.writelines(one_text)
