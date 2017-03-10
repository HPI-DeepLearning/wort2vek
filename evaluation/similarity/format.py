import sys

input_file = sys.argv[1]
output = sys.argv[2]

with open(input_file) as f_in, open(output, 'w') as f_out:
    lines = ['\t'.join(l.split(':')[:-2]) + '\n' for l in f_in]
    f_out.writelines(lines)
