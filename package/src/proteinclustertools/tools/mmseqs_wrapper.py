import argparse
import subprocess
import os

def RunMMseqs(fasta, out_dir='', prefix=''):
    # Run mmseqs
    outfile=prefix+'_mmseqs_search.tsv'
    mmseqs_args = ['mmseqs', 'easy-search', fasta, fasta, os.path.join(out_dir,outfile), 'tmp']
    opt_args=['--max-seqs', '1000', '--format-mode', '4', '--format-output', 'query,target,bits']

    mmseqs_args.extend(opt_args)
    subprocess.run(mmseqs_args)
    return {"mmseqs_result":outfile, "mmseqs_command":' '.join(mmseqs_args)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', type=str, required=True)
    parser.add_argument('out_label', type=str, required=True, help='Output label for the mmseqs search table')
    args = parser.parse_args()

    # Run mmseqs
    RunMMseqs(args.fasta, args.out_label)