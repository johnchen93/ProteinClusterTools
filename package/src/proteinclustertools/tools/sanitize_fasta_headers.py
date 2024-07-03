from Bio import SeqIO
import pandas as pd
import os

def sanitize_fasta_headers(fasta_file, out_dir='', out_prefix=''):
    
    cleaned_filename=out_prefix+'_cleaned.fasta'
    cleaned_file = open(os.path.join(out_dir,cleaned_filename), 'w')
    mapping_filename=out_prefix+'_header_map.txt'

    i=0
    mapping={}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        id=record.id
        print(f'>{i}\n{record.seq}', file=cleaned_file)
        mapping[i]=id
        i+=1
        
    cleaned_file.close()
    mapping=pd.DataFrame.from_dict(mapping, orient='index').reset_index()
    mapping.columns=['id','header']
    mapping.to_csv(os.path.join(out_dir,mapping_filename), index=False)

    return {'cleaned_fasta': cleaned_filename, 'header_map': mapping_filename}