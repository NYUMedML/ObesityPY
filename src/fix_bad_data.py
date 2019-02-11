def fix_data_notes(baddatafile, gooddatafile):
    f = open(baddatafile, 'r')
    out = open(gooddatafile, 'w')
    true_line = ''
    for l in f.readlines():
        true_line = true_line +' '+ l.strip('\n')
        if true_line.endswith('</SOAP-ENV:Envelope>'):
            out.write(true_line+'\n')
            out.flush()
            true_line = ''
    out.close()
    f.close()

