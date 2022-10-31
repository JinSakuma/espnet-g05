import xml.etree.ElementTree as ET
import os
import argparse


XMLROOT = '/mnt/aoni01/db/CSJ/CSJ2004/XML'


def run(args):
    input_path = args.input
    output_path1 = args.output1
    output_path2 = args.output2
    
    file_id = input_path.split('/')[-1] 
    name = '{}.xml'.format(file_id)
    
    xml_path = os.path.join(XMLROOT, name)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    talk_id = root.attrib['TalkID']
    text = ''
    output = ''
    ostart = ''
    clause_id = -1
    pend = -1
    for ipu in root:
        if ipu.tag != 'IPU':
            continue

        for luws in ipu:                
            for suws in luws:
                if 'ClauseUnitID' in suws.attrib:
                    if int(clause_id) >= 0 and pend != -1:                         
                        try:
                            output += (talk_id+'_'+oid+' '+ostart+' '+pend+' '+ '<s>' + text + ' 。+句点 </s>')
                        except:
                            print(talk_id)
                        with open(output_path1, 'a') as f:
                            f.write(output+'\n')
                            
                        text = ''
                        output= ''

                    clause_id = suws.attrib['ClauseUnitID']                                
                    ostart = ipu.attrib['IPUStartTime']
                    oid = ipu.attrib['IPUID']
                    pend = -1

                if 'SUWPOS' not in suws.attrib:                
                    continue

                if 'SUWMiscPOSInfo1' in suws.attrib:
                    pos = suws.attrib['SUWPOS'] + '/' + suws.attrib['SUWMiscPOSInfo1']
                elif 'SUWConjugateForm' in suws.attrib:
                    if 'SUWConjugateType' in suws.attrib:
                        pos = suws.attrib['SUWPOS'] + '/' + suws.attrib['SUWConjugateType'] + '/' + suws.attrib['SUWConjugateForm']
                    else:
                        pos = suws.attrib['SUWPOS'] + '/' + suws.attrib['SUWConjugateForm']
                else:
                    pos = suws.attrib['SUWPOS']
                                                        
                text += ' ' + suws.attrib['PlainOrthographicTranscription'] + '+' + pos
                lex = suws.attrib['PlainOrthographicTranscription'] + '+' + suws.attrib['SUWDictionaryForm'] + '+' + pos
                with open(output_path2, 'a') as f:
                    f.write(lex+'\n')

        pend = ipu.attrib['IPUEndTime']

    output += (talk_id+'_'+oid+' '+ostart+' '+pend+' '+ '<s>' + text + ' 。+句点 </s>')
    if '×' not in output:
        with open(output_path1, 'a') as f:
            f.write(output+'\n')        
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to sdb file')
    parser.add_argument('--output1', type=str, help='path to output file')
    parser.add_argument('--output2', type=str, help='path to output file')
    args = parser.parse_args()
          
    run(args)