#!/usr/bin/env python
#coding=gbk
 
from sys import *
path.append("./config/")
import sofa, urllib, json, re, sys, os

os.system('export SOFA_CONFIG=./config/drpc_client.xml')
sofa.use('drpc.ver_1_0_0', 'S')
sofa.use('nlpc.ver_1_0_0', 'wordseg')
conf = sofa.Config()
conf.load('./config/drpc_client.xml')
wordseg_agent = S.ClientAgent(conf['sofa.service.nlpc_wordseg_3016'])

def GET_TERM_POS(property):
    return ((property) & 0x00FFFFFF)
def GET_TERM_LEN(property):
    return ((property) >> 24)


def get_wordseg(line, seg_level):

    query=line

    qwords = []
    m_input = wordseg.wordseg_input()
    m_input.query = query
    m_input.lang_id = int(0)
    m_input.lang_para = int(0)
    input_data = sofa.serialize(m_input)
    for i in range(5) :
        try:
            ret, output_data = wordseg_agent.call_method(input_data)
            break
        except Exception as e:
            continue
    if len(output_data) == 0:
        #stdout.write('No result' + '\n')
        return []

    m_output = wordseg.wordseg_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    m_output = m_output.scw_out

    #basic
    if seg_level == 'basic' :
        for i in range(m_output.wsbtermcount):
            posidx = GET_TERM_POS(m_output.wsbtermpos[i])
            poslen = GET_TERM_LEN(m_output.wsbtermpos[i])
            word = m_output.wordsepbuf[posidx : posidx + poslen]
            try:
                word = word.strip().decode('gbk')
            except Exception as e:
                print >> sys.stderr,e
                continue
            qwords.append(word)

    #segment
    if seg_level == 'segment' :
        for i in range(m_output.wpbtermcount):
            posidx = GET_TERM_POS(m_output.wpbtermpos[i])
            poslen = GET_TERM_LEN(m_output.wpbtermpos[i])
            word = m_output.wpcompbuf[posidx : posidx + poslen]
            try:
                word = word.strip().decode('gbk')
            except Exception as e:
                print >> sys.stderr,e
                continue
            qwords.append(word)

    return qwords

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("gbk")
    print ' '.join(get_wordseg("中华人名共和国", 'basic')).encode('gbk')
