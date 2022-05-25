# -*- coding: utf-8 -*-
"""
@author: 
@date: 2021-05-02
@func:
    
"""





if __name__ == "__main__":
    import sys
    cnt = 1
    for line in sys.stdin:
        line = line.strip()
        print(line+"\t"+str(cnt))
        cnt += 1
