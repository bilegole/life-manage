#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def op_di (dicti) :
	print('*******************************************************')
	#for num in dicti :
	#	print('%s\t%s' % (num,dicti[num]))
	print(dicti)
	print('*******************************************************','\njack',dicti.get('jack','don\'t exist'),'\n\n')
def ins (qqq) :
	d['jack']=198
	print('try to insert jack = 198\n')
d = {'Michael':95,'Bob':75,'Tracy':85}


op_di(d)
ins(3)
op_di(d)

