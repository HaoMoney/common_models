#coding=utf-8
"""
author : qianhao
to do : annoy 
"""
from annoy import AnnoyIndex
f=32
t = AnnoyIndex(f, 'euclidean')
subject_dict = {}
with open("../dict/db_subjects_ch.id","r") as f:
	data = f.readlines()
	cnt = 0
	for line in data:
		line = line.strip()
		s_line = line.split("\t")
		if len(s_line) == 3:
			subject_dict[s_line[1]] = s_line[2]

user_field_dict = {}
with open("../test_data/test_guids_feats","r") as f:
	data = f.readlines()
	cnt = 0
	for line in data:
		line = line.strip()
		s_line = line.split("|")
		if len(s_line) < 2:
			continue
		res = []
		for item in s_line[1:]:
			item = item.split(":")
			if len(item) == 3:
				if float(item[2]) > 0.05 and item[1] in subject_dict:
					res.append(subject_dict[item[1]])
		user_field_dict[s_line[0]] = " ".join(res)
		cnt += 1
query_dict = {}
with open("white.querys.vec","r") as f:
	data = f.readlines()
	cnt = 0
	for line in data:
		line = line.strip()
		s_line = line.split("\t")
		if len(s_line) < 2:
			continue
		v = [float(i) for i in s_line[1].split()]
		t.add_item(cnt,v)
		items = s_line[0].split("@")

		if len(items) == 2 and items[1] in subject_dict:
			query_dict[cnt] = items[0] + "@" + subject_dict[items[1]]
		else:
			query_dict[cnt] = s_line[0]
		cnt += 1
t.build(20)
with open("vision.user.vec","r") as f:
	data = f.readlines()
	cnt = 0
	for line in data:
		line = line.strip()
		s_line = line.split("\t")
		if len(s_line) < 4:
			continue
		user_vec = [float(i) for i in s_line[3].split()]
		ans = t.get_nns_by_vector(user_vec,50,include_distances=True)
		out_str = ""
		for i in range(len(ans[0])):
			if ans[0][i] in query_dict:
				out_str += query_dict[ans[0][i]] + ":" + str(ans[1][i]) + ";"
		print(out_str)

