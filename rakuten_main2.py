#! /usr/local/bin/python2.7
#-*- coding: utf-8 -*-
#/////////////////////////////////////評価はary13番目1から5、内容がary15番目
#楽天のデータを分割するプログラム


######################################################################
# ジャンル分けしたレビューデータを読み込み，処理をするプログラム
#
# 作成日：2015/07/02～
######################################################################


###########################################################################################
# 
# 「みんなのレビュー・口コミ情報」
#
# 0 : 投稿者			：「user1」のようにマスクしたユーザ名
# 1 : 年齢				：10歳代
# 2 : 性別				：[0:男 1:女 2:不明]
# 				　※ユーザ情報が非公開に設定されているものは、
# 				　　投稿者は「購入者さん」、年齢・性別は空白となります。
# 3 : 商品コード			：「店舗コード:商品id」
# 4 : 商品名
# 5 : 店舗名
# 6 : 商品URL			：商品URLのドメイン以降の部分
# 　　　　　　　　　　　　　　　　　「http://item.rakuten.co.jp/[商品URL]」で商品ページのURL
# 　　　　　　　　　　　　　　　　　　※全ての商品ページが存在するとは限りません
# 7 : 商品ジャンルID		：商品のジャンルID
# 8 : 商品価格			：商品購入時の価格
# 9 : 購入フラグ			：[0:購入なし 1:購入あり]
# 10: 内容				：「実用品・普段使い」などの文字列
# 11: 目的				：「自分用」などの文字列
# 12: 頻度				：「はじめて」などの文字列
# 13: 評価ポイント			：0-5の6段階評価
# 14: レビュータイトル
# 15: レビュー内容
# 16: レビュー登録日時		：レビュー登録年月日（フォーマット「yyyy-mm-dd HH:MM:SS」）
#
#############################################################################################


import numpy as np
import re
import MeCab
import matplotlib.pyplot as plt
import nltk
import sys
import codecs
import csv
import pprint
import gensim

import pylab
from pylab import *
from matplotlib.font_manager import FontProperties

sys.path.append("home/python/rakuten_program/")
from rakuten_module.rakuten_age_purpose_module import rakuten_age_purpose
from rakuten_module.rakuten_genre_module import rakuten_genre_purse
from rakuten_module.rakuten_age_genre_module import rakuten_age_genre
from rakuten_module.rakuten_timeclassify_module import rakuten_timeclassify
from rakuten_module.rakuten_tfidf_module import rakuten_tfidf
#from calc_tf import rakuten_tf_class

from gensim import corpora, models, similarities



# 日本語を含む文字列を標準入出力とやり取りする場合
# UTF-8の文字列を標準出力に出力したり，標準入力から入力したりできるようになる
# そうしないと，UnicodeEncodeError/UnicodeDecodeErrorが発生する場合がある
# sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
# sys.stdin = codecs.getreader('utf_8')(sys.stdin)
# しかし，これをやると，「print row」のところで，ASCII関係のエラーが出てしまう


def pp(obj):
	pp = pprint.PrettyPrinter(indent=4, width=160)
	str = pp.pformat(obj)
	return re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),
																16)), str)


def purse1(ary, item1):

	# データの有無を管理するフラグ
	data_flag1 = 0

	sub_ary = []
	ary_purse = []

	# 指定した項目のデータを抽出して，ファイルに書き込む(リストに追加する)
	for i in range(len(ary)):

		sub_ary = [] # リストの初期化

		# 抽出したい項目のとき，その項目の内容をファイルに書き込む
		for j in range(0,17):
			# print i,ary[j][i]

			# 項目が13のとき
			if (j == item1): #13

				# データがある場合を検出する
				searchOb1 = re.search(".", ary[i][item1])
				if searchOb1:
					data_flag1 = 1
		
		# 欲しい項目がすべて存在するレビューであった場合，sub_aryに各項目を格納する
		if data_flag1 == 1:
			sub_ary.append(ary[i][item1])

			# 欲しい項目のみが格納されたリストを，新たなリストに格納する
			ary_purse.append(sub_ary)

		# フラグの初期化
		data_flag1 = 0


	return ary_purse


def purse2(ary, item1, item2):

	# データの有無を管理するフラグ
	data_flag1 = 0
	data_flag2 = 0

	sub_ary = []
	ary_purse = []

	# 指定した項目のデータを抽出して，ファイルに書き込む(リストに追加する)
	for i in range(len(ary)):

		sub_ary = [] # リストの初期化

		# 抽出したい項目のとき，その項目の内容をファイルに書き込む
		for j in range(0,17):
			# print i,ary[j][i]

			# 項目が13のとき
			if (j == item1): #13

				# データがある場合を検出する
				searchOb1 = re.search(".", ary[i][item1])
				if searchOb1:
					data_flag1 = 1

			# 項目が15のとき
			if (j == item2): #15
				
				# データがある場合を検出する
				searchOb2 = re.search(".", ary[i][item2])
				if searchOb2:
					data_flag2 = 1
		
		# 欲しい項目がすべて存在するレビューであった場合，sub_aryに各項目を格納する
		if data_flag1 == 1 and data_flag2 == 1:
			sub_ary.append(ary[i][item1])
			sub_ary.append(ary[i][item2])

			# 欲しい項目のみが格納されたリストを，新たなリストに格納する
			ary_purse.append(sub_ary)

		# フラグの初期化
		data_flag1 = 0
		data_flag2 = 0

	return ary_purse	


def purse3(ary, item1, item2, item3):

	# データの有無を管理するフラグ
	data_flag1 = 0
	data_flag2 = 0
	data_flag3 = 0

	sub_ary = []
	ary_purse = []

	# 指定した項目のデータを抽出して，ファイルに書き込む(リストに追加する)
	for i in range(len(ary)):

		sub_ary = [] # リストの初期化

		# 抽出したい項目のとき，その項目の内容をファイルに書き込む
		for j in range(0,17):
			# print i,ary[j][i]

			# 項目が13のとき
			if (j == item1): #13

				# データがある場合を検出する
				searchOb1 = re.search(".", ary[i][item1])
				if searchOb1:
					data_flag1 = 1

			# 項目が15のとき
			if (j == item2): #15
				
				# データがある場合を検出する
				searchOb2 = re.search(".", ary[i][item2])
				if searchOb2:
					data_flag2 = 1

			# 項目がitem3のとき
			if (j == item3): #7
				
				# データがある場合を検出する
				searchOb3 = re.search(".", ary[i][item3])
				if searchOb3:
					data_flag3 = 1
		
		# 欲しい項目がすべて存在するレビューであった場合，sub_aryに各項目を格納する
		if data_flag1 == 1 and data_flag2 == 1 and data_flag3 == 1:
			sub_ary.append(ary[i][item1])
			sub_ary.append(ary[i][item2])
			sub_ary.append(ary[i][item3])

			# 欲しい項目のみが格納されたリストを，新たなリストに格納する
			ary_purse.append(sub_ary)

		# フラグの初期化
		data_flag1 = 0
		data_flag2 = 0
		data_flag3 = 0

	return ary_purse



def make_ary(fl):

	review_num = 0  # レビューの件数をカウントするための変数
	doc = [] # レビューを件ごとに入れるリスト
	ary = [] # レビューを項目ごとに入れるリスト

	ary1 = [] # 特定の項目を格納するリスト
	ary2 = [] # ジャンルについて格納されたリスト
	# sub_ary = [] # 項目を格納するためのリスト

	# 欲しい項目の指定
	item1 = 1
	item2 = 11

	# 一つのrowは，0(投稿者)～16(投稿日時)までのデータをひとまとまりにしたもの(レビュー1件)である
	# そのレビュー1件を順番に，リストに追加する
	# 追加するレビュー数は，2000件までとする
	for row in fl:

		# if (review_num % 2) == 0 and review_num < 1000:
		doc.append(row)   # レビューをリストに追加する

		# 確認のための出力
		if review_num == 1:
			print row

		review_num = review_num + 1 # レビュー件数のカウント

	print "1つのファイルにおける全レビュー数：", review_num # 全てのレビュー件数を出力する


	# レビューを項目ごとに分割し，各項目の内容を新たなリストに追加する
	# ary[a][b]
	# a : レビューの件数(何件目か)
	# b : レビューの項目(0～16)
	# 17個にスプリットしたものを一気にいれていくため，二次元配列として扱う？
	# 17個の項目を，1つのリストとしてリストに追加している
	# つまり，17個の項目のリストを要素とするリストがaryである
	for i in range(len(doc)):

		# すべての項目の情報があるレビューのみをリストに追加する
		if len(doc[i].split(",")) == 17:
			# 表示された項目がタブで区切られていたため，タブでスプリットしている？
			ary.append(doc[i].split(",")) # 本文を分けるためのもの

	# 登録日時の改行を消す
	for i in range(len(ary)):
		searchOb_kaigyou = re.search("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ary[i][16])
		if searchOb_kaigyou:
			ary[i][16] = searchOb_kaigyou.group()

	print len(ary)

	return ary




# def top_genre(genre_ary, ary):

# 	top_genre = []
# 	top_genre_id = ""
# 	flag = 0
# 	# ジャンルIDから，トップジャンルを抽出する
# 	for i in range(len(ary)):
# 		flag = 0
# 		top_genre_id = ary[i][2]
# 		while flag < 1:
# 			for i in range(len(genre_ary)):
# 				if int(genre_ary[i][0]) == int(top_genre_id):
# 					top_genre_id = genre_ary[i][2]

# 					if int(top_genre_id) == 0:
# 						flag = 1
# 						top_genre.append(genre_ary[i][1])

# 	return top_genre





if __name__ == "__main__":

	# 日本語を含む文字列を標準入出力とやり取りする場合に書く
	# UTF-8の文字列を標準出力に出力したり，標準入力から入力したりできるようになる
	sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
	sys.stdin = codecs.getreader('utf-8')(sys.stdin)

	print sys.getdefaultencoding() # デフォルトの文字エンコーディングの確認

	# デフォルトのエンコーディングを変更する
	reload(sys)
	sys.setdefaultencoding('utf-8')
	print sys.getdefaultencoding()


	mecab = MeCab.Tagger ("--node-format=%m\s%f[0]\\n --eos-format='' ")

	f1 = open("/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/consumer_electronics2012_01_limit.csv","r") # 2361件
	# f2 = open("/home/ogawa/rakuten_data/review_ichiba/ichiba04_review201202_20140221.tsv","r")
	# f3 = open("/home/ogawa/rakuten_data/review_ichiba/ichiba04_review201003_20140221.tsv","r")

	# fp = open('rakuten_time.txt','w') # 書き込み用のファイル

	ary201201 = []

	ary1 = []

	ary201201 = make_ary(f1)
	# ary201002 = make_ary(f2)
	# ary201003 = make_ary(f3)

	# 最後の登録日時のデータの対してのみ，最後に改行がついている
	# ary1 = purse3(ary201201, 16, 15, 7)
	# ary2 = purse2(ary201002, 16, 15)
	# ary3 = purse2(ary201003, 16, 15)
	# ary1 = purse3(doc, ary, 1, 11, 7)

	# print doc[0]

	# レビュー1件の項目数を出力する
	# 結果は17であることから，aryの1つの要素には，17個のデータが入っているといえる
	print "レビュー1件の項目数：", len(ary201201[0])

	# aryの1つの要素を出力すると，リストであることから，aryはリストのリストであると分かる
	print "レビュー1件の情報：", ary201201[2]

	# 確認のために，リストの1つの要素について，リストの中身を順番に出力する
	for i in range(0,17):
		print i, ":", ary201201[2][i] # 項目の内容を出力する

		# データがない場合を検出する
		searchOb = re.search(".", ary201201[2][i])
		if not searchOb:
			print "no data"


	# 各レビューにおける，欲しい項目のみのリストを格納したリストを使用し，ファイルに書き込む
	# for i in range(len(ary1)):

	# 	if i > 0:
	# 		fp.write(", ")

	# 	fp.write("{")
	# 	fp.write(str(ary1[i][0]))
	# 	fp.write("|, |") # 項目の内容を区切る文字
	# 	fp.write(str(ary1[i][1]))
	# 	fp.write("}")

	f1.close()
	# fp.close()


	# ジャンルマスタの情報を取得する
	genre_ary = []
	genre_ary = rakuten_genre_purse.genre_purse()




	#----------------------------------------------------------------------------------
	# ary_gr201201 = []

	# ary_gr2010_01 = rakuten_timeclassify.timeclassify(ary1)
	# # rakuten_timepurse.timepurse(ary2)
	# # rakuten_timepurse.timepurse(ary3)


	# レビュー内容の情報だけを抽出してくる
	rv2012_01 = []
	for i in range(len(ary201201)):
		# if (i % 100) == 0:
		rv2012_01.append(ary201201[i][15])


	# rakuten_tfidf.calc_tfidf(rv2012_01)


	# f_w = open("/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/review_content2012_01.txt", "w")

	# for i in range(len(rv2012_01)):
	# 	# if (i % 2) == 0:
	# 	f_w.write(rv2012_01[i])
	# 	f_w.write("\n")

	# f_w.close()



	###########################################################################
	#
	# LDAの実行
	#
	###########################################################################


	# リスト内のそれぞれの要素を，一つの文書とする
	documents = []
	for i in range(len(rv2012_01)):
		documents.append(rv2012_01[i])
		# f_w.write(rv2012_01[i])
		# f_w.write("\n")
	# f_w.close()


	# 単語を分割する
	# f_w2 = open("/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/input2.txt", "w")
	f_w2 = open("/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/2012_01_lda/input.txt", "w")
	word = []
	part = []
	norn = []
	texts = []
	doc_count = 0
	for doc in documents:

		norn = []
		data_mecab = mecab.parse(doc)

		data_mecab = data_mecab.split() # 真ん中の空白をなくす
		word = data_mecab[0::2] # 偶数番目の要素だけを取り出す
		part = data_mecab[1::2] # 奇数番目の要素だけを取り出す


		# 名詞の単語リストを作成
		for i in range(len(part)):

			# norn.append(word[i])

			searchOb = re.search("名詞", part[i])
			searchOb2 = re.search("形容詞", part[i])
			searchOb_no = re.search("の", word[i])
			searchOb_half = re.search(u'[a-zA-Z0-9]+', word[i])
			searchOb_full = re.search("[、-◯]+", word[i])
			# searchOb_full = re.search("[Ａ-Ｚ]+", word[i])


			if searchOb_no or searchOb_half:
				continue

			if searchOb or searchOb2:
				norn.append(word[i])

		texts.append(norn)
		doc_count += 1

#----------------------------------------------------------------------------------------------------------

	# 辞書を作る
	dictionary = corpora.Dictionary(texts)
	dictionary.filter_extremes(no_below=20, no_above=0.3) # 辞書にフィルターをかける
	# no_berow: 使われてる文章がno_berow個以下の単語無視
	# no_above: 使われてる文章の割合がno_above以上の場合無視

	# 辞書を保存する
	#dictionary.save('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/review_lda.dict') # store the dictionary, for future reference
	dictionary.save_as_text('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/2012_01_lda/review_lda_text.txt') # テキスト形式で保存する

	#辞書ファイルのロード
	# dictionary = corpora.Dictionary.load_from_text('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/review_lda_text.txt')
	print dictionary # 辞書の出力
	print "ディクショナリの長さ：", len(dictionary)

	# 辞書は単語と単語IDとのマッピングを表している
	print pp(dictionary.token2id)

#-----------------------------------------------------------------------------------------------------
	# text全体に対する特徴ベクトルの集合= corpusを作成する。
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/2012_01_lda/review_lda.mm', corpus) # store to disk, for later use
	# print corpus

	# corpusは膨大になってしまうため、一度Matrix Market fileに保存される（した）
	# そのコーパスを呼び出す
	# corpus = corpora.MmCorpus('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/review_lda.mm')


	# 特徴ベクトルをtfidfベクトル空間のベクトルに変換する
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus] # 変換したものをcorpusに適用する
	print corpus_tfidf


	# LDAの実行  ( http://sucrose.hatenablog.com/entry/2013/10/29/001041より )
	# 現在は別の方法でやっている
	# corpus = gensim.corpora.lowcorpus.LowCorpus('/home/ogawa/rakuten_data/review_ichiba/review_genre/2012_01/corpus2.txt')

	# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=20, id2word=corpus.id2word)
	# for topic in lda.show_topics(-1):
	# 	print topic
	# for topics_per_document in lda[corpus]:
	# 	print topics_per_document



	# LSIを実行する
	lsi = gensim.models.LsiModel(corpus=corpus_tfidf, num_topics=10, id2word=dictionary)
	# lsi.save('jawiki_lsi_topics300.model')  # せっかく計算したので保存
	# 次回からは次のコマンドでロードできる
	# lsi = gensim.models.LsiModel.load('jawiki_lsi_topics300.model')

	print "LSIの実行結果"
	for topic in lsi.show_topics(-1):
		print topic
	print "\n"


	# LDAを実行する
	lda = gensim.models.LdaModel(corpus=corpus_tfidf, num_topics=10, id2word=dictionary)
	# lda.save('result_lda.model')  # せっかく計算したので保存
	# lda = gensim.models.LdaModel.load('jawiki_lda.model')

	print "LDAの実行結果"
	topic_num = 0
	for topic in lda.show_topics(-1):
		print "トピック", topic_num, ":", topic
		topic_num += 1
	print "\n"


	topic_dic = {} # 各トピックが最も現れている文書数を見るためのディクショナリ
	topic_dic_flag = 0 # 最も確率が大きいトピックだけをカウントするためのフラグ

	# 各トピックの登録
	topic_dic["トピック0"] = 0
	topic_dic["トピック1"] = 0
	topic_dic["トピック2"] = 0
	topic_dic["トピック3"] = 0
	topic_dic["トピック4"] = 0
	topic_dic["トピック5"] = 0
	topic_dic["トピック6"] = 0
	topic_dic["トピック7"] = 0
	topic_dic["トピック8"] = 0
	topic_dic["トピック9"] = 0

	cn = 0
	cn_flag = 0
	for topics_per_document in lda[corpus_tfidf]:
		topic_dic_flag = 0
		cn_flag = 0
		# if cn == 238:
		for k,l in sorted(topics_per_document, key=lambda x:x[1], reverse=True):
			
			# 特定のレビューに対するトピックの分布を出力する
			if cn == 0:
				if cn_flag == 0:
					print "###### ", cn, "番目のレビュー ######"
					print "--- 商品名 ---"
					print ary201201[cn][4]
					print "--- レビュー内容 ---"
					print documents[cn]
					print "--- 投稿日時 ---"
					print ary201201[cn][16]
					print "--- トピックの分布 ---"
					cn_flag = 1

				print k,l
			# 特定のレビューに対するトピックの分布を出力する	
			if cn == 4000:
				if cn_flag == 0:
					print "###### ", cn, "番目のレビュー ######"
					print "--- 商品名 ---"
					print ary201201[cn][4]
					print "--- レビュー内容 ---"
					print documents[cn]
					print "--- 投稿日時 ---"
					print ary201201[cn][16]
					print "--- トピックの分布 ---"
					cn_flag = 1

				print k,l
			# 特定のレビューに対するトピックの分布を出力する	
			if cn == 8000:
				if cn_flag == 0:
					print "###### ", cn, "番目のレビュー ######"
					print "--- 商品名 ---"
					print ary201201[cn][4]
					print "--- レビュー内容 ---"
					print documents[cn]
					print "--- 投稿日時 ---"
					print ary201201[cn][16]
					print "--- トピックの分布 ---"
					cn_flag = 1

				print k,l
			# 特定のレビューに対するトピックの分布を出力する	
			if cn == 12000:
				if cn_flag == 0:
					print "###### ", cn, "番目のレビュー ######"
					print "--- 商品名 ---"
					print ary201201[cn][4]
					print "--- レビュー内容 ---"
					print documents[cn]
					print "--- 投稿日時 ---"
					print ary201201[cn][16]
					print "--- トピックの分布 ---"
					cn_flag = 1

				print k,l
			# 特定のレビューに対するトピックの分布を出力する	
			if cn == 16000:
				if cn_flag == 0:
					print "###### ", cn, "番目のレビュー ######"
					print "--- 商品名 ---"
					print ary201201[cn][4]
					print "--- レビュー内容 ---"
					print documents[cn]
					print "--- 投稿日時 ---"
					print ary201201[cn][16]
					print "--- トピックの分布 ---"
					cn_flag = 1

				print k,l	

			# 最も確率が高いトピックのとき，そのトピックをディクショナリでカウントする
			if topic_dic_flag == 0:
				if k == 0:
					topic_dic["トピック0"] += 1
				if k == 1:
					topic_dic["トピック1"] += 1
				if k == 2:
					topic_dic["トピック2"] += 1
				if k == 3:
					topic_dic["トピック3"] += 1	
				if k == 4:
					topic_dic["トピック4"] += 1
				if k == 5:
					topic_dic["トピック5"] += 1
				if k == 6:
					topic_dic["トピック6"] += 1
				if k == 7:
					topic_dic["トピック7"] += 1
				if k == 8:
					topic_dic["トピック8"] += 1
				if k == 9:
					topic_dic["トピック9"] += 1
			topic_dic_flag = 1

		cn += 1

	# 各トピックが最も現れている文書数を出力する
	print "\n各トピックが最も現れている文書数"
	for k,l in topic_dic.items():
		print k,l

