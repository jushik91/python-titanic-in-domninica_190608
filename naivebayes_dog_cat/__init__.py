from naivebayes_dog_cat.niavebayes_naver_movie import NaiveBayesClassfier
context = './data/'
model = NaiveBayesClassfier()
model.train(context+'review_train.csv')
##print(model.classfy('내 인생에서 쓰레기같은 영화'))
##print(model.classfy('내 인생 최고의 영화'))
print(model.classfy('평범한 영화'))
