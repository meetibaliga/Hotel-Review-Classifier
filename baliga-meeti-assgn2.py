#!/usr/bin/env python
"""part1.py"""

import math
import csv
import sys
import random
import numpy as np

#
#Common Functions
#

#Read files
def read_file(file):
    file = open(file, 'r', encoding='utf-8')
    file_data = file.readlines()
    return file_data

#Split review into lists of review-id, review-comment
def format_review_data(reviews):
    id_list = []
    review_list = []
    for i in reviews:
        reviews = i.split('\t')
        id_list.append(reviews[0])
        review_list.append(reviews[1].lower())
    review_list_size = len(review_list)
    return id_list, review_list, review_list_size

#Strip unwanted \n from lists
def clean_list(raw_word_list):
    clean_word_list = []
    for word in raw_word_list:
        word = word.rstrip("\n")
        clean_word_list.append(word)
    return clean_word_list

#Split reviews into word-list
def get_words_list(review):
    return review.replace("/"," ")\
                 .replace(". "," ").replace("."," ")\
                 .replace("(","").replace(")","")\
                 .replace(", "," ").replace(","," ").split()

#
#Feature Extraction Functions
#

#Count number of positive, negetive, pronoun words in the review
def get_word_count(x_words_list, words_list):
    count = 0
    for word in words_list:
        #if x_words_list.count(word) > 0:
            #print(word)
        count = count + x_words_list.count(word)
    return count

#Check if the review contains exclamation mark
def get_if_exc(words_list):
    is_exc = 0
    new_words_list = []
    for word in words_list:
        if word.find('!') >= 0:
            word = word.strip('!')
            is_exc = 1
        new_words_list.append(word)
    return new_words_list, is_exc
        
#Check if the review contains the word 'no'
def get_if_no(words_list):
    if "no" in words_list:
        return 1
    return 0
    
#Print features to CSV
def get_features(review_list, review_list_size, pos_words_list, neg_words_list, pronouns_list, review_id_list, review_class):
    feature_list = []
    for i in range(0, review_list_size):
        features = []
        words_list = get_words_list(review_list[i])
        words_list, is_exc = get_if_exc(words_list)
        #print(review_id_list[i])
        positive_word_count = get_word_count(pos_words_list, words_list)
        negative_word_count = get_word_count(neg_words_list, words_list)
        is_no = get_if_no(words_list)
        pronoun_count = get_word_count(pronouns_list, words_list)
        if(review_class == None):
            features = list((review_id_list[i] +" " +str(positive_word_count).strip() +" " +str(negative_word_count).strip() +\
                            " " +str(is_no).strip() +" " +str(pronoun_count).strip() +" " +str(is_exc).strip() +" " +\
                            str(round(math.log(len(words_list)),2))).split())
        else:
            features = list((review_id_list[i] +" " +str(positive_word_count).strip() +" " +str(negative_word_count).strip() +\
                            " " +str(is_no).strip() +" " +str(pronoun_count).strip() +" " +str(is_exc).strip() +" " +\
                            str(round(math.log(len(words_list)),2))+" " +str(review_class).strip()).split())
        #print(features)
        feature_list.append(features)
    return feature_list

def extract_features():
    print("Extracting Features")
    pronouns_list = clean_list(read_file('pronouns.txt'))
    pos_words_list = clean_list(read_file('positive-words.txt'))
    neg_words_list = clean_list(read_file('negative-words.txt'))
    
    #Read from HW2-testset.txt for testing
    if(len(sys.argv) == 3 and sys.argv[2] == "TEST_ONLY"):
        reviews_list = read_file('HW2-testset.txt')
        review_id, review_list, review_size = format_review_data(reviews_list)
        feature_list = get_features(review_list, review_size, pos_words_list, neg_words_list, pronouns_list, review_id, None)
        file = open('baliga-meeti-assgn2-part1.csv', 'w')
        writer = csv.writer(file)
        writer.writerows(feature_list)
        return

    #Read from hotelPosT-train.txt and hotelNegT-train.txt for training phase
    pos_reviews_list = read_file('hotelPosT-train.txt')
    neg_reviews_list = read_file('hotelNegT-train.txt')
    pos_review_id, pos_review_list, pos_review_size = format_review_data(pos_reviews_list)
    neg_review_id, neg_review_list, neg_review_size = format_review_data(neg_reviews_list)
    pos_feature_list = get_features(pos_review_list, pos_review_size, pos_words_list, neg_words_list, pronouns_list, pos_review_id, 1)
    neg_feature_list = get_features(neg_review_list, neg_review_size, pos_words_list, neg_words_list, pronouns_list, neg_review_id, 0)
    file = open('baliga-meeti-assgn2-part1.csv', 'w')
    writer = csv.writer(file)
    writer.writerows(pos_feature_list)
    writer.writerows(neg_feature_list)

#
#Make Training and Testing split
#

#Classify reviews into positive and negative
def segregate_reviews(feature_reviews_list):
    pos_list = []
    neg_list = []
    for feature in feature_reviews_list:
        f = feature.split(",")[7].rstrip("\n")
        #print(f)
        if f == "0":
           neg_list.append(feature)
        else:
           pos_list.append(feature)

    return pos_list, neg_list

def make_sets():
    print("Splitting features into training(80%) and testing(20%) sets")
    feature_reviews_list = read_file('baliga-meeti-assgn2-part1.csv')

    #Seperate positive and negetive reviews and shuffle
    positive_features, negative_features = segregate_reviews(feature_reviews_list)
    random.shuffle(positive_features)
    random.shuffle(negative_features)

    #Select 80% of positive reviews and 80% negetive reviews to make up the training set
    training_data = positive_features[:int(len(positive_features)*0.80)]
    training_data.extend(negative_features[:int(len(negative_features)*0.80)])

    #Rest of the 20% goes in test data
    testing_data = positive_features[int(len(positive_features)*0.80):]
    testing_data.extend(negative_features[int(len(negative_features)*0.80):])

    file = open('training_data.csv', 'w')
    writer = csv.writer(file)
    for t in training_data:
        writer.writerow(t.rstrip("\n").split(","))

    file = open('testing_data.csv', 'w')
    writer = csv.writer(file)
    for t in testing_data:
        writer.writerow(t.rstrip("\n").split(","))

#
#Train Function
#

def train():
    print("Training Data")
    training_data = read_file('training_data.csv')
    weights = np.array([0, 0, 0, 0, 0, 0, 0])

    features_list = training_data

    for c in range(1, 100000):
        index = random.randint(0, len(features_list)-1)
        feature = features_list[index]
#        print(feature)
        correct = float(feature.split(",")[7])

        xp = [float(i) for i in features_list[index].split(",")[1:7]]
        xp.append(1.0)
        x = np.array(xp)
#        print(x)

        rawscore = float(np.dot(weights, x))
        score = float(1/(1+np.exp(-rawscore)))
        gradient = (score - correct) * x
#        print(gradient)

        learningrate = 0.1
        new_weights = weights - learningrate * gradient
#        print(new_weights)
        weights = new_weights

    final_weights = weights

    file = open('final_weights.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(final_weights)

#
#Test Function
#

def test():
    weights = read_file("final_weights.csv")
    final_output = []    
    if(len(sys.argv) == 3 and sys.argv[2] == "TEST_ONLY"):
        testing_data = read_file("baliga-meeti-assgn2-part1.csv")
    else:
        testing_data = read_file("testing_data.csv")

    count = 0    
    for feature in testing_data:
        xp = [float(i) for i in feature.split(",")[1:7]]
        xp.append(1.0)
        x = np.array(xp)

        wp = [float(i) for i in weights[0].split(",")]
        w = np.asarray(wp)
        rawscore = round(float(np.dot(w, x)),2)
        score = round(float(1/(1+np.exp(-rawscore))),2)

        if score > 0.5:
            final_output.append(list((feature.split(",")[0]+","+"POS").split(",")))
            if(len(sys.argv) == 2 and int(feature.split(",")[7]) == 1):
                count = count + 1
        else:
            final_output.append(list((feature.split(",")[0]+","+"NEG").split(",")))
            if(len(sys.argv) == 2 and int(feature.split(",")[7]) == 0):
                count = count + 1

    print((count*100)/len(testing_data))
    #print(final_output)
    file = open('baliga-meeti-assgn2-out.txt', 'w')
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(final_output)

def main():
    if(sys.argv[1] == "extract_features"):
        extract_features()
    elif(sys.argv[1] == "make_sets"):
        make_sets()
    elif(sys.argv[1] == "train"):
        train()
    elif(sys.argv[1] == "test"):
        test()

if __name__ == '__main__':
    main()	
