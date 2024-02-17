# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.
import os
import argparse
import csv
import math
# import sklearn
# from sklearn import preprocessing


def train_model_interpolation(train_data,lambdas,n):
    #Calling ngrams for different values of n
    zerograms = create_ngrams(train_data,0)
    unigrams = create_ngrams(train_data,n-2)
    bigrams = create_ngrams(train_data,n-1)
    trigrams =  create_ngrams(train_data,n)
    #we store the values of lambda into variables 
    l1,l2,l3 = lambdas[0],lambdas[1],lambdas[2]
    
    prob = {}

    for lang in list(trigrams.keys()):
        prob[lang] = {}
        for ngram in trigrams[lang]:
            count_tri = trigrams[lang][ngram]
            count_uni = unigrams[lang][ngram[:len(ngram)-2]]
            count_bi = bigrams[lang][ngram[:len(ngram)-1]]
            count_N = zerograms[lang][ngram[:len(ngram)-3]]
            #calculate probabilites using the counts and lambdas
            prob[lang][ngram] = math.log(l1[lang]*float((count_tri)/(float(count_bi)))+ l2[lang]*float((count_bi)/(float(count_uni)))+l3[lang]*float((count_uni/count_N)))
            
    return prob

def test_model_interpolation(test_data,train_data,vocab, prob, lambdas, n):
    #Calling ngrams for different values of n
    ngrams = create_ngrams(test_data,n)
    N = create_ngrams(test_data, 0)
    zerogramsTrain = create_ngrams(train_data,0)
    n1gramsTrain = create_ngrams(train_data, n-1)
    n2gramsTrain = create_ngrams(train_data, n-2)
    l1,l2 = lambdas[0],lambdas[1]
    test_results = {}

    for ngramFile in ngrams.keys():
        test_results[ngramFile] = {}
        perplexityList = [] #list to store perplexities
        for lang in prob.keys():
            probability = 0
            perplexity = 0
            for ngram in ngrams[ngramFile]:
                smooth = True
                for x in ngram:
                    if x not in vocab[lang]:
                        smooth = False
                if ngram in prob[lang].keys(): # if value is found in the prob dictionary then we update the value for probability
                    probability = probability + prob[lang][ngram] * ngrams[ngramFile][ngram]
                elif smooth:
                    probability_model = 0
                    try:
                        probability_model += l2[lang]*(n1gramsTrain[lang][ngram[:len(ngram)-1]]/n2gramsTrain[lang][ngram[:len(ngram)-2]])
                    except:
                        count_bi = 0
                    try:
                        probability_model += l1[lang]*(n2gramsTrain[lang][ngram[:len(ngram)-2]]/zerogramsTrain[lang][:len(ngram)-3])
                    except:
                        count_uni = 0
                    try:
                        prob[lang][ngram] = math.log(probability_model)
                        probability = probability + prob[lang][ngram] * ngrams[ngramFile][ngram]
                    except:
                        continue
                elif "<UNK>" not in ngram:
                    perplexity = math.inf

            if perplexity != math.inf:
                perplexity = math.exp(probability)**(-1/N[ngramFile][()])
            perplexityList.append((perplexity, lang))
        #geting the min  perplexity
        minPerplexity = min(perplexityList, key=lambda x:x[0])
        test_results[ngramFile] = (minPerplexity[1], minPerplexity[0], n)

    return test_results
    

def deleted_interpolation(train_data,n):
    l1,l2,l3 = {},{},{} # variables to store lambdas

    #calling ngrams for different n
    zerograms = create_ngrams(train_data,0)
    unigrams = create_ngrams(train_data,n-2)
    bigrams = create_ngrams(train_data,n-1)
    trigrams =  create_ngrams(train_data,n)
    
    #iterating through the trigrams dict 
    for lang in list(trigrams.keys()):
        l1_temp,l2_temp,l3_temp = 0,0,0 #setting temp lambda values
        for ngram in trigrams[lang]:
            #extracting count values from these dictionaries
            count_tri = trigrams[lang][ngram]
            count_bi = bigrams[lang][ngram[:len(ngram)-1]]
            count_uni = unigrams[lang][ngram[:len(ngram)-2]]
            count_zero = zerograms[lang][ngram[:len(ngram)-3]]
            
            #obtaining values for tri,bi and uni
            if float(count_bi-1) == 0:
                tri = 0
            else:
                tri = (float((count_tri-1)/(float(count_bi-1))))
            if float(count_uni-1) == 0:
                bi = 0
            else:
                bi = (float((count_bi-1)/(float(count_uni-1))))
            if float(count_zero-1) == 0:
                uni = 0
            else:
                uni = (float((count_uni-1)/(float(count_zero-1))))
            max_list = [tri,bi,uni] # stroing values to a list

            max_item =  max(max_list) #getting max
            #comparing max value and updating lamdas accordingly
            if tri == max_item:
                l1_temp += count_tri
            elif bi == max_item:
                l2_temp += count_tri
            else:
                l3_temp += count_tri
        #normalizing lambdas
        lamda_sum = l1_temp+l2_temp+l3_temp # calculating sum of the lambdas
        l1[lang] = l1_temp/lamda_sum
        l2[lang] = l2_temp/lamda_sum
        l3[lang] = l3_temp/lamda_sum

    return [l1,l2,l3]


def train_model_laplace(train_data, vocab, n):
    # To obtain ngram and n-1gram
    n1grams = create_ngrams(train_data,n-1)
    ngrams =  create_ngrams(train_data,n)

    prob = {}  # Dict to hold each probability

    # To extract counts and use them to calculate probability
    for lang in list(ngrams.keys()):
        prob[lang] = {}
        for ngram in ngrams[lang]:
            count_bi = ngrams[lang][ngram]
            count_uni = n1grams[lang][ngram[:len(ngram)-1]]
            prob[lang][ngram] = math.log(float((count_bi+1)/(float(count_uni)+len(vocab[lang]))))
        
    return prob


# This function accepts pre-processed test data, train data, 
# vocab, probability list and n as arguments.
# Opens the file and returns a list of lines read from the file
def test_model_laplace(test_data, train_data, vocab, prob, n):
    # To create ngrams using test data
    ngrams = create_ngrams(test_data,n)
    # To create n-1grams using train_data
    n1grams = create_ngrams(train_data, n-1)
    N = create_ngrams(test_data, 0)

    test_results = {}  # Dict to hold test results computed below

    for ngramFile in ngrams.keys():
        test_results[ngramFile] = {}  # Setting file names as keys in test_results dict. 
        perplexityList = []  # List to store perplexities
        for lang in prob.keys():
            probability = 0
            perplexity = 0
            for ngram in ngrams[ngramFile]:
                smooth = True
                for x in ngram:
                    if x not in vocab[lang]:
                        smooth = False
                if ngram in prob[lang].keys():
                    probability = probability + prob[lang][ngram] * ngrams[ngramFile][ngram]
                elif smooth:
                    try: 
                        count_uni = n1grams[lang][ngram[:len(ngram)-1]]
                        prob[lang][ngram] = math.log(float((1)/(float(count_uni)+len(vocab[lang]))))
                        probability = probability + prob[lang][ngram] * ngrams[ngramFile][ngram]
                    except:
                        continue
                elif "<UNK>" not in ngram:
                    perplexity = math.inf

            if perplexity != math.inf:
                perplexity = math.exp(probability)**(-1/N[ngramFile][()])
            perplexityList.append((perplexity, lang))
        minPerplexity = min(perplexityList, key=lambda x:x[0])
        test_results[ngramFile] = (minPerplexity[1], minPerplexity[0], n)

    return test_results


# This function accepts input directory path as an argument,
# Opens the file and returns a list of lines read from the file
def train_model_unsmoothed(train_data,n):
    n1grams = create_ngrams(train_data,n-1)
    ngrams =  create_ngrams(train_data,n)

    prob = {}

    for lang in list(ngrams.keys()):
        prob[lang] = {}
        for ngram in ngrams[lang]:
            count_bi = ngrams[lang][ngram]
            count_uni = n1grams[lang][ngram[:len(ngram)-1]]
            prob[lang][ngram] = math.log(float(count_bi/float(count_uni)))
        
    return prob


# This function accepts input directory path as an argument,
# Opens the file and returns a list of lines read from the file
def test_model_unsmoothed(test_data, prob, n):
    ngrams = create_ngrams(test_data,n)
    N = create_ngrams(test_data, 0)
    test_results = {}

    for ngramFile in ngrams.keys():
        test_results[ngramFile] = {}
        perplexityList = []
        for lang in prob.keys():
            probability = 0
            perplexity = 0
            for ngram in ngrams[ngramFile]:
                if ngram in prob[lang].keys():
                    probability = probability + prob[lang][ngram] * ngrams[ngramFile][ngram]
                elif "<UNK>" not in ngram:
                    perplexity = math.inf

            if perplexity != math.inf:
                perplexity = math.exp(probability)**(-1/N[ngramFile][()])
            perplexityList.append((perplexity, lang))
        minPerplexity = min(perplexityList, key=lambda x:x[0])
        test_results[ngramFile] = (minPerplexity[1], minPerplexity[0], n)

    return test_results
    

# This function accepts the tokenized data and n as arguments,
# Opens the file and returns a list of lines read from the file
def create_ngrams(data, n):
    ngrams = {}  # Dict. to store the ngrams

    # To extract the data for each language and create ngrams list
    for lang in list(data.keys()):
        ngram = []
        words = data[lang]
        for i in range(len(words)- n+1):
            ngram.append(tuple(words[i:i+n]))

        ngram_count = {}  # Dict to store counts for each ngram in ngrams

        # To calculate the individual ngram counts
        for item in ngram:
            if item in list(ngram_count.keys()):
                ngram_count[item] += 1
            else:
                ngram_count[item] = 1
        
        # adding the count for each ngram to the the ngrams dict.
        ngrams[lang] = ngram_count 

    return ngrams


# This function accepts the tokenized training data as an argument,
# Creates a vocabulary of all words in text using a dict. data structure
def create_vocab(train_data):
    vocab = {}
    for lang in list(train_data.keys()):
        
        vocabulary = list(set(train_data[lang]))
        vocab[lang] = vocabulary  # Setting file name as key
    return vocab


# This function accepts the tokenized test data and the vocabulary as arguments,
# Replaces words not seen before in the test data set with the <UNK> tag
# so that unseen words don't hinder final test results
def test_vocab(test_data,vocab):
    for lang in list(test_data.keys()):
        for count,word in enumerate(test_data[lang]):
            # cross checking if a word from test data is in existing vocabulary
            if word not in vocab[lang[:-3]+"tra"]:
                test_data[lang][count] = "<UNK>"  # Replaced with <UNK> if word is new
    return test_data


# This function accepts input directory path as an argument,
# Opens the file and returns a tokenized list of text read from the file
def read_files(file_location):
    # To tokenize the languate text and return it in a list
    with open(file_location,'r') as x:
        lines = x.read().splitlines()   # List containing each line read from file location
    lang_text = ""
    for line in lines:
        lang_text += line + " "  # we add the space to ensure distinct words are separated properly
    lang_text = lang_text.split()
    return lang_text


# This function accepts output directory path and the output data dict. as arguments,
# Opens the output file and writes the output data to the csv file.
def output_file(output_location, data_out):

    # List containing the headers of our output file
    titles = ["Testing_file", "Training_file","Perplexity", "n"]

    # To open the output csv file in write mode
    with open(output_location, 'w', encoding='UTF8', newline='') as f:

        # Returns a writer object 
        writer = csv.writer(f)

        # write the header line to the file
        writer.writerow(titles)

        # For loop to iterate through all key, value pairs in the output data dict.
        for key,value in sorted(data_out.items()):
            row = []  # List to store the next row to be written

            
            # To append all the required details in the right order to the row list 
            row.append(str(key)) 
            for item in value:
                row.append(str(item))

            # To write the row to the output file
            writer.writerow(row)
            # To empty the row list to collect details for next row
            row = []


def main():
    # To create an argument parser object
    parser = argparse.ArgumentParser()

    # To add arguments to the parser object
    parser.add_argument("train_data_location", help="Please provide the location of the train data file")
    parser.add_argument("test_data_location", help="Please provide the location of the test data file")
    parser.add_argument("output_location", help="Please provide the location of the output file")
    parser.add_argument("model_type", help="Please choose a model from;  --unsmoothed, --laplace , or  --interpolation")
    parser.add_argument("-v", "--verbosity", help="Increase output verbosity", action= "store_true" )

    # parses arguments through the parse_args() method. This will inspect the command line, 
    # convert each argument to the appropriate type and then invoke the appropriate action.
    args = parser.parse_args()
    
    train_data_location = args.train_data_location  # Variable to store input directory path
    output_location = args.output_location  # Variable to store output file path 
    test_data_location = args.test_data_location  # Variable to store output file path 
    model_type = args.model_type  ## Variable to store selected model type 

    if args.verbosity:  # Helps the user check what input and output file paths they have provided
        print(f"the location of the  training data is {train_data_location} ")
        print(f"the location of the test data is {test_data_location} ")
        print(f"the location of the output file is {output_location} ")
        print(f"the type of model specified to train is {model_type} ")

    train_data = {}  # Dictionary to store all lines read from each input file
    # To extract the file path for each txt file in the provided folder path
    for file in os.listdir(train_data_location):
        if file.endswith(".txt.tra"):
            file_location = os.path.join(train_data_location,file)
            lines_read = read_files(file_location)  # Function call to read each input file in folder
            # File name and read lines are added to the dict. as a key-value pair
            train_data[file] = lines_read 

    test_data = {}  # Dictionary to store all lines read from each input file
    # To extract the file path for each txt file in the provided folder path
    for file in os.listdir(test_data_location):
        if file.endswith(".txt.dev"):
            file_location = os.path.join(test_data_location,file)
            lines_read = read_files(file_location)  # Function call to read each input file in folder
            # File name and read lines are added to the dict. as a key-value pair
            test_data[file] = lines_read 

    # Function call to obtain a vocabulary for training data set
    vocab = create_vocab(train_data)

    # Function call to test the above vocab using the test data set
    test_vocab(test_data, vocab)
    
    deleted_interpolation(train_data,3)
    # If the user selected the unsmoothed model type
    if model_type == "unsmoothed":
        n = 1  # Since we only need the unigrams
        prob = train_model_unsmoothed(train_data, n)
        test_results = test_model_unsmoothed(test_data, prob, n)

    # If the user selected the laplace model type   
    elif model_type == "laplace":
        n = 2  # We need both the bigrams and the unigrams
        prob = train_model_laplace(train_data, vocab, n)
        test_results = test_model_laplace(test_data, train_data, vocab, prob, n)

    # If the user selected the interpolation model type
    elif model_type == "interpolation":
        n = 3
        lambdas = deleted_interpolation(train_data, n)  # We need the trigrams, bigrams and the unigrams
        prob = train_model_interpolation(train_data,lambdas,n)
        test_results = test_model_interpolation(test_data, train_data, vocab, prob, lambdas,n)
        

    # If the user did not specify any model type, we use the unsmoothed model type
    else:
        n = 1
        prob = train_model_unsmoothed(train_data,1)
        test_results = test_model_unsmoothed(test_data, prob, 1)
    
    # Function call to write test results to output file
    output_file(output_location, test_results)

main()