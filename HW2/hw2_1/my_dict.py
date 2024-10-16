import json
import re


def parse_sentence(sentence):
    pattern = r'\w+|\S'
    tokens = re.findall(pattern, sentence.lower())
    return tokens


def create_word_dictionary(file_path):
    max_length = 0
    word_dict = {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    for item in data:
        caption = item['caption']
        for sentence in caption:
            tokens = parse_sentence(sentence)
            if len(tokens) > max_length:
                max_length = len(tokens)
            for token in tokens:
                if token not in word_dict:
                    word_dict[token] = len(word_dict)

    added_words = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    for word in added_words:
        if word not in word_dict:
            word_dict[word] = len(word_dict)
    return word_dict, max_length


training_file_path = 'MLDS_hw2_1_data/training_label.json'
training_word_dict, max_length = create_word_dictionary(training_file_path)
# Save the word dictionary to a file
with open('word_dict.json', 'w') as file:
    json.dump(training_word_dict, file)
print(
    f'The word dictionary of {len(training_word_dict)} words has been saved to word_dict.json. \nThe maximum length of a sentence is {max_length}.')