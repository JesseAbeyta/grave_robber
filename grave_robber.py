# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:46:46 2018

@author: Jesse Abeyta
"""

import nltk
import pronouncing 
from pronouncing import grade_rhyme
import json
import random
import requests
import ast
# for conjugating verbs
import realizer

# For making plurals
import inflect

from functools import reduce

inflector = inflect.engine()


# Holds the grammatical skeleton for later analysis
skelly = None

# dict of syllable count of most words
syllables_dict = None
with open("syllable_dict.json") as f:
    syllables_dict = json.loads(f.read())

POS_exceptions_dict = {"here": "here", "is": "is", "'s": "'s", "'re": "'re", "was": "was", "were": "were",
                       "am": "am", "are": "are", "then": "then"}

# Class that holds functions/variables relating to composing a poem
class Composition:
    def __init__(self):
        self.source_text = None
        self.POS_skeleton = None
        self.syllable_skeleton = None
        self.syllables_by_line = []
        self.choice_skeleton = [] # To recover from morph failures
        self.rhyme_structure = None
        self.rhyme_dict = {}
        self.num_lines = None
        self.composition = []
        self.composition_syllables = []
        
    # Loads a random poem and builds a skeleton from it
    def load_poem(self, echo = True):
        skeletons = rand_gr()
        self.POS_skeleton = skeletons[1]
        self.source_text = skeletons[0]
        self.syllable_skeleton = syllable_analysis(self.source_text)
        self.count_line_syllables()
        self.num_lines = len(self.POS_skeleton)
        self.build_rhyme_skeleton()

    def compose_line(self, seed_word, line_num, print_progress = False):
        
        # Adjustable parameters for what order to prioritize aspects of poem
        priority_list = ["syllables", "POS"]
        
        # Back up in case WAN fails
        # !!! not currently implemented
        line_seed = seed_word
        
        # How many syllables you can afford to use for each slot as a maximum
        syllable_leeway = self.calc_reducability(line_num)
        
        # How many syllables it's ideal to use in each slot
        ideal_syllables = self.syllable_skeleton[line_num][:]
        
        # Determines the position of the last word in the POS skeleton
        rhyme_position = len(self.syllable_skeleton[line_num]) - 1
        
        if self.syllable_skeleton[line_num][rhyme_position] == 0:        
            # If the last slot is punctuation, shift back until you find a word
            j = rhyme_position
            while self.syllable_skeleton[line_num][j] == 0:
                j -= 1
            rhyme_position = j
        
        # Determines whether there's anything to rhyme with
        if self.rhyme_structure[line_num] in self.rhyme_dict:
            free_rhyme = False
        else:
            free_rhyme = True
            
        
        # List showing the POS for a line
        line_POS = self.POS_skeleton[line_num]
        line = []
        choice_line = [] # To recover from morph failures
        for i, token_POS in enumerate(line_POS):
            # Weird character filter list
            if token_POS in ["``", "''"]:
                continue
            # If the token is punctuation, just append it and move on
            if len(token_POS) == 1 and token_POS not in ['a', 'o', 'i']:
                line.append(token_POS)
                choice_line.append(token_POS)
                continue
            # If the association network is to be used
            if get_by_assoc(token_POS):
                # Prevents failure spiral of no result
                #while True:
                
                # Calculate the maximum number of syllables you could use here
                max_syllables = syllable_leeway[i] + self.syllable_skeleton[line_num][i]
                
                # !!! dirty hack fix later !!!
                if max_syllables < 1:
                    max_syllables = 1
                
                # seed_word is used in WAN, result is morph of said word
                # Rhyme is currently not implemented, do later !!!
                '''
                seed_word, result = choose_assoc_custom(seed_word, token_POS, max_syllables, "")
                '''
                temp_priority_list = ["syllables", "POS"]
                # If you need to rhyme, determine with what
                rhyme = ""
                if i == rhyme_position and not free_rhyme:
                    rhyme = self.rhyme_dict[self.rhyme_structure[line_num]]
                    temp_priority_list = priority_list
                
                seed_word, result = choose_assoc_custom(
                        seed_word,
                        token_POS,
                        max_syllables,
                        ideal_syllables[i],
                        rhyme,
                        temp_priority_list)
                line.append(result)
                choice_line.append(seed_word)
                
            # If the token is a pronoun and it's not the last one on the line
            elif token_POS == "PRP" and (i < (len(line_POS) - 1)):
                result = pronoun_agreer(line_POS[i + 1])
                # If the next token is a verb, make the pronoun agree
                if result:
                    line.append(result[0])
                    choice_line.append(result[0])
                    result = result[0]
                # !!! Need to work syllables into rand_grammer choices
                # Else, pick a random pronoun
                else:
                    result = rand_grammar(token_POS)[0]
                    line.append(result)
                    choice_line.append(result)
            else:
                result = rand_grammar(token_POS)[0]
                line.append(result)
                choice_line.append(result)
        
            # Update the leeway array
            # Calculate the difference between what the skeleton recommended
            # and what you chose
            syllable_balance = count_syllables(result) - self.syllable_skeleton[line_num][i]
            map(lambda x: x - syllable_balance, syllable_leeway)
            
            # Shift the ideal/target number of syllables for all slots
            map(lambda x: x - syllable_balance, ideal_syllables)
        
        # If the rhyme for this pattern hasn't been established
        if free_rhyme:
            # Record the last word in the line for rhyming reasons
            lw = line[rhyme_position]
            self.rhyme_dict[self.rhyme_structure[line_num]] = lw
        
        if print_progress:
            poetry_print([line])
        # Save the newly composed line
        self.composition[line_num] = line
        self.choice_skeleton[line_num] = choice_line
        
        # Return the last associatable word used as the seed for the next line
        return seed_word

    # Composes a poem based off the provided seed word, resulting variables are
    # stored in the composition class
    def compose_poem(self, seed_word, print_when_done = True):
        # Prepare an empty list to hold the composition
        self.composition = []
        self.choice_skeleton = []
        self.rhyme_dict = {}
        for i in range(0, self.num_lines):
            self.composition.append([])
            self.choice_skeleton.append([])
            
            
        for i in range(0, self.num_lines):
            seed_word = self.compose_line(seed_word, i)
        poetry_print(self.composition)
        
        self.count_comp_syllables()
        
        
    # Counts the syllables in a finished composition
    def count_comp_syllables(self):
        punctuation = [".", ",", "'", '"', "!", "?", "``", ">", "<", ";", ":", "--", "-"]
        self.composition_syllables = []
        # Analyse the produced poem
        for line in self.composition:
            count = 0
            for word in line:
                if word in punctuation:
                    continue
                count += count_syllables(word)
            self.composition_syllables.append(count)
    
    # Allows manual editing of the source syllable skeleton
    def fix_syllable(self, line, token, count):
        # Fix syllable count
        self.syllable_skeleton[line][token] = count
        # Update line counts list
        self.count_line_syllables()
    
    def count_line_syllables(self):
        self.syllables_by_line = []
        # Counts the syllables in each line
        for line in self.syllable_skeleton:
            self.syllables_by_line.append(reduce(lambda x, y: x + y, line))
    
    # Extracts the base poem's rhyme scheme
    def build_rhyme_skeleton(self):
        self.rhyme_structure = []
        
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        punctuation = [".", ",", "'", '"', "!", "?", "``", ">", "<", ";", ":", "--", "-"]
        contractions = ["'d", "'s", "'t", "'re"]       
        
        # !!! set up code to put contractions into the right pronunciation
        
        # Collect all the last words of each line
        last_words = []
        for line_num, line in enumerate(self.source_text):
            i = -1
            last_word = line[i]
            if last_word in punctuation:
                while last_word in punctuation:
                    i -= 1
                    last_word = line[i]
            # Collapse contractions onto the end of the previous word for rhyme
            if last_word in contractions:
                last_word = line[i - 1] + line[i]
            last_words.append([last_word, "", self.POS_skeleton[line_num][i]])
            
        # See which ones rhyme
        for word1 in last_words:
            # If a rhyme was already found
            if word1[1] != "":
                continue
            # label all rhyming words with the same letter
            word1[1] = labels[0]
            for word2 in last_words:
                if word2[1] != "":
                    continue
                score = grade_rhyme(word1[0], word2[0])
                if score > 0:
                    word2[1] = labels[0]
            
            # this label is done being used
            labels.pop(0)
            
        self.rhyme_structure = [x[1] for x in last_words]
        
        # Some parts of speech have few options, lock those in now
        # Only one option, highest priority
        for word in last_words:
            if word[2] in POS_exceptions_dict:
                self.rhyme_dict[word[1]] = word[0]
        

    # Determines how much syllable "surplus" there is at each slot in a line.
    # Surplus is the number of syllables that can be potentially 
    def calc_reducability(self, line_num):
        reducable_arr = []
        leeway_total = 0
        for token_count in reversed(self.syllable_skeleton[line_num]):
            if token_count > 1:
                leeway_total = leeway_total + (token_count - 1)
            reducable_arr.append(leeway_total)
        return reducable_arr[::-1]
    
    def display(self):
        poetry_print(self.composition)
    
# Given a list of lines (line = list of strings), prints them in a mostly
# formatted fashion
def poetry_print(frame):
    for line in frame:
        for i, word in enumerate(line):
            # Don't add spacing if the next token is punctuation
            if i != 0:
                word = word.lower()
            else:
                word = word.capitalize()
            print(word, end = "")
            if i < (len(line) - 1):
                if line[i + 1] not in [",", ".", "!", "?", ":", ";", "'d", "'re", "'s"]:
                    print(" ", end="")
        print()
    

# Given a part of speech, returns a random word of that POS
def rand_grammar(POS, k = 1):
    if POS in POS_exceptions_dict:
        return [POS]
    if POS == "TO":
        return ["to"]
    if POS == "EX":
        return ["there"]
    if POS == "POS":
        return ["'s"]
    if POS == "here":
        return ["here"]
    
    # Do not change any forms of "to be"
    if POS in ["is", "was", "being", "were", "am", "are"]:
        return [POS]
    
    with open("grammar_probabilities.json") as f:
        POS_dict = json.loads(f.read())
        words = []
        weights = []
        for word in POS_dict[POS]:
            words.append(word)
            weights.append(POS_dict[POS][word])
        if k == 1:
            random.choices(words, weights)[0]
        return random.choices(words, weights, k=k)

# !!! ADD PROBABILITIES TO SELECTION LATER
# Given a part of speech for a following word, give a pronoun that agrees with it
def pronoun_agreer(POS):
    if POS == "VB":
        return random.choices(["I", "me", "you", "they"])
    if POS == "VBD":
        return random.choices(["I", "you", "he", "she", "they", "we"])
    if POS == "VBP":
        return random.choices(["I", "you"])
    if POS == "VBZ":
        return random.choices(["he", "she"])
    if POS == "'s":
        return random.choices(["he", "she", "it"])
    if POS == "'re":
        return random.choices(["we", "they"])
    if POS == "was":
        return random.choices(["I", "he", "she"])
    
    # Returns false if POS given can't be agreed with
    return False


# Given a word, returns a list of associations across all parts of speech
# ordered by weight
def get_assoc(word, POS=None):
    key = ""
    site = "https://api.wordassociations.net/associations/v1.0/json/search?"
    request = "apikey=" + key + "&text=" + word + "&lang=en&limit=300"
    if POS is not None:
        request = request + "&pos=" + POS
    
    response = requests.get(site + request)
    thing = ast.literal_eval(bytes.decode(response.content))
    return thing['response'][0]['items']


# New and improved version of WAN chooser
# !!! FINISH LATER
# !!! Implement some form of check to prevent it from doubling a word
def choose_assoc_custom(seed_word, POS, max_num_syllables, ideal_syllables, rhyme, priority_list):
    # Convert between upenn POS labels to wordassociation net POS labels
    pos_dict = {
            "VB": "verb",
            "VBG": "verb",
            "VBN": "verb",
            "VBP": "verb",
            "VBZ": "verb",
            "VBD": "verb",
            "NN": "noun",
            "NNS": "noun",
            "NNP": "noun",
            "NNPS": "noun",            
            "JJ": "adjective",
            "JJR": "adjective",
            "JJS": "adjective",
            "RB": "adverb",
            "RBR": "adverb",
            "RBS": "adverb"
                }
    pos2 = pos_dict[POS]
    
    # Just trust WAN until POS conversion is implemented !!!
    candidates = get_assoc(seed_word, POS= pos2)
    
    candidates = [[x["item"], x['weight']] for x in candidates]
    # Filter out everything that's not of the part of speech we want
    def filter_POS(word):
        if pos2 == "adverb":
            if nltk.pos_tag(word) == POS:
                return True
            else:
                return False
        else:
            return True
    
    # Filter everything with too many syllables
    def filter_syllables(word):
        if count_syllables(morph(word, POS)) <= max_num_syllables:
            return True
        else:
            return False
    
    # If rhymes don't matter, don't do anything
    if rhyme == "":
        def filter_rhymes(word):
            return True
    else:
        # Filter out everything that doesn't rhyme
        def filter_rhymes(word):
            grade = grade_rhyme(morph(word, POS), rhyme)
            if grade == 0:
                return False
            else:
                return True
    
    filter_dict = {"POS": filter_POS, "syllables": filter_syllables, "rhyme": filter_rhymes}
    
    # Filter by each criteria in order of priority. If no candidates are left
    # after a filtering, don't apply that filter
    for criteria in priority_list:
        # Apply the criteria to the word, not the score
        filtered_cands = list(filter(lambda x: filter_dict[criteria](x[0]), candidates))
        # If there is at least one candidate remaining, use this filter
        if filtered_cands:
            candidates = filtered_cands
    
    def weight_syllables(word):
        count = count_syllables(morph(word, POS))
        if count == ideal_syllables:
            return 1
        # Going over target is worse than going under target
        if count > ideal_syllables:            
            multiplier = (1 - (.1 + .15 * (count - ideal_syllables)))
            if multiplier <= 0:
                return .01
            else:
                return multiplier
        else:
            multiplier = (1 - (.05 * (ideal_syllables - count)))
            if multiplier <= 0:
                return .01
            else:
                return multiplier
            
    def weight_rhymes(word):
        if rhyme == "":
            return 1
        grade = grade_rhyme(morph(word, POS), rhyme)
        if grade >= 3:
            return 5
        elif grade == 2:
            return 3
        else:
            return 0.5
        
        
    # Apply weights to choices
    map(lambda x: x[1] * weight_syllables(x[0]) * weight_rhymes(x[0]), candidates)
    
    weights = [x[1] for x in candidates]
    words = [x[0] for x in candidates]
    
    choice = None
    try:
        choice = random.choices(words, weights)[0]
    except:
        print("WAN failed on: " + seed_word + "\tPOS: " + POS)
            
    morphed = morph(choice, POS)    
    return choice, morphed


# Random grave rob, takes a random poem between 4 and 16 lines in length, and
# then returns its source text and POS skeleton
def rand_gr(echo = True):
    with open("corpus_finalis.txt") as f:
        poems_dict = json.loads(f.read())
        poems_keys = list(poems_dict.keys())
        
        # Try poems until you find one of the right length
        poem = None
        while True:
            poem = poems_dict[random.choice(poems_keys)]
            if poem['len'] < 16 and poem['len'] > 4:
                break
            else:
                print("invalid poem")
        if echo:
            print(poem['text'])
        return grave_rob(poem['text'])
        
# Given a poem, create a syllabic skeleton for it
def syllable_analysis(poem):
    dont_count_list = [",", ".", "!", "?", ":", ";", "'re", "'d", "'s", "'ll", "``", "''", "--"]
    
    # Holds the syllabic structure of the poem
    skeleton = []
    for line in poem:
        # List to keep track of syllables in current line
        line_list = []
        for word in line:
            # If the token actually has syllables to be counted
            if word not in dont_count_list:
                line_list.append(count_syllables(word))
            else:
                line_list.append(0)
                
        # Add the syllabic structure for the just completed line
        skeleton.append(line_list)
        #skeleton.append(reduce(lambda x, y: x + y, line_list))
    return skeleton
    
# Create a POS skeleton for a poem
def grave_rob(poem):
    skeleton = my_POS_tagger(poem)
    
    return skeleton

# !!! maybe should rewrite to be dictionary rather than function
# Returns true if the given POS should be found using the association network,
# false otherwise
def get_by_assoc(POS):
    # If the part of speech is in this dict, don't use WAN
    dont_assoc_dict = {
        "is": "is", "was": "was", "were": "were", "am": "am", "are": "are", "then": "then",
        "'s": "'s", "'re": "'re",
        "here": "here",
        "POS": "possessive ending",
        "EX": "existential there",
        "CC": "Coordinating conjunction",
        "CD": 	"Cardinal number",
        "DT": 	"Determiner",
        "IN": 	"Preposition or subordinating conjunction",
        "LS": 	"List item marker",
        "MD": 	"Modal",
        "PDT": 	"Predeterminer",
        "PRP": 	"Personal pronoun",
        "PRP$": 	"Possessive pronoun",
        "RP": 	"Particle",
        "SYM": 	"Symbol",        
        "UH": 	"Interjection",        
        "WDT": 	"Wh-determiner",
        "WP": 	"Wh-pronoun",
        "WP$": 	"Possessive wh-pronoun",
        "WRB": 	"Wh-adverb",
        "TO": "to"
    }
    if POS in dont_assoc_dict:
        return False
    else:
        return True

# Conjugates verbs using Nick Montfort's curveship realizer code.
def conjugate(word, POS):
    if POS in ["VB"]:
        return word
    
    # !!! What do I put in "time" parameter?
    vb = realizer.Verb(word, "")
    if POS == "VBZ":
        return vb.third_person_singular()
    elif POS == "VBP":
        if word == 'be':
            return 'am'
        return word
    elif POS == "VBN":
        return vb.past_participle()
    elif POS == "VBD":
        return vb.preterite()
    elif POS == "VBG":
        return vb.present_participle()


# Returns two lists of lists in a list, first is the text of the poem, second is
# The grammatical skeleton of the poem. Both are broken up by line
def my_POS_tagger(poem):
    # Remember the length of each line for when puting the poem back together
    line_lengths = [len(nltk.word_tokenize(line)) for line in poem]
    
    # Put the poem all on one line so the POS tagger has more context info.
    single_string_poem = " ".join(poem)
    
    token_list = nltk.word_tokenize(single_string_poem)
    
    # Convert pos_tag tuples to lists of two elements so they can be modified
    tag_list = [list(x) for x in nltk.pos_tag(token_list)]
    
    for tag_pair in tag_list:
        # each tag pair is formatted like ["dog", "NN"]
        # Substitute in my exceptions for specific words
        if tag_pair[0].lower() in POS_exceptions_dict:
            tag_pair[1] = POS_exceptions_dict[tag_pair[0].lower()]
        
    POS_skeleton = []
    text_skeleton = []
    # Reconstruct lines
    for length in line_lengths:
        line = []
        for i in range(0, length):
            line.append(tag_list.pop(0))
        text_skeleton.append([x[0] for x in line])
        POS_skeleton.append([x[1] for x in line])
        
    return [text_skeleton, POS_skeleton]
    
    
# Returns how many syllables are in a word
def count_syllables(word, print_fails = True):
    phones = pronouncing.phones_for_word(word)[0]
    return pronouncing.syllable_count(phones)
        
def add_to_syllable_dict(word, count):
    syllables_dict[word] = count
    with open("syllable_dict.json", mode = "w") as f:
        f.write(json.dumps(syllables_dict))
        
# Generic function that pluralizes nouns and conjugates verbs as needed
def morph(word, POS):
    if word is None:
        return None
    if POS in ["VB", "VBG", "VBN", "VBP", "VBZ", "VBD"]:
        return conjugate(word, POS)
        
    elif POS == "NNS":
        return inflector.plural(word)
    
    else:
        return word
        
    