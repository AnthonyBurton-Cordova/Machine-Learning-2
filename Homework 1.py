import matplotlib.pyplot as plt
from collections import Counter

### Please fill in an explanation of each function and an example of how to use it below.

## clear() - Removes every item in the counter, returning an empty counter
# Creating a list of digits to count
list1 = [1,1,1,1,4,6,7,7,7,6,3,4,8,9,0,7,6,9,9,9,6,3,2,0]
list1

# Check how many of each digit are in list1 using Counter()
counter1 = Counter(list1)
counter1

# clear the counter using the .clear() function
counter1.clear()
counter1

## copy() - Copies every item in the counter and returns each copied element in dict form
# Start with list1 and use the counter function
list1
counter1 = Counter(list1)
counter1

# Use the copy function to create a copy of the first counter into a new counter
counter_copy = counter1.copy()
counter_copy

## 	elements() - Returns a data structure composed of several instances of each unique value in the counter.
## The number of instances of each value is equal to their value count
# Start with list1 and use counter
list1
counter1 = Counter(list1)
counter1

# use the elements() funtion to get the elements in the counter
# Use a for loop to print all the elements in the counter
elements_in_counter = counter1.elements()
for i in elements_in_counter:
    print(i)

### get() - Retrieves the value corresponding to the specified key
# Start with list1 and use the counter
list1
counter1 = Counter(list1)
counter1
# use the get() function to see how many elements are in the counter
get_1_count = counter1.get(1)
get_1_count
print("there are", get_1_count, "counts of 1 in list1")

### items() - Returns all key/value pairs in a counter
# using counter1, find the key/value pairs using items()
counter1
items_in_counter1 = counter1.items()
items_in_counter1
print("The Key/value pairs in counter1 are", items_in_counter1)

### most_common() - Returns a list of tuples where each tuple contains both the key and the value of the counter.
### The list is sorted in descending order based on the value of the key
# using counter1, create a sorted list of the value of key
counter1
most_common_in_list1 = counter1.most_common()
most_common_in_list1

### pop() - Removes the key-value pair corresponding to the input key and returns the corresponding value
# using count1, pop a key and return the value
counter1
key_value_for_7 = counter1.pop(7)
key_value_for_7

###	setdefault() - Returns the value corresponding with the input key. 
### If the key does not exist, insert the key into the Counter and 
### assign it a value corresponding with the second method input
# using counter1, set a new default value for a key and insert a new key with a defauly value
counter1
value_9 = counter1.setdefault(9)
value_A = counter1.setdefault('A', 60)

# since A was not in the originally in the counter, it is added with a value of 60
print("The value for 9 is", value_9)
print("The value for new key A is", value_A)
print("the updated counter including A is", counter1)

from itertools import *
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']


#### Q2
### 1. Build your own counter object, then use the built-in Counter() and confirm they have the same values.
order_counter = Counter(flower_orders)
order_counter


### 2. Count how many objects have color W in them.
count_with_w = sum('W' in order for order in flower_orders)
count_with_w

### 3. Make histogram of colors
# Count the occurrences of each element
element_counts = Counter(flower_orders)
elements = list(element_counts.keys())
counts = list(element_counts.values())

# Plotting the histogram
plt.bar(elements, counts, color='skyblue')
plt.xlabel('Flower Orders')
plt.ylabel('Count')
plt.title('Histogram of Flower Orders')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

# Hint from JohnP - Itertools has a permutation function that might help with these next two.
### 4. Rank the pairs of colors in each order regardless of how many colors are in an order.
# Rank the color pairs based on counts
ranked_color_pairs = element_counts.most_common()
ranked_color_pairs

# Display the ranked color pairs
print("Ranked Color Pairs:")
for rank, (color_pair, count) in enumerate(ranked_color_pairs, start=1):
    print(f"{rank}. {color_pair}: {count}")

### 5. Rank the triplets of colors in each order regardless of how many colors are in an order.
three_color_orders = [order for order in flower_orders if order.count('/') == 2]
three_color_orders

count_of_triplets = Counter(three_color_orders)
count_of_triplets

### 6. Make a dictionary with key=”color” and values = “what other colors it is ordered with”.
# Create a defaultdict to store the colors and their associated colors
from collections import defaultdict
color_dict = defaultdict(list)
color_dict

# Iterate through each flower order
for order in flower_orders:
    colors = order.split('/')
    
    # Iterate through each color
    for color in colors:
        other_colors = [c for c in colors if c != color]
        color_dict[color].extend(other_colors)

# Convert defaultdict to a regular dictionary
color_dict = dict(color_dict)

# Display the dictionary
print("Color Dictionary:")
for color, associated_colors in color_dict.items():
    print(f"{color}: {', '.join(associated_colors)}")

### 7. Make a graph showing the probability of having an edge between two colors based on how often they co-occur.  (a numpy square matrix)
# Extract color pairs from each order
import numpy as np
color_pairs = [tuple(order.split('/')) for order in flower_orders]

# Count the occurrences of each color pair
color_pair_counts = Counter(color_pairs)

# Create a numpy square matrix with zeros
num_colors = len(set(color for pair in color_pairs for color in pair))
color_matrix = np.zeros((num_colors, num_colors))

# Populate the matrix with co-occurrence probabilities
for i, color_i in enumerate(set(color for pair in color_pairs for color in pair)):
    for j, color_j in enumerate(set(color for pair in color_pairs for color in pair)):
        pair_count = color_pair_counts.get((color_i, color_j), 0)
        total_count_i = sum(color_pair_counts.get((color_i, c), 0) for c in set(color for pair in color_pairs for color in pair))
        total_count_j = sum(color_pair_counts.get((c, color_j), 0) for c in set(color for pair in color_pairs for color in pair))
        
        # Avoid division by zero
        if total_count_i == 0 or total_count_j == 0:
            color_matrix[i, j] = 0.0
        else:
            color_matrix[i, j] = pair_count / max(total_count_i, total_count_j)

# Display the color matrix
print("Co-occurrence Probability Matrix:")
print(color_matrix)

# Plot the color matrix
plt.imshow(color_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Probability')
plt.xticks(np.arange(num_colors), sorted(set(color for pair in color_pairs for color in pair)))
plt.yticks(np.arange(num_colors), sorted(set(color for pair in color_pairs for color in pair)))
plt.title('Co-occurrence Probability Matrix')
plt.show()

### 8. Make 10 business questions related to the questions we asked above.
# 1. What specific color combinations are most popular among customers?
# 2. Can we streamline supply to account for the most popular flower colors?
# 3. Are there specific colors that are more popular during certain seasons?
# 4. Can we use customer input to refine product offerings and help elevate less popular color combinations?
# 5. Are there gaps in the market that we can exploit by introducing unique color combinations?
# 6. Can we change our price strategy to elevate less popular combinations?
# 7. Can we use the same idea to boost profit on our popular flowers?
# 8. Are there inventory waste trends that can be analyzed by looking at the color combinations?
# 9. If there different stores geographically, can we segment those and look at popular combinations by region?
# 10. Can we break these segments down by time to predict what color combinations will be popular in the future?




#### Q3
dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']

dead_men_tell_tales

### Join everything
text_joined = ','.join(dead_men_tell_tales)
text_joined

### Remove spaces
text_with_no_spaces = text_joined.replace(' ', '')
text_with_no_spaces

### Occurrence probabilities for letters
# Count the occurrences of each letter
letter_counts = Counter(text_with_no_spaces) 
letter_counts

# Calculate the total number of letters
total_letters = sum(letter_counts.values())

# Calculate occurrence probabilities for each letter
letter_probabilities = {letter: count / total_letters for letter, count in letter_counts.items()}

# Display occurrence probabilities
print("Occurrence Probabilities for Letters:")
for letter, probability in letter_probabilities.items():
    print(f"{letter}: {probability:.4f}")

    
### Tell me transition probabilities for every pair of letters
# Create pairs of consecutive letters
letter_pairs = [text_with_no_spaces[i:i+2].lower() for i in range(len(text_with_no_spaces) - 1)]
letter_pairs

# Count the occurrences of each letter pair
letter_pair_counts = Counter(letter_pairs)

# Calculate the total number of letter pairs
total_letter_pairs = sum(letter_pair_counts.values())

# Calculate transition probabilities for each letter pair
letter_pair_probabilities = {pair: count / total_letter_pairs for pair, count in letter_pair_counts.items()}

# Display transition probabilities
print("Transition Probabilities for Letter Pairs:")
for pair, probability in letter_pair_probabilities.items():
    print(f"{pair}: {probability:.4f}")

### Make a 26x26 graph of 4. in numpy
# Create a 26x26 matrix 
letter_matrix = np.zeros((26, 26))

# Populate the matrix with transition probabilities
for i, char_i in enumerate('abcdefghijklmnopqrstuvwxyz'):
    for j, char_j in enumerate('abcdefghijklmnopqrstuvwxyz'):
        pair = char_i + char_j
        letter_matrix[i, j] = letter_pair_counts.get(pair, 0) / max(letter_counts[char_i], 1)

# Display the matrix
print("Transition Probability Matrix:")
print(letter_matrix)

### plot graph of transition probabilities from letter to letter
# Plot the matrix
plt.imshow(letter_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Probability')
plt.xticks(np.arange(26), list('abcdefghijklmnopqrstuvwxyz'))
plt.yticks(np.arange(26), list('abcdefghijklmnopqrstuvwxyz'))
plt.title('Transition Probability Matrix')
plt.show()





